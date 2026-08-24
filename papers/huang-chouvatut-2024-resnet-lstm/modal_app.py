"""Modal entry points for the clean-room ResNet-18 + LSTM reproduction."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

import modal


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[1]
DATASET_SLUG = "lsa64"
DATASET_URL = "https://drive.google.com/file/d/1C7k_m2m4n5VzI4lljMoezc-uowDEgIUh/view?usp=sharing"
RESULTS_VOLUME = "huang-chouvatut-2024-results"
app = modal.App("huang-chouvatut-2024-resnet-lstm")
# The study-wide image supplies simple-video-utils; this paper only adds its code.
base_image = modal.Image.from_dockerfile(REPOSITORY_ROOT / "Dockerfile", context_dir=REPOSITORY_ROOT)
image = base_image.add_local_file(ROOT / "train.py", "/app/train.py")
data_image = base_image.pip_install("gdown==5.2.0")
datasets = modal.Volume.from_name("datasets", create_if_missing=False)
cache = modal.Volume.from_name("huggingface-cache", create_if_missing=False)
results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    for member in archive.infolist():
        member_path = PurePosixPath(member.filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"unsafe archive entry: {member.filename}")
    archive.extractall(destination)


@app.function(
    image=data_image,
    volumes={"/datasets": datasets, "/cache/huggingface": cache},
    timeout=2 * 60 * 60,
    cpu=2,
    env={"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"},
)
def populate_lsa64() -> dict[str, object]:
    """Idempotently populate the approved original raw LSA64 release."""
    root = Path("/datasets") / DATASET_SLUG
    manifest = root / "manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "source.zip"
    subprocess.run(["gdown", "--fuzzy", "--output", str(archive), DATASET_URL], check=True)
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise ValueError("LSA64 archive failed ZIP integrity check")
        safe_extract(zipped, root)
    videos = sorted(root.rglob("*.mp4"))
    if len(videos) != 3200:
        raise ValueError(f"expected 3200 LSA64 videos after extraction, found {len(videos)}")
    manifest_data = {
        "source_url": DATASET_URL,
        "source_archive_sha256": sha256(archive),
        "video_count": len(videos),
        "relative_paths_sha256": hashlib.sha256("\n".join(str(path.relative_to(root)) for path in videos).encode()).hexdigest(),
    }
    archive.unlink()
    manifest.write_text(json.dumps(manifest_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    datasets.commit()
    return manifest_data


@app.function(
    image=image,
    gpu="A10G",
    cpu=8,
    timeout=24 * 60 * 60,
    volumes={"/datasets": datasets, "/cache/huggingface": cache, "/results": results},
    env={"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub", "TORCH_HOME": "/cache/huggingface/torch"},
)
def train() -> str:
    """Run the Table 3 epoch-30/batch-16 reconstruction once."""
    output_dir = Path("/results/resnet18-lstm-inferred-split")
    if (output_dir / "run.json").exists():
        return (output_dir / "run.json").read_text(encoding="utf-8")
    if output_dir.exists():
        raise FileExistsError(f"incomplete output directory: {output_dir}")
    # Paper Table 3 reports the 30-epoch, batch-16 row. The remaining
    # command-line values are documented reconstruction decisions in README.
    subprocess.run([
        "python", "/app/train.py", "--data-root", "/datasets/lsa64", "--output-dir", str(output_dir), "--epochs", "30", "--batch-size", "16", "--workers", "8", "--seed", "2024",
    ], check=True)
    results.commit()
    return (output_dir / "run.json").read_text(encoding="utf-8")
