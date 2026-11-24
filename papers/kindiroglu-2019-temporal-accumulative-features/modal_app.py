"""Modal jobs for Kindiroglu et al. 2019, Table 5 (Temporal Accumulative Features).

populate:        mirror the BosphorusSign22k Google Drive folder into the shared datasets
                 Volume, verify it against Drive's MD5s and write a SHA-256 manifest.
populate_public: download the public hand-segmentation datasets (HGR2A, HGR2B, Rendered
                 Handpose Dataset) from their official pages into the same Volume.
"""

import modal

PAPER = "13fbc667"
SLUG = "bosphorussign22k"
DATA_ROOT = f"/datasets/{SLUG}"
RCLONE = "v1.76.0"

app = modal.App(f"taf-{PAPER}")
datasets = modal.Volume.from_name("datasets", version=2)
cache = modal.Volume.from_name("huggingface-cache", version=2)

CACHE_ENV = {"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"}
populate_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl")
    .run_commands(
        f"curl -fsSLo /tmp/rclone.deb https://downloads.rclone.org/{RCLONE}/rclone-{RCLONE}-linux-amd64.deb",
        "dpkg -i /tmp/rclone.deb && rm /tmp/rclone.deb",
    )
    .env(CACHE_ENV)
)


@app.function(
    image=populate_image,
    volumes={"/datasets": datasets, "/cache/huggingface": cache},
    # Read-only Drive login for the folder, as RCLONE_CONFIG_GDRIVE_* variables.
    secrets=[modal.Secret.from_name("gdrive-rclone")],
    cpu=4,
    memory=8192,
    timeout=24 * 3600,
)
def populate(depth: bool = False):
    """Safe to re-run: files already copied are skipped.

    The 360 GiB depth.tar.gz is copied only with --depth; Table 5 does not use depth.
    """
    import json
    import subprocess

    def rclone(*args):
        subprocess.run(
            ["rclone", *args, "--stats", "5m", "--stats-one-line", "-v"], check=True
        )

    manifest = "--exclude=/MANIFEST.sha256", "--exclude=/manifest.json"
    skip = () if depth else ("--exclude=/depth.tar.gz",)
    # --fast-list: one recursive listing; a request per folder (22,542 x 3) starves the transfers.
    # Smallest files first, so the skeletons and videos land before the depth archive.
    rclone("copy", "gdrive:", DATA_ROOT, "--transfers", "8", "--checkers", "8", "--fast-list",
           "--order-by", "size,ascending", "--retries", "5", "--low-level-retries", "20", *skip)  # fmt: skip
    datasets.commit()
    # Compare every file with Drive's MD5; fails on a missing or differing file.
    rclone(
        "check", "gdrive:", DATA_ROOT, "--one-way", "--checkers", "8", *manifest, *skip
    )
    rclone(
        "hashsum",
        "sha256",
        DATA_ROOT,
        "--output-file",
        "/tmp/MANIFEST.sha256",
        *manifest,
    )
    lines = open("/tmp/MANIFEST.sha256", encoding="utf-8").read().splitlines()
    lines.sort(key=lambda line: line.split("  ", 1)[1])
    with open(f"{DATA_ROOT}/MANIFEST.sha256", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    size = subprocess.run(["rclone", "size", DATA_ROOT, "--json", *manifest],
                          check=True, capture_output=True, text=True).stdout  # fmt: skip
    summary = {
        "slug": SLUG,
        "source": "Google Drive folder shared by the dataset owners, mirrored unchanged",
        "rclone": RCLONE,
        "verified": "rclone check against Google Drive MD5",
        **json.loads(size),
    }
    with open(f"{DATA_ROOT}/manifest.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    datasets.commit()
    print(json.dumps(summary))


HGR = "https://sun.aei.polsl.pl/~mkawulok/gestures"
HGR_TERMS = (
    "The database is free to use for research, non-commercial purposes, provided that "
    "the following papers are cited in the works that exploit the database."
)
HGR_PARTS = (
    "images",
    "skin",
    "feature_pts",
    "images_large",
    "skin_large",
    "feature_pts_large",
)
# slug -> (official page, terms of use quoted from that page, archive URLs)
PUBLIC = {
    "hgr2a": (f"{HGR}/", HGR_TERMS, [f"{HGR}/hgr2a_{part}.zip" for part in HGR_PARTS]),
    "hgr2b": (f"{HGR}/", HGR_TERMS, [f"{HGR}/hgr2b_{part}.zip" for part in HGR_PARTS]),
    "rendered-handpose-dataset": (
        "https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html",
        "This dataset is provided for research purposes only and without any warranty. "
        "Any commercial use is prohibited. If you use the dataset or parts of it in your "
        "research, you must cite the respective paper.",
        ["https://lmb.informatik.uni-freiburg.de/data/RenderedHandpose/RHD_v1-1.zip"],
    ),
}


def sha256_file(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@app.function(
    image=populate_image,
    volumes={"/datasets": datasets, "/cache/huggingface": cache},
    cpu=4,
    memory=8192,
    timeout=3 * 3600,
)
def populate_public():
    """Each archive is kept under source/ and extracted to a folder named after it.

    Safe to re-run: a dataset with a manifest.json is skipped.
    """
    import json
    import os
    import subprocess
    import zipfile

    for slug, (page, terms, urls) in PUBLIC.items():
        root = f"/datasets/{slug}"
        if os.path.exists(f"{root}/manifest.json"):
            print(f"{slug}: already populated")
            continue
        os.makedirs(f"{root}/source", exist_ok=True)
        archives = []
        for url in urls:
            name = url.rsplit("/", 1)[1]
            path = f"{root}/source/{name}"
            subprocess.run(
                ["curl", "-fsSL", "-A", "Mozilla/5.0", "-o", path, url], check=True
            )
            with zipfile.ZipFile(path) as archive:
                assert archive.testzip() is None, name
                archive.extractall(f"{root}/{name[:-4]}")
            archives.append(
                {
                    "url": url,
                    "sha256": sha256_file(path),
                    "size_bytes": os.path.getsize(path),
                }
            )
        lines = []
        for folder, _, files in os.walk(root):
            for name in files:
                path = os.path.join(folder, name)
                lines.append(f"{sha256_file(path)}  {os.path.relpath(path, root)}")
        lines.sort(key=lambda line: line.split("  ", 1)[1])
        with open(f"{root}/MANIFEST.sha256", "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        summary = {
            "slug": slug,
            "page": page,
            "terms_of_use": terms,
            "archives": archives,
            "files": len(lines),
        }
        with open(f"{root}/manifest.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        datasets.commit()
        print(json.dumps(summary))
