"""Modal jobs for Kindiroglu et al. 2019, Table 5 (Temporal Accumulative Features).

populate: mirror the BosphorusSign22k Google Drive folder into the shared datasets
          Volume, verify it against Drive's MD5s and write a SHA-256 manifest.
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
