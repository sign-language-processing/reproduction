"""Modal jobs for Kindiroglu et al. 2019, Table 5 (Temporal Accumulative Features).

populate:        extract the uploaded BosphorusSign22k Drive export in the shared datasets
                 Volume, verify it against Drive's MD5s and write a SHA-256 manifest.
populate_public: download the public hand-segmentation datasets (HGR2A, HGR2B, Rendered
                 Handpose Dataset) from their official pages into the same Volume.
"""

import modal

PAPER = "13fbc667"
SLUG = "bosphorussign22k"
DATA_ROOT = f"/datasets/{SLUG}"

app = modal.App(f"taf-{PAPER}")
datasets = modal.Volume.from_name("datasets", version=2)
cache = modal.Volume.from_name("huggingface-cache", version=2)

CACHE_ENV = {"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"}
populate_image = (
    modal.Image.debian_slim(python_version="3.11").apt_install("curl").env(CACHE_ENV)
)


@app.function(
    image=populate_image,
    volumes={"/datasets": datasets, "/cache/huggingface": cache},
    cpu=8,
    memory=16384,
    timeout=6 * 3600,
)
def populate():
    """Extract the Drive-export zips under source/ and verify every file.

    source/drive-md5.txt is Google Drive's own MD5 list of the folder (rclone md5sum);
    every listed file must exist with that MD5, and no other file may appear.
    Safe to re-run.
    """
    import hashlib
    import json
    import os
    import zipfile

    source = f"{DATA_ROOT}/source"
    for name in sorted(os.listdir(source)):
        path = f"{source}/{name}"
        if name.endswith(".zip"):
            with zipfile.ZipFile(path) as archive:
                archive.extractall(DATA_ROOT)
        elif name != "drive-md5.txt":
            with open(path, "rb") as src, open(f"{DATA_ROOT}/{name}", "wb") as dst:
                dst.write(src.read())
        print("unpacked", name, flush=True)
    datasets.commit()

    expected = {}
    for line in open(f"{source}/drive-md5.txt", encoding="utf-8"):
        md5, path = line.rstrip("\n").split("  ", 1)
        expected[path] = md5
    lines, wrong, extra, size = [], [], [], 0
    for folder, _, files in os.walk(DATA_ROOT):
        if folder.startswith(source):
            continue
        for name in files:
            path = os.path.join(folder, name)
            rel = os.path.relpath(path, DATA_ROOT)
            if rel in ("MANIFEST.sha256", "manifest.json"):
                continue
            md5, sha = hashlib.md5(), hashlib.sha256()
            with open(path, "rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    md5.update(block)
                    sha.update(block)
                    size += len(block)
            lines.append(f"{sha.hexdigest()}  {rel}")
            if rel not in expected:
                extra.append(rel)
            elif expected.pop(rel) != md5.hexdigest():
                wrong.append(rel)
    summary = {
        "slug": SLUG,
        "source": "Google Drive folder shared by the dataset owners, exported as zips; depth.tar.gz not included",
        "verified": "every file against Google Drive's MD5 list (source/drive-md5.txt)",
        "files": len(lines),
        "bytes": size,
        "missing": sorted(expected),
        "md5_mismatch": sorted(wrong),
        "not_on_drive": sorted(extra),
    }
    print(
        json.dumps(
            {
                k: (v if not isinstance(v, list) else [len(v), v[:20]])
                for k, v in summary.items()
            }
        )
    )
    assert not (expected or wrong or extra), (
        "extracted files differ from the Drive listing"
    )
    lines.sort(key=lambda line: line.split("  ", 1)[1])
    with open(f"{DATA_ROOT}/MANIFEST.sha256", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    with open(f"{DATA_ROOT}/manifest.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    datasets.commit()


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
