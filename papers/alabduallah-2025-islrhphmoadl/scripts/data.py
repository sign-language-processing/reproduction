"""Approved conditional ASL subset; data population is separate from training."""
import hashlib
import json
import string
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

CLASSES = list(string.ascii_uppercase) + ["space", "del", "nothing"]
ROOT = Path("/datasets/asl-alphabet")
MANIFEST = ROOT / "reproductions/734bfc1c/split-seed42.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare():
    """Hash real files, create one immutable conditional split on shared datasets."""
    source = ROOT / "source/asl_alphabet_train/asl_alphabet_train"
    rng = np.random.default_rng(42)
    rows = []
    counts = {}
    for label, name in enumerate(CLASSES):
        paths = sorted((source / name).glob("*.jpg"))
        if len(paths) != 3000:
            raise ValueError(f"Expected 3000 images for {name}, found {len(paths)}")
        counts[name] = len(paths)
        for position, index in enumerate(rng.permutation(len(paths))[:2000]):
            split = "train" if position < 1260 else "validation" if position < 1400 else "test"
            rows.append({"path": str(paths[index].relative_to(ROOT)), "label": label, "split": split})
    def inspect(row):
        from PIL import Image
        path = ROOT / row["path"]
        with Image.open(path) as im:
            im.verify()
        return {**row, "sha256": sha256(path)}
    with ThreadPoolExecutor(max_workers=16) as pool:
        rows = list(pool.map(inspect, rows))
    # Duplicate content across splits would invalidate a held-out interpretation.
    by_hash = {}
    cross_split_duplicates = 0
    for row in rows:
        old = by_hash.setdefault(row["sha256"], row["split"])
        cross_split_duplicates += old != row["split"]
    document = {"conditional": True, "seed": 42, "rng": "numpy.default_rng/PCG64",
                "source_manifest_sha256": sha256(ROOT / "manifest.json"),
                "classes": CLASSES, "source_counts": counts, "records": rows,
                "cross_split_duplicate_content_count": cross_split_duplicates}
    if MANIFEST.exists():
        if json.loads(MANIFEST.read_text()) != document:
            raise ValueError("Existing split differs; refusing to overwrite provenance")
    else:
        write_json(MANIFEST, document)
    return {"manifest": str(MANIFEST), "sha256": sha256(MANIFEST),
            "source_counts": counts, "split_counts": {s: sum(r["split"] == s for r in rows)
                                                       for s in ("train", "validation", "test")},
            "cross_split_duplicate_content_count": cross_split_duplicates}


def load_manifest():
    d = json.loads(MANIFEST.read_text())
    assert d["classes"] == CLASSES and len(d["records"]) == 58000
    assert len({r["path"] for r in d["records"]}) == 58000
    if d["cross_split_duplicate_content_count"]:
        raise ValueError("Exact image content crosses splits; protocol review required")
    for split, count in (("train", 1260), ("validation", 140), ("test", 600)):
        for label in range(29):
            assert sum(r["split"] == split and r["label"] == label for r in d["records"]) == count
    return d
