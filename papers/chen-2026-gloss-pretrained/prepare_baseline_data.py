"""Write plain aligned src(gloss)/tgt(text) files for onmt_preprocess.

Standalone (no HF `transformers`/`datasets` dependency) since this runs in
the separate older-stack OpenNMT-py/torchtext environment. Reuses the same
loading/preprocessing logic as train_finetune.py's LOADERS (same gloss
lowercasing for Phoenix, same character-level Chinese handling for
CSL-Daily) so the Transformer-tiny baseline sees identical data to mBART/mT5.
"""
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path


def load_phoenix(root: Path, split: str) -> tuple[list[str], list[str]]:
    name = {"train": "train", "dev": "dev", "test": "test"}[split]
    path = root / f"PHOENIX-2014-T.{name}.corpus.csv"
    gloss, text = [], []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="|")
        for row in reader:
            gloss.append(row["orth"].lower())
            text.append(row["translation"])
    return gloss, text


def load_aslgpc12(root: Path, split: str) -> tuple[list[str], list[str]]:
    gloss = (root / f"{split}.gloss").read_text(encoding="utf-8").splitlines()
    text = (root / f"{split}.en").read_text(encoding="utf-8").splitlines()
    return gloss, text


def load_csldaily(root: Path, split: str) -> tuple[list[str], list[str]]:
    with (root / "csl2020ct_v2.pkl").open("rb") as fh:
        data = pickle.load(fh)
    split_by_name = {}
    with (root / "split_1.txt").open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            name, split_name = line.strip().split("|")
            split_by_name[name] = split_name
    gloss, text = [], []
    for entry in data["info"]:
        if split_by_name.get(entry["name"]) != split:
            continue
        gloss.append(" ".join(entry["label_gloss"]))
        text.append(" ".join(entry["label_char"]))  # space-separate for OpenNMT tokenization
    return gloss, text


LOADERS = {"phoenix": load_phoenix, "aslgpc12": load_aslgpc12, "csldaily": load_csldaily}
DATASET_PATHS = {
    "phoenix": "/datasets/rwth-phoenix-2014-t/annotations",
    "aslgpc12": "/datasets/aslg-pc12",
    "csldaily": "/datasets/csl-daily/sentence_label",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=LOADERS, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        gloss, text = LOADERS[args.dataset](Path(DATASET_PATHS[args.dataset]), split)
        assert len(gloss) == len(text) and len(gloss) > 0
        (output_dir / f"{split}.src").write_text("\n".join(gloss) + "\n", encoding="utf-8")
        (output_dir / f"{split}.tgt").write_text("\n".join(text) + "\n", encoding="utf-8")
        print(f"{args.dataset}/{split}: {len(gloss)} pairs")


if __name__ == "__main__":
    main()
