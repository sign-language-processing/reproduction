"""Write SignXmDA input pickles (neccam/slt format) from SMKD features.

Each entry is {name, signer, gloss, text, sign} with sign a float32 torch
tensor (frames x 1024); name/signer/gloss/text come from the index that
extract_smkd_features.py wrote next to the features. With --teacher-outputs,
every training video is repeated once per teacher with that teacher's
translation as text. The repeats get distinct names
("<name>#teacher<k>") because SignTranslationDataset keeps at most one
duplicate per name (dataset.py L191-200); they share the original tensor, so
pickle stores the features once.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch


def read_teacher_output(path: Path) -> dict:
    """Parse a SignXmDA '<sequence>|<hypothesis>' translation file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return dict(line.split("|", 1) for line in lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="prefix of .npy/.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--teacher-outputs", type=Path, nargs="*", default=[])
    args = parser.parse_args()

    index = json.loads(Path(f"{args.features}.json").read_text(encoding="utf-8"))["index"]
    frames = np.load(Path(f"{args.features}.npy"), mmap_mode="r")
    teachers = [read_teacher_output(path) for path in args.teacher_outputs]

    entries = []
    for item in index:
        sign = torch.from_numpy(np.array(frames[item["start"] : item["end"]]))
        entry = {key: item[key] for key in ("name", "signer", "gloss", "text")}
        entry["sign"] = sign
        entries.append(entry)
        for number, teacher in enumerate(teachers, 1):
            entries.append({**entry, "name": f"{item['name']}#teacher{number}",
                            "text": teacher[item["name"]]})

    if len(frames) != index[-1]["end"]:
        raise ValueError(f"{len(frames)} feature rows but index ends at {index[-1]['end']}")
    with args.out.open("wb") as handle:
        pickle.dump(entries, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps({"out": str(args.out), "entries": len(entries), "teachers": len(teachers)}))


if __name__ == "__main__":
    main()
