"""Check the authors' released logits against their published prediction CSVs.

Evidence audit only. The official validation/test labels are concealed, so no
accuracy is computed. For every published prediction file, the script reports
how often it agrees with the argmax of the corresponding fusion rule applied to
the released logits.

Usage: python3 -P -E audit_author_logits.py LOGIT_DIR UPSTREAM_CHECKOUT
  LOGIT_DIR          extracted Logit_folder/ from the authors' Google Drive zip
  UPSTREAM_CHECKOUT  impressncsu/ICCV2025_MSLR at 7dd73c4e67b2959a95f828c103269723528b9271
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

logit_dir, repo = Path(sys.argv[1]), Path(sys.argv[2])


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def load(split):
    frames = {
        "slgcn": pd.read_csv(logit_dir / f"logit_slgcnn_{split}.csv"),
        "md": pd.read_excel(logit_dir / f"logit_md_{split}.xlsx"),
        "video": pd.read_excel(logit_dir / f"logit_video_{split}.xlsx"),
    }
    logits = {}
    ids = None
    for name, df in frames.items():
        df = df.sort_values("id").reset_index(drop=True)
        values = df.iloc[:, 1:].to_numpy(dtype=np.float64)
        print(f"  {name}: rows={len(df)} classes={values.shape[1]} "
              f"unique_ids={df['id'].nunique()} finite={bool(np.isfinite(values).all())}")
        if ids is None:
            ids = df["id"].to_numpy()
        assert np.array_equal(ids, df["id"].to_numpy()), f"{name} ids differ"
        logits[name] = values
    return ids, logits


def compare(ids, pred, csv_path, rule):
    ref = pd.read_csv(csv_path).sort_values("id").reset_index(drop=True)
    assert np.array_equal(ref["id"].to_numpy(), ids), f"ids differ in {csv_path}"
    differ = int((ref["pred"].to_numpy() != pred).sum())
    print(f"  {rule:<34} vs {csv_path.relative_to(repo)}: {differ} of {len(ids)} differ")


for split in ("val", "test"):
    print(f"[{split}]")
    ids, z = load(split)
    rv = repo / "MSLR_radar_video"
    fs = rv / "fusion_with_slgcnn" / "predictions"
    rules = [
        ("argmax md logits", z["md"], rv / f"results/predictions/md_preds_{split}.csv"),
        ("argmax video logits", z["video"], rv / f"results/predictions/video_preds_{split}.csv"),
        ("0.5 md + 0.5 video (probabilities)", 0.5 * softmax(z["md"]) + 0.5 * softmax(z["video"]),
         rv / f"results/predictions/fusion_preds_{split}.csv"),
        ("0.5 md + 0.5 video (logits)", 0.5 * z["md"] + 0.5 * z["video"],
         rv / f"results/predictions/fusion_preds_{split}.csv"),
        ("0.25 md + 0.75 slgcn", 0.25 * z["md"] + 0.75 * z["slgcn"], fs / f"fusion_md_slgcnn_{split}.csv"),
        ("0.25 video + 0.75 slgcn", 0.25 * z["video"] + 0.75 * z["slgcn"], fs / f"fusion_video_slgcnn_{split}.csv"),
        ("0.5 slgcn + 0.25 md + 0.25 video", 0.5 * z["slgcn"] + 0.25 * z["md"] + 0.25 * z["video"],
         fs / f"fusion_slgcnn_md_video_{split}.csv"),
        ("0.5 slgcn + 0.25 md + 0.25 video", 0.5 * z["slgcn"] + 0.25 * z["md"] + 0.25 * z["video"],
         repo / f"Final_prediction/Final_LIS_prediction_{split}.csv"),
    ]
    for rule, scores, path in rules:
        compare(ids, np.argmax(scores, axis=1), path, rule)
