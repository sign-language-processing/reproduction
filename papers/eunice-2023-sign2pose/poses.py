"""MediaPipe Holistic pose extraction and Sign2Pose key-frame selection for WLASL, plus SPOTER-format CSV export.

extract: decode each WLASL instance per the official start_kit/preprocess.py frame rule, run MediaPipe Holistic on
every frame and map it to SPOTER's 54 Apple Vision joints, compute the key-frame mask (Sign2Pose Section 3.2 as read in README),
and append one JSON line per instance. Resumable: instances already in the output are skipped.

csv: write SPOTER-format train/val/test CSVs for one WLASL subset, with or without key frames.
"""

import argparse
import csv
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

BODY = ["nose", "neck", "rightEye", "leftEye", "rightEar", "leftEar",
        "rightShoulder", "leftShoulder", "rightElbow", "leftElbow", "rightWrist", "leftWrist"]
HAND = ["wrist", "indexTip", "indexDIP", "indexPIP", "indexMCP", "middleTip", "middleDIP", "middlePIP", "middleMCP",
        "ringTip", "ringDIP", "ringPIP", "ringMCP", "littleTip", "littleDIP", "littlePIP", "littleMCP",
        "thumbTip", "thumbIP", "thumbMP", "thumbCMC"]
SIDES = ["left", "right"]  # SPOTER hand suffixes _0 / _1
COLUMNS = BODY + [f"{j}_{s}" for s in SIDES for j in HAND]  # 54 joints


def keyframe_mask(hists):
    """Sign2Pose Algorithm 1, read as: Ed(t) = ||H(t-1) - H(t)||_2 over 256-bin grayscale histograms,
    Th = mean(Ed) + std(Ed) (Section 3.2 computes mu/sigma over the measured distances), keep t where Ed(t) > Th.
    Frame 0 has no predecessor and is never a key frame. If nothing passes (constant or <3 frames), keep all."""
    h = np.asarray(hists, dtype=np.float64)
    ed = np.linalg.norm(np.diff(h, axis=0), axis=1)
    keep = np.zeros(len(h), dtype=bool)
    if len(ed):
        keep[1:] = ed > ed.mean() + ed.std()
    return keep if keep.any() else np.ones(len(h), dtype=bool)


def frame_range(url, extra):
    """Official WLASL start_kit/preprocess.py: YouTube instances with frame_end > 0 keep native frames
    [frame_start-1, frame_end-1]; every other instance is the whole file."""
    if ("youtube" in url or "youtu.be" in url) and extra["frame_end"] > 0:
        return extra["frame_start"] - 1, extra["frame_end"] - 1
    return None, None


# MediaPipe Holistic indices for the 54 Apple Vision joints SPOTER uses. Both libraries name sides from the
# signer's point of view. neck has no MediaPipe landmark: midpoint of the shoulders.
MP_BODY = {"nose": 0, "rightEye": 5, "leftEye": 2, "rightEar": 8, "leftEar": 7, "rightShoulder": 12,
           "leftShoulder": 11, "rightElbow": 14, "leftElbow": 13, "rightWrist": 16, "leftWrist": 15}
MP_HAND = {"wrist": 0, "indexTip": 8, "indexDIP": 7, "indexPIP": 6, "indexMCP": 5, "middleTip": 12,
           "middleDIP": 11, "middlePIP": 10, "middleMCP": 9, "ringTip": 16, "ringDIP": 15, "ringPIP": 14,
           "ringMCP": 13, "littleTip": 20, "littleDIP": 19, "littlePIP": 18, "littleMCP": 17, "thumbTip": 4,
           "thumbIP": 3, "thumbMP": 2, "thumbCMC": 1}


def _xy(lm):
    """MediaPipe normalised (x, y down) -> Apple Vision convention (x, y up), as in SPOTER's published CSVs."""
    return lm.x, 1.0 - lm.y


def _pose(holistic, rgb):
    r = holistic.process(rgb)
    out = np.zeros((len(COLUMNS), 2), dtype=np.float32)  # undetected joints stay (0, 0), as in SPOTER data
    if r.pose_landmarks:
        lms = r.pose_landmarks.landmark
        for name, i in MP_BODY.items():
            out[BODY.index(name)] = _xy(lms[i])
        out[BODY.index("neck")] = (out[BODY.index("leftShoulder")] + out[BODY.index("rightShoulder")]) / 2
    for side, hand in (("left", r.left_hand_landmarks), ("right", r.right_hand_landmarks)):
        if hand:
            base = len(BODY) + SIDES.index(side) * len(HAND)
            for name, i in MP_HAND.items():
                out[base + HAND.index(name)] = _xy(hand.landmark[i])
    return out


def _extract_file(job):
    import mediapipe as mp
    from simple_video_utils.frames import read_frames_exact
    path, instances = job
    results = []
    for inst in instances:
        start, end = frame_range(inst["url"], inst["extra"])
        poses, hists, shape = [], [], None
        try:  # fresh tracker per instance; library defaults (model_complexity=1, confidences 0.5)
            with mp.solutions.holistic.Holistic(static_image_mode=False) as holistic:
                for rgb in read_frames_exact(path, start_frame=start, end_frame=end):
                    shape = rgb.shape
                    gray = (rgb @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)
                    hists.append(np.bincount(gray.ravel(), minlength=256))
                    poses.append(_pose(holistic, np.ascontiguousarray(rgb)))
        except Exception as exc:  # recorded per instance, never silently dropped
            results.append({**inst["meta"], "error": f"{type(exc).__name__}: {exc}"})
            continue
        if not poses:
            results.append({**inst["meta"], "error": "no frames decoded"})
            continue
        results.append({**inst["meta"], "height": shape[0], "width": shape[1],
                        "poses": np.round(np.stack(poses), 6).tolist(),
                        "keyframes": np.flatnonzero(keyframe_mask(hists)).tolist()})
    return results


def extract(args):
    glosses = [g["gloss"] for g in json.load(open(args.wlasl_json))]
    gloss_index = {g: i for i, g in enumerate(glosses)}
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.loads(line)["key"] for line in f}
    jobs = {}
    with open(args.index) as f:
        for row in csv.DictReader(f):
            extra = json.loads(row["extra"])
            key = f"{row['text']}|{extra['instance_id']}"
            if key in done:
                continue
            meta = {"key": key, "gloss": row["text"], "label": gloss_index[row["text"]], "split": row["split"],
                    "instance_id": extra["instance_id"], "file": row["file"], "fps": extra["fps"]}
            jobs.setdefault(os.path.join(args.videos_root, row["file"]), []).append(
                {"url": extra["url"], "extra": extra, "meta": meta})
    print(f"{len(done)} done, {sum(map(len, jobs.values()))} instances in {len(jobs)} files to go", flush=True)
    with Pool(args.workers) as pool, open(args.out, "a") as out:
        for n, results in enumerate(pool.imap_unordered(_extract_file, jobs.items()), 1):
            for r in results:
                out.write(json.dumps(r) + "\n")
            if n % 200 == 0:
                out.flush()
                print(f"{n}/{len(jobs)} files", flush=True)


def export_csv(args):
    rows = {"train": [], "validation": [], "test": []}  # split names as in WLASL index.csv
    with open(args.poses) as f:
        for line in f:
            r = json.loads(line)
            if "error" in r or r["label"] >= args.subset:
                continue
            p = np.asarray(r["poses"])
            if args.keyframes:
                p = p[r["keyframes"]]
            row = {"labels": r["label"] + 1,  # SPOTER's loader subtracts 1 (datasets/czech_slr_dataset.py)
                   "video_size_height": r["height"], "video_size_width": r["width"], "video_fps": r["fps"]}
            for j, name in enumerate(COLUMNS):
                row[f"{name}_X"] = json.dumps(np.round(p[:, j, 0], 6).tolist())
                row[f"{name}_Y"] = json.dumps(np.round(p[:, j, 1], 6).tolist())
            rows[r["split"]].append(row)
    os.makedirs(args.out_dir, exist_ok=True)
    tag = "keyframes" if args.keyframes else "allframes"
    for split, split_rows in rows.items():
        path = os.path.join(args.out_dir, f"WLASL{args.subset}_{split}_{tag}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(split_rows[0]))
            w.writeheader()
            w.writerows(split_rows)
        print(path, len(split_rows))


def self_check():
    hists = [np.full(256, 10)] * 5 + [np.r_[np.full(128, 20), np.zeros(128)]] + [np.full(256, 10)] * 5
    assert keyframe_mask(hists).nonzero()[0].tolist() == [5, 6], keyframe_mask(hists)
    assert keyframe_mask([np.ones(256)] * 4).all()
    assert frame_range("https://www.youtube.com/watch?v=x", {"frame_start": 11, "frame_end": 40}) == (10, 39)
    assert frame_range("https://www.youtube.com/watch?v=x", {"frame_start": 1, "frame_end": -1}) == (None, None)
    assert frame_range("http://aslbricks.org/a.mp4", {"frame_start": 5, "frame_end": 9}) == (None, None)
    assert len(COLUMNS) == 54
    assert set(MP_BODY) == set(BODY) - {"neck"} and set(MP_HAND) == set(HAND)
    assert sorted(MP_HAND.values()) == list(range(21)) and len(set(MP_BODY.values())) == 11
    print("self-check ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--index", required=True)
    e.add_argument("--wlasl-json", required=True)
    e.add_argument("--videos-root", required=True, help="directory containing the index's videos/ paths")
    e.add_argument("--out", required=True)
    e.add_argument("--workers", type=int, default=4)
    c = sub.add_parser("csv")
    c.add_argument("--poses", required=True)
    c.add_argument("--subset", type=int, choices=[100, 300, 1000, 2000], required=True)
    c.add_argument("--keyframes", action="store_true")
    c.add_argument("--out-dir", required=True)
    sub.add_parser("check")
    a = parser.parse_args()
    {"extract": extract, "csv": export_csv, "check": lambda _: self_check()}[a.cmd](a)
    sys.exit(0)
