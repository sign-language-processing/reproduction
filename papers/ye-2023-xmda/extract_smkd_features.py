"""Embed sign videos with a released SLTUNet SMKD checkpoint.

Runs SLTUNet's own SignEmbedding (smkd/sign_embedder.py, the path its infer.sh
uses) and stores the 1024-d TemporalConv output ('visual_features') per video.
Output is numpy/JSON only so the torch 1.4 SignXmDA image can read it:
<out>.npy (concatenated frames) and <out>.json (index with name, signer,
gloss, text and the row range of each video).

PHOENIX-2014T is read from the original PNG frames, as in SLTUNet. The CSL-Daily
copy on the Volume is mp4 only (SLTUNet read the official 512x512 JPG frames),
so InferFeeder.read_video is replaced for .mp4 inputs by a simple-video-utils
decode followed by the upstream 256x256 Lanczos resize; transforms and collation
stay upstream.

ctcdecode is only needed for CTC beam decoding of glosses, never for the
features, so it is stubbed and the model's decoder is replaced by a no-op; a
greedy CTC collapse of 'sequence_logits' gives an approximate WER control.
"""

import argparse
import gzip
import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np


def levenshtein(ref, hyp):
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        previous, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (r != h))
    return row[-1]


def phoenix_annotations(root: Path, split: str) -> list:
    """neccam/slt pami0 annotations; frames under fullFrame-210x260px/<split>/<id>/."""
    path = root / f"features/annotations/phoenix14t.pami0.{split}.annotations_only.gzip"
    with gzip.open(path, "rb") as handle:
        annotations = pickle.load(handle)
    frames = root / "raw/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px"
    # SLTUNet's loader silently embeds one black frame when a glob is empty.
    empty = [a["name"] for a in annotations if not any((frames / a["name"]).glob("*.png"))]
    if empty:
        raise FileNotFoundError(f"{len(empty)} videos without PNG frames, e.g. {empty[:3]}")
    return [{"name": a["name"], "signer": a["signer"], "gloss": a["gloss"], "text": a["text"],
             "source": f"{frames / a['name']}/*.png"} for a in annotations]


def csl_daily_annotations(root: Path, split: str) -> list:
    """csl2020ct_v2.pkl with split_1.txt; SLT target is label_char (dataset README)."""
    labels = root / "sentence_label"
    names = [line.split("|")[0] for line in (labels / "split_1.txt").read_text().split()[1:]
             if line.endswith(f"|{split}")]
    with (labels / "csl2020ct_v2.pkl").open("rb") as handle:
        info = {item["name"]: item for item in pickle.load(handle)["info"]}
    missing = [name for name in names if not (root / "videos" / f"{name}.mp4").is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} videos missing, e.g. {missing[:3]}")
    # split_1.txt lists train clip S000005_P0004_T00, which has no entry in
    # csl2020ct_v2.pkl (or video_map.txt); without a label it cannot be used.
    unlabelled = [name for name in names if name not in info]
    if unlabelled:
        print(json.dumps({"skipped_without_annotation": unlabelled}))
    names = [name for name in names if name in info]
    return [{"name": f"{split}/{name}", "signer": str(info[name]["signer"]),
             "gloss": " ".join(info[name]["label_gloss"]),
             "text": " ".join(info[name]["label_char"]),
             "source": str(root / "videos" / f"{name}.mp4")} for name in names]


def decode_mp4_with_simple_video_utils() -> None:
    import cv2
    import dataset.dataloader_video as loader
    from simple_video_utils.frames import read_frames_exact

    upstream = loader.InferFeeder.read_video

    def read_video(self, index, num_glosses=-1):
        info = self.inputs_list[index]
        if not info["folder"].endswith(".mp4"):
            return upstream(self, index, num_glosses)
        frames = [cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                  for frame in read_frames_exact(info["folder"], start_frame=0, end_frame=None)]
        if not frames:
            raise ValueError(f"no frames decoded from {info['folder']}")
        return frames, [0], info

    loader.InferFeeder.read_video = read_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["phoenix14t", "csl-daily"], required=True)
    parser.add_argument("--root", type=Path, required=True, help="dataset directory on /datasets")
    parser.add_argument("--split", choices=["train", "dev", "test"], required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    sys.modules["ctcdecode"] = types.SimpleNamespace(CTCBeamDecoder=lambda *a, **k: None)
    sys.path.insert(0, "/sltunet/smkd")
    from sign_embedder import SignEmbedding  # noqa: E402

    if args.dataset == "phoenix14t":
        annotations = phoenix_annotations(args.root, args.split)
        gloss_dict = args.release / "phoenix2014/gloss_dict.npy"
    else:
        annotations = csl_daily_annotations(args.root, args.split)
        gloss_dict = args.release / "csldaily/gloss_dict.npy"
        decode_mp4_with_simple_video_utils()
    if args.limit:
        annotations = annotations[: args.limit]
    listing = Path(f"{args.out}.sources.txt")
    listing.parent.mkdir(parents=True, exist_ok=True)
    listing.write_text("".join(a["source"] + "\n" for a in annotations), encoding="utf-8")

    embedder = SignEmbedding(
        cfg=str(args.release / "baseline.yaml"),
        gloss_path=str(gloss_dict),
        sign_video_path=str(listing),
        model_path=str(args.release / "signemb_ckpt/average.pt"),
        gpu_id=0,
        batch_size=args.batch_size,
    )
    embedder.model.decoder = types.SimpleNamespace(decode=lambda *a, **k: None)
    i2g = {index: gloss for gloss, (index, *_rest) in embedder.gloss_dict.items()}

    # SignEmbedding.embed() drops the greedy gloss output; keep it for a WER control.
    import torch

    embedder.model.eval()
    features, hypotheses = {}, {}
    for data in embedder.data_loader["infer"]:
        video = embedder.device.data_to_device(data[0])
        lengths = embedder.device.data_to_device(data[1])
        with torch.no_grad():
            output = embedder.model(video, lengths)
        feat_len = output["feat_len"].cpu().numpy().astype(np.int32)
        visual = output["visual_features"].permute(1, 0, 2)
        for index, info in enumerate(data[-1]):
            features[info[1]] = visual[index][: feat_len[index]].cpu().numpy().astype(np.float32)
            best = output["sequence_logits"][: int(feat_len[index]), index].argmax(-1).tolist()
            collapsed = [k for position, k in enumerate(best) if k != 0 and (position == 0 or best[position - 1] != k)]
            hypotheses[info[1]] = [i2g[k] for k in collapsed]

    index, arrays, start, errors, words = [], [], 0, 0, 0
    for position, annotation in enumerate(annotations):
        array = features[position]
        reference = annotation["gloss"].split()
        errors += levenshtein(reference, hypotheses[position])
        words += len(reference)
        entry = {k: annotation[k] for k in ("name", "signer", "gloss", "text")}
        index.append({**entry, "start": start, "end": start + len(array)})
        arrays.append(array)
        start += len(array)
    np.save(Path(f"{args.out}.npy"), np.concatenate(arrays))
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "videos": len(index),
        "feature_frames": start,
        "feature_dim": int(arrays[0].shape[1]),
        "gloss_dict_size": len(embedder.gloss_dict),
        "greedy_gloss_wer_percent": 100.0 * errors / words,
        "index": index,
    }
    Path(f"{args.out}.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "index"}))


if __name__ == "__main__":
    main()
