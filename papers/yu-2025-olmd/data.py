"""Dataset access for the OLMD reproduction.

Three corpora, three on-disk formats, all mounted read-only from the shared
Modal Volume `datasets`:

  * PHOENIX14   -- extracted per-utterance PNG frame directories
  * PHOENIX14-T -- per-split mp4 (plus PAMI feature archives OLMD does not use,
                   since it trains a ResNet34 on images)
  * CSL-Daily   -- per-utterance mp4

Video files are decoded with simple-video-utils per AGENTS.md. PHOENIX14 needs
no decoder because its frames are already extracted.

Conventions the OLMD paper leaves unstated are taken from its immediate
methodological ancestor (CorrNet, github.com/hulianyuyy/CorrNet, pinned at
a812814f) and each is recorded as a deviation in reproduction.json.
"""

from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Paths inside the mounted `datasets` Volume, as verified at the data gate.
PHOENIX14_ROOT = "rwth-phoenix-2014/raw/phoenix2014-release/phoenix-2014-multisigner"
PHOENIX14T_ROOT = "rwth-phoenix-2014-t"
CSL_DAILY_ROOT = "csl-daily"

# Split sizes confirmed at the data gate against the canonical published counts.
EXPECTED_COUNTS = {
    "phoenix14": {"train": 5672, "dev": 540, "test": 629},
    "phoenix14t": {"train": 7096, "dev": 519, "test": 642},
    # CSL-Daily's split_1.txt lists 18401 unique training names, but one of
    # them -- S000005_P0004_T00 -- has no entry in csl2020ct_v2.pkl at all, so
    # only 18400 training utterances are usable. This is a defect in the
    # published corpus, not in this loader; see diagnose_csl_daily below.
    "csl-daily": {"train": 18400, "dev": 1077, "test": 1176},
}
CSL_DAILY_MISSING_FROM_PICKLE = ("S000005_P0004_T00",)


@dataclass
class Sample:
    """One utterance: where its frames live, and its gloss sequence."""

    name: str
    source: str  # frame directory or video file, relative to the dataset root
    glosses: list[str]


def load_phoenix14(root: Path, split: str) -> list[Sample]:
    """PHOENIX14 multisigner. Frames are PNG directories, so no decoding."""
    annotation = root / PHOENIX14_ROOT / "annotations/manual" / f"{split}.corpus.csv"
    samples = []
    with annotation.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="|"):
            glosses = row["annotation"].strip().split()
            if not glosses:
                continue
            # The CSV's `folder` column is a frame glob, e.g.
            # "01April_2010_Thursday_heute_default-1/1/*.png"; the frames live
            # in the directory that glob sits in (verified on the Volume).
            folder = row["folder"].rsplit("/", 1)[0]
            samples.append(
                Sample(
                    name=row["id"],
                    source=f"{PHOENIX14_ROOT}/features/fullFrame-210x260px/{split}/{folder}",
                    glosses=glosses,
                )
            )
    return samples


def load_phoenix14t(root: Path, split: str) -> list[Sample]:
    """PHOENIX14-T. The `orth` column holds glosses; `translation` is for SLT."""
    annotation = root / PHOENIX14T_ROOT / "annotations" / f"PHOENIX-2014-T.{split}.corpus.csv"
    samples = []
    with annotation.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="|"):
            glosses = row["orth"].strip().split()
            if not glosses:
                continue
            samples.append(
                Sample(
                    name=row["name"],
                    source=f"{PHOENIX14T_ROOT}/videos/{split}/{row['name']}.mp4",
                    glosses=glosses,
                )
            )
    return samples


def load_csl_daily(root: Path, split: str) -> list[Sample]:
    """CSL-Daily. Splits come from split_1.txt; glosses from csl2020ct_v2.pkl.

    Per the corpus's own README, the pickle's `info` entries carry `name`,
    `length`, `label_gloss` (recognition target), `label_char`, `label_word`,
    `signer` and `time`. `label_gloss` is the CSLR target.
    """
    base = root / CSL_DAILY_ROOT / "sentence_label"
    wanted = set()
    with (base / "split_1.txt").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="|"):
            if row["split"].strip() == split:
                wanted.add(row["name"].strip())

    with (base / "csl2020ct_v2.pkl").open("rb") as handle:
        annotations = pickle.load(handle)

    samples = []
    for entry in annotations["info"]:
        name = entry["name"]
        if name not in wanted:
            continue
        glosses = list(entry["label_gloss"])
        if glosses:
            samples.append(
                Sample(
                    name=name,
                    source=f"{CSL_DAILY_ROOT}/videos/{name}.mp4",
                    glosses=glosses,
                )
            )
    return samples


LOADERS = {
    "phoenix14": load_phoenix14,
    "phoenix14t": load_phoenix14t,
    "csl-daily": load_csl_daily,
}


def build_vocabulary(root: Path, dataset: str) -> dict[str, int]:
    """Gloss vocabulary with index 0 reserved for the CTC blank.

    Built over train, dev AND test, which is what the corpora's own gloss
    dictionaries do and what reproduces the paper's stated vocabulary sizes:
    PHOENIX14 gives 1295 glosses (the paper says "1,295 words"; 1296 classes
    with the blank, the familiar figure in this literature), whereas a
    train-only vocabulary gives 1231 and would leave dev/test glosses
    unmappable. CSL-Daily ships gloss_map.txt with 2000 entries, matching.
    """
    loader = LOADERS[dataset]
    glosses = sorted(
        {
            gloss
            for split in ("train", "dev", "test")
            for sample in loader(root, split)
            for gloss in sample.glosses
        }
    )
    return {gloss: index + 1 for index, gloss in enumerate(glosses)}


def inspect_all(mount: str = "/datasets") -> dict:
    """Verify every split parses and matches the counts recorded at the data gate.

    Cheap, CPU-only, and it fails loudly on a count mismatch rather than
    silently training on the wrong data.
    """
    root = Path(mount)
    report: dict = {}
    for dataset, loader in LOADERS.items():
        entry: dict = {"splits": {}, "mismatches": []}
        for split in ("train", "dev", "test"):
            try:
                samples = loader(root, split)
            except Exception as error:  # surfaced, never swallowed
                entry["splits"][split] = {"error": f"{type(error).__name__}: {error}"}
                entry["mismatches"].append(split)
                continue

            expected = EXPECTED_COUNTS[dataset][split]
            entry["splits"][split] = {
                "parsed": len(samples),
                "expected": expected,
                "example_name": samples[0].name if samples else None,
                "example_glosses": samples[0].glosses[:6] if samples else None,
                "example_source": samples[0].source if samples else None,
            }
            if len(samples) != expected:
                entry["mismatches"].append(split)

        if not any("error" in split for split in entry["splits"].values()):
            vocabulary = build_vocabulary(root, dataset)
            entry["vocabulary"] = len(vocabulary)
            entry["num_classes_with_blank"] = len(vocabulary) + 1
        report[dataset] = entry
    return report


# --- Preprocessing -------------------------------------------------------
# The paper states: resize to 256x256, random crop 224x224, 50% horizontal
# flip, "a 20% probability of temporal scale adjustment", and a central
# 224x224 crop at inference.
#
# INFERRED (guess_id: temporal-scale-range). The paper's "20%" is ambiguous:
# the lineage's TemporalRescale takes temp_scaling=0.2 as a *range*, sampling a
# length factor uniformly from [0.8, 1.2] on every training clip, with no
# probability gate. The paper's wording instead reads as a 20% chance of
# applying the augmentation, leaving its magnitude unstated. The lineage
# reading is used because it is the only one with a defined magnitude, and the
# difference is recorded as a deviation.
TEMPORAL_SCALE_RANGE = (0.8, 1.2)
TEMPORAL_MIN_LEN = 32
TEMPORAL_MAX_LEN = 230


def temporal_rescale(num_frames: int, rng: np.random.Generator) -> np.ndarray:
    """Resample frame indices to a rescaled length, clamped like the lineage."""
    low, high = TEMPORAL_SCALE_RANGE
    new_len = int(num_frames * (low + (high - low) * rng.random()))
    new_len = max(TEMPORAL_MIN_LEN, min(TEMPORAL_MAX_LEN, new_len))
    # Keep the length compatible with the {K5,P2,K5,P2} head, as the lineage does.
    if (new_len - 4) % 4 != 0:
        new_len += 4 - (new_len - 4) % 4
    if new_len <= num_frames:
        return np.sort(rng.choice(num_frames, size=new_len, replace=False))
    return np.sort(rng.choice(num_frames, size=new_len, replace=True))


# Padding for the {K5,P2,K5,P2} head, which uses padding=0 and therefore
# shortens each sequence. The lineage pads by REPLICATING the first and last
# frames (never zeros), so boundary frames stay in-distribution; the same
# choice makes LMA's frame differences zero rather than spuriously large at
# clip edges (guess_id: batching-and-padding).
LEFT_PAD = 6
TOTAL_STRIDE = 4


def padded_length(num_frames: int) -> int:
    """Length after the lineage's left/right replicate padding."""
    return int(np.ceil(num_frames / TOTAL_STRIDE)) * TOTAL_STRIDE + 2 * LEFT_PAD


def diagnose_csl_daily(root: Path) -> dict:
    """Explain any split_1 name that produces no training sample.

    Exists because the loader initially yielded 18400 of the expected 18401
    training utterances, and a silently dropped sample is not acceptable
    evidence.
    """
    base = root / CSL_DAILY_ROOT / "sentence_label"
    wanted = []
    with (base / "split_1.txt").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="|"):
            if row["split"].strip() == "train":
                wanted.append(row["name"].strip())

    with (base / "csl2020ct_v2.pkl").open("rb") as handle:
        annotations = pickle.load(handle)
    by_name = {entry["name"]: entry for entry in annotations["info"]}

    missing = [name for name in wanted if name not in by_name]
    empty = [
        name for name in wanted if name in by_name and not list(by_name[name]["label_gloss"])
    ]
    return {
        "split_1_train_names": len(wanted),
        "unique_names": len(set(wanted)),
        "pickle_info_entries": len(by_name),
        "absent_from_pickle": missing,
        "present_but_empty_glosses": empty,
    }


# --- Frame loading -------------------------------------------------------
# PHOENIX14 ships extracted PNG directories; PHOENIX14-T and CSL-Daily ship
# mp4. Video files are decoded with simple-video-utils per AGENTS.md; PHOENIX14
# needs no decoder.
RESIZE = 256
CROP = 224


def load_frames(root: Path, sample: Sample) -> np.ndarray:
    """Return an utterance's frames as uint8 (T, H, W, 3) in RGB."""
    source = root / sample.source
    if source.is_dir():
        return _load_png_directory(source)
    return _load_video(source)


def _load_png_directory(directory: Path) -> np.ndarray:
    from PIL import Image

    paths = sorted(directory.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"no PNG frames in {directory}")
    return np.stack([np.asarray(Image.open(path).convert("RGB")) for path in paths])


def _load_video(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing video {path}")
    # simple-video-utils is the decoder AGENTS.md requires for video paths.
    from simple_video_utils import read_video

    return np.asarray(read_video(str(path)))


def preprocess(
    frames: np.ndarray, train: bool, rng: "np.random.Generator | None" = None
) -> np.ndarray:
    """Resize to 256x256, then crop (random + flip when training, else central).

    The paper: "All input frames are first resized to 256x256 and then randomly
    cropped to 224x224 during training, with a 50% chance of horizontal
    flipping ... For inference, we simply use a central crop of 224x224."
    Returns float32 (T, 3, 224, 224).
    """
    from PIL import Image

    resized = np.stack(
        [
            np.asarray(Image.fromarray(frame).resize((RESIZE, RESIZE), Image.BILINEAR))
            for frame in frames
        ]
    )

    if train:
        if rng is None:
            rng = np.random.default_rng()
        top = int(rng.integers(0, RESIZE - CROP + 1))
        left = int(rng.integers(0, RESIZE - CROP + 1))
        flip = rng.random() < 0.5
    else:
        top = left = (RESIZE - CROP) // 2
        flip = False

    cropped = resized[:, top : top + CROP, left : left + CROP, :]
    if flip:
        cropped = cropped[:, :, ::-1, :]

    # INFERRED (guess_id: unstated-training-hyperparameters): the paper states
    # no normalization. The lineage scales to [-1, 1] as x/127.5 - 1 rather
    # than using ImageNet channel statistics, despite the ImageNet-pretrained
    # backbone. That convention is followed and recorded as a deviation.
    scaled = cropped.astype(np.float32) / 127.5 - 1.0
    return np.ascontiguousarray(scaled.transpose(0, 3, 1, 2))
