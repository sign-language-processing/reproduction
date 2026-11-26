"""Reimplementation of Kindiroglu et al. 2019, Table 5 row "Keyshot, FL-4, Single".

No code was released. Every choice marked "guess" below is recorded in
reproduction.json (gate taf-unstated-protocol, guesses_and_deviations).

Subcommands (run in order):
  seg    train the DeepLab v3+ hand segmentation model (Sec. 3.3)
  masks  detect keyframes and segment the keyshot frames of every General-subset video
  train  build the Temporal Accumulative Features and train/evaluate the CNN (Sec. 3.4)
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

BSIGN = Path("/datasets/bosphorussign22k")
RHD = Path("/datasets/rendered-handpose-dataset/RHD_v1-1/RHD_published_v2/training")
HGR = [  # (images, skin masks); in the masks skin is False/black (checked at the feature points)
    ("/datasets/hgr1/original_images", "/datasets/hgr1/skin_masks"),
    (
        "/datasets/hgr2a/hgr2a_images/original_images",
        "/datasets/hgr2a/hgr2a_skin/skin_masks",
    ),
    (
        "/datasets/hgr2b/hgr2b_images/original_images",
        "/datasets/hgr2b/hgr2b_skin/skin_masks",
    ),
]
W, H = 116, 64  # Sec. 3.1.1: heatmaps scaled to 116 x 64
K = 4  # FL-4: four keyframes kept (guess)
SHOT = 5  # Sec. 3.3: keyshot = five consecutive frames centered at the keyframe
SIGMA = 2.0  # guess: Gaussian standard deviation in heatmap pixels
# Sec. 3.1.1: left and right upper arms (guess: shoulders), elbows, wrists, hand tips, thumbs
JOINTS = [
    "ShoulderLeft",
    "ShoulderRight",
    "ElbowLeft",
    "ElbowRight",
    "WristLeft",
    "WristRight",
    "HandTipLeft",
    "HandTipRight",
    "ThumbLeft",
    "ThumbRight",
]
SEG_SIZE = 320  # guess: training resolution, the Rendered Handpose image size
FRAME_SIZE = (
    544,
    960,
)  # guess: video frames are segmented at half resolution (height, width)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def general_rows() -> list[dict]:
    with open(BSIGN / "BosphorusSign22k.csv", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["SubsetID"] == "General"]
    for row in rows:
        row["stem"] = f"{row['ClassID']}/{row['UserID']}_{row['RepeatID']}"
    return rows


def load_skeleton(stem: str):
    """Per joint: (T, 11) = camera x, y, z; orientation quaternion; color-image x, y; depth-image x, y."""
    import scipy.io

    return scipy.io.loadmat(
        BSIGN / "skeleton_kinect" / f"{stem}.mat",
        squeeze_me=True,
        struct_as_record=False,
    )["skeleton"]


# ---------------------------------------------------------------- keyframes (Sec. 3.2, HS+DC)


def keyframes(skeleton, k: int = K) -> list[int]:
    """k keyframes from the dominant hand's speed minima, chosen by density-peaks clustering.

    Guesses: hand = Kinect Hand joint in camera coordinates; dominant = longer path;
    density peaks (Rodriguez & Laio) on min-max normalized (frame, speed) points with a
    Gaussian kernel whose cutoff is the 2nd percentile of pairwise distances; the k + 2
    points with the largest density x distance, of which the first and last in time are dropped.
    """
    speeds = [
        np.linalg.norm(np.diff(getattr(skeleton, name)[:, :3], axis=0), axis=1)
        for name in ("HandLeft", "HandRight")
    ]
    speed = max(speeds, key=np.sum)  # speed[i] = movement between frames i and i + 1
    frames = np.array(
        [
            i
            for i in range(1, len(speed) - 1)
            if speed[i] <= speed[i - 1] and speed[i] <= speed[i + 1]
        ]
    )
    total = len(speed) + 1
    if len(frames) < k + 2:  # too few minima: evenly spaced frames instead
        return [int(round(x)) for x in np.linspace(0, total - 1, k + 2)[1:-1]]
    points = np.stack(
        [frames / (total - 1), speed[frames] / (speed.max() + 1e-12)], axis=1
    )
    dist = np.linalg.norm(points[:, None] - points[None], axis=2)
    cutoff = max(np.percentile(dist[np.triu_indices(len(points), 1)], 2), 1e-6)
    rho = np.exp(-((dist / cutoff) ** 2)).sum(axis=1)
    delta = np.array(
        [
            dist[i, rho > rho[i]].min() if (rho > rho[i]).any() else dist[i].max()
            for i in range(len(points))
        ]
    )
    chosen = np.sort(frames[np.argsort(-(rho * delta), kind="stable")[: k + 2]])
    return [int(frame) for frame in chosen[1:-1]]


def keyshot_frames(keys: list[int], total: int) -> list[int]:
    return [
        min(max(key + offset, 0), total - 1)
        for key in keys
        for offset in range(-(SHOT // 2), SHOT // 2 + 1)
    ]


# ---------------------------------------------------------------- features (Sec. 3.1)


def heatmaps(skeleton) -> np.ndarray:
    """(T, 11, H, W): Gaussians at the 10 joints' color-image positions, plus all 25 joints (background)."""
    ys, xs = np.mgrid[0:H, 0:W]

    def gaussian(name: str) -> np.ndarray:
        position = getattr(skeleton, name)[
            :, 7:9
        ]  # guess: Kinect's own projection to the 1920x1080 color image
        x = position[:, 0, None, None] * W / 1920
        y = position[:, 1, None, None] * H / 1080
        return np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * SIGMA**2))

    joints = [gaussian(name) for name in JOINTS]
    background = np.max([gaussian(name) for name in skeleton._fieldnames], axis=0)
    return np.nan_to_num(np.stack(joints + [background], axis=1)).astype(np.float32)


def taf(maps: np.ndarray, keys: list[int]) -> np.ndarray:
    """(11 * 7, H, W) hue-based accumulation.

    Segment n (between keyframes) has hue n / (K + 1) of the color circle, the paper's
    n x 180 / (K + 1) on OpenCV's 0-180 hue scale. Guesses: saturation rises linearly from
    1/len to 1 within a segment; value is the heatmap. Per joint, as in PoTion: the colored
    sum U (3 channels, divided by its maximum), the intensity I = sum of U over channels,
    and the normalized N = U / (I + 1).
    """
    total = len(maps)
    bounds = [0] + [key for key in keys] + [total]
    colors = np.zeros((total, 3), dtype=np.float32)
    for n in range(len(bounds) - 1):
        start, end = bounds[n], max(bounds[n + 1], bounds[n] + 1)
        for t in range(start, min(end, total)):
            colors[t] = colorsys.hsv_to_rgb(
                n / (len(keys) + 1), (t - start + 1) / (end - start), 1.0
            )
    u = np.einsum("tjhw,tc->jchw", maps, colors)
    u /= u.max(axis=(2, 3), keepdims=True) + 1e-12
    i = u.sum(axis=1, keepdims=True)
    return np.concatenate([u, i, u / (i + 1)], axis=1).reshape(-1, H, W)


# ---------------------------------------------------------------- hand segmentation (Sec. 3.3)


def seg_model():
    import segmentation_models_pytorch as smp

    # guess: backbone and ImageNet initialization; classes: 0 background, 1 left hand, 2 right hand
    return smp.DeepLabV3Plus("resnet50", encoder_weights="imagenet", classes=3)


class HandData(torch.utils.data.Dataset):
    """Rendered Handpose training images (left/right labels) and HGR (skin only, label 3 = some hand)."""

    def __init__(self, items: list[tuple[str, str, str]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        from PIL import Image

        kind, image_path, mask_path = self.items[index]
        image = (
            Image.open(image_path)
            .convert("RGB")
            .resize((SEG_SIZE, SEG_SIZE), Image.BILINEAR)
        )
        mask = np.array(
            Image.open(mask_path).resize((SEG_SIZE, SEG_SIZE), Image.NEAREST)
        )
        if kind == "rhd":  # 0 background, 1 person, 2-17 left hand, 18-33 right hand
            label = np.where(mask >= 18, 2, np.where(mask >= 2, 1, 0))
        else:
            label = np.where(mask, 0, 3)
        return torch.from_numpy(np.array(image)).permute(2, 0, 1), torch.from_numpy(
            label.astype(np.int64)
        )


def hand_items() -> list[tuple[str, str, str]]:
    items = [
        ("rhd", str(path), str(RHD / "mask" / path.name))
        for path in sorted((RHD / "color").glob("*.png"))
    ]
    for images, masks in HGR:
        items += [
            ("hgr", f"{images}/{name}", f"{masks}/{Path(name).stem}.bmp")
            for name in sorted(os.listdir(images))
        ]
    return items


def seg_loss(logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Cross-entropy; an HGR skin pixel (label 3) only says "left or right hand" (no side label is invented)."""
    log_p = F.log_softmax(logits.float(), dim=1)
    hand = torch.logsumexp(log_p[:, 1:], dim=1)
    picked = log_p.gather(1, label.clamp(max=2)[:, None])[:, 0]
    return -torch.where(label == 3, hand, picked).mean()


def seg(args) -> None:
    seed_all(0)
    items = hand_items()
    random.shuffle(items)
    held = max(1, round(0.01 * len(items)))  # Sec. 3.3: 1% of images for validation
    validation, training = items[:held], items[held:]
    if args.limit:
        training, validation = (
            training[: args.limit],
            validation[: max(8, args.limit // 10)],
        )

    def loader(data, shuffle):
        return torch.utils.data.DataLoader(
            HandData(data),
            batch_size=32,
            shuffle=shuffle,
            num_workers=args.workers,
            drop_last=shuffle,
        )

    model = seg_model().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # guess
    scaler = torch.amp.GradScaler()
    mean, std = MEAN.cuda(), STD.cuda()
    log = []
    for epoch in range(
        args.epochs
    ):  # guess: 20 epochs, no augmentation (a mirrored left hand is a right hand)
        model.train()
        started, losses = time.time(), []
        for image, label in loader(training, True):
            image = (image.cuda().float() / 255 - mean) / std
            with torch.autocast("cuda"):
                logits = model(image)
            loss = seg_loss(logits, label.cuda())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        model.eval()
        inter, union = np.zeros(3), np.zeros(3)
        with torch.no_grad():
            for image, label in loader(validation, False):
                with torch.autocast("cuda"):
                    predicted = (
                        model((image.cuda().float() / 255 - mean) / std).argmax(1).cpu()
                    )
                side = (
                    label < 3
                )  # IoU on pixels with a side label, i.e. Rendered Handpose images only
                for c in range(3):
                    inter[c] += ((predicted == c) & (label == c) & side).sum().item()
                    union[c] += (((predicted == c) | (label == c)) & side).sum().item()
        iou = (inter / np.maximum(union, 1)).tolist()
        log.append(
            {
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "iou_background_left_right": iou,
                "seconds": time.time() - started,
                "images_per_second": len(training) / (time.time() - started),
                "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
        )
        print(json.dumps(log[-1]), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / "seg.pt")
    (args.out / "seg.json").write_text(
        json.dumps(
            {
                "training_images": len(training),
                "validation_images": len(validation),
                "epochs": log,
            },
            indent=1,
        )
    )


def masks(args) -> None:
    """Per video: keyframes and the 20 keyshot hand masks at 116 x 64 (both hands, binary)."""
    from simple_video_utils.frames import read_frames_exact
    from simple_video_utils.metadata import open_video

    rows = general_rows()[: args.limit or None]
    model = seg_model().cuda().eval()
    model.load_state_dict(torch.load(args.out / "seg.pt"))
    mean, std = MEAN.cuda(), STD.cuda()
    out = args.out / "masks"
    started, hand_fraction = time.time(), []
    for count, row in enumerate(rows):
        target = out / f"{row['stem']}.npz"
        if target.exists():
            continue
        skeleton = load_skeleton(row["stem"])
        total = len(skeleton.HandRight)
        keys = keyframes(skeleton)
        wanted = keyshot_frames(keys, total)
        with open_video(
            str(BSIGN / "videos" / f"{row['stem']}.mp4"), thread_type="AUTO"
        ) as video:
            frames = {
                index: frame
                for frame, index in read_frames_exact(video, return_indices=True)
                if index in wanted
            }
        batch = (
            torch.from_numpy(np.stack([frames[index] for index in wanted]))
            .cuda()
            .permute(0, 3, 1, 2)
            .float()
            / 255
        )
        batch = F.interpolate(batch, size=FRAME_SIZE, mode="bilinear", antialias=True)
        with torch.no_grad(), torch.autocast("cuda"):
            hand = (model((batch - mean) / std).argmax(1) > 0).float()[:, None]
        # Sec. 3.3: "resized, converted to binary masks"; guess: whole frame, both hands, any hand pixel in a cell
        small = (F.adaptive_max_pool2d(hand, (H, W))[:, 0] > 0).cpu().numpy()
        hand_fraction.append(float(hand.mean()))
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target, keys=np.array(keys), frames=np.array(wanted), masks=small
        )
        if count % 200 == 0:
            print(
                count,
                len(rows),
                round(time.time() - started),
                "s; mean hand fraction",
                np.mean(hand_fraction),
                flush=True,
            )
    print(
        json.dumps(
            {
                "videos": len(rows),
                "seconds": time.time() - started,
                "mean_hand_pixel_fraction": float(np.mean(hand_fraction or [0])),
            }
        ),
        flush=True,
    )


# ---------------------------------------------------------------- classifier (Sec. 3.4)


def classifier(channels: int, classes: int) -> nn.Module:
    """Table 3's row "0.25, 2 blocks, 2 convs, 128 filters": strides 2 and 1, filters doubled per block.

    Guesses: 'same' padding, ReLU, dropout after every convolution (as in PoTion), one linear layer
    on the pooled features. Keras defaults are mirrored: Glorot-uniform weights, zero biases,
    batch-norm epsilon 1e-3 and momentum 0.99.
    """
    layers, width = [], 128
    for _ in range(2):
        for stride in (2, 1):
            layers += [
                nn.Conv2d(channels, width, 3, stride, 1),
                nn.BatchNorm2d(width, eps=1e-3, momentum=0.01),
                nn.ReLU(),
                nn.Dropout(0.25),
            ]
            channels = width
        width *= 2
    model = nn.Sequential(
        *layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, classes)
    )
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    return model


def features(row: dict, mask_dir: Path) -> np.ndarray:
    saved = np.load(mask_dir / f"{row['stem']}.npz")
    skeleton = load_skeleton(row["stem"])
    keys = [int(key) for key in saved["keys"]]
    assert keys == keyframes(skeleton), row["stem"]
    return np.concatenate(
        [taf(heatmaps(skeleton), keys), saved["masks"].astype(np.float32)]
    ).astype(np.float16)


def train(args) -> None:
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    rows = general_rows()
    if args.limit:
        rows = [
            row for row in rows if (args.out / "masks" / f"{row['stem']}.npz").exists()
        ]
    classes = sorted({row["ClassID"] for row in rows})
    with ProcessPoolExecutor(args.workers) as pool:
        data = np.stack(
            list(
                pool.map(
                    partial(features, mask_dir=args.out / "masks"), rows, chunksize=16
                )
            )
        )
    labels = np.array([classes.index(row["ClassID"]) for row in rows])
    test = np.array(
        [row["UserID"] == "User_4" for row in rows]
    )  # Sec. 4.1: user 4 is the test user
    print(
        "features",
        data.shape,
        "train",
        int((~test).sum()),
        "test",
        int(test.sum()),
        "classes",
        len(classes),
        flush=True,
    )
    if not args.limit:
        assert data.shape[1:] == (97, H, W) and (
            (~test).sum(),
            test.sum(),
            len(classes),
        ) == (4839, 949, 174)
    x_train, y_train = (
        torch.from_numpy(data[~test]).cuda(),
        torch.from_numpy(labels[~test]).cuda(),
    )
    x_test, y_test = (
        torch.from_numpy(data[test]).cuda().float(),
        torch.from_numpy(labels[test]),
    )

    seed_all(args.seed)
    model = classifier(data.shape[1], len(classes)).cuda()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, eps=1e-7
    )  # Keras defaults (guess: default learning rate)
    log, started = [], time.time()

    def evaluate() -> tuple[float, float, torch.Tensor]:
        model.eval()
        with torch.no_grad():
            scores = torch.cat(
                [model(x_test[i : i + 256]) for i in range(0, len(x_test), 256)]
            ).cpu()
        top5 = scores.topk(min(5, scores.shape[1]), dim=1).indices
        return (
            float((top5[:, 0] == y_test).float().mean() * 100),
            float((top5 == y_test[:, None]).any(1).float().mean() * 100),
            scores,
        )

    for epoch in range(args.epochs):  # guess: 100 epochs, weights after the last epoch
        model.train()
        order = torch.randperm(len(x_train), device="cuda")
        losses = []
        for i in range(0, len(order), 32):  # Sec. 3.4: batch size 32
            index = order[i : i + 32]
            batch = x_train[index].float()
            flip = (
                torch.rand(len(index), device="cuda") < 0.5
            )  # Sec. 3.4: random horizontal flip
            batch = torch.where(flip[:, None, None, None], batch.flip(3), batch)
            loss = F.cross_entropy(model(batch), y_train[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        accuracy, top5, _ = evaluate()
        log.append(
            {
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "test_accuracy": accuracy,
                "test_top5": top5,
            }
        )
        print(json.dumps(log[-1]), flush=True)
    accuracy, top5, scores = evaluate()
    out = args.out / f"classifier-seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    np.savez_compressed(
        out / "scores.npz",
        scores=scores.numpy(),
        labels=y_test.numpy(),
        stems=np.array([row["stem"] for row, held in zip(rows, test) if held]),
    )
    reloaded = classifier(data.shape[1], len(classes)).cuda()
    reloaded.load_state_dict(torch.load(out / "model.pt"))
    model = reloaded
    assert evaluate()[:2] == (accuracy, top5), (
        "saved weights do not reproduce the evaluation"
    )
    metrics = {
        "seed": args.seed,
        "epochs": args.epochs,
        "accuracy": accuracy,
        "top5_accuracy": top5,
        "test_videos": int(test.sum()),
        "train_videos": int((~test).sum()),
        "classes": len(classes),
        "input_shape": list(data.shape[1:]),
        "train_seconds": time.time() - started,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "per_epoch": log,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(
        json.dumps(
            {key: value for key, value in metrics.items() if key != "per_epoch"}
        ),
        flush=True,
    )


def self_check() -> None:
    """Keyframe and feature logic on a synthetic skeleton: three holds between movements."""
    from types import SimpleNamespace

    path = np.concatenate(
        [
            np.linspace(a, b, 15)
            for a, b in ((0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 0))
        ]
    )
    joint = np.zeros((len(path), 11))
    joint[:, 0] = path + np.random.default_rng(0).normal(0, 1e-3, len(path))
    joint[:, 7:9] = [960, 540]
    names = JOINTS + ["HandLeft", "HandRight"]
    still = joint * ([0] * 7 + [1, 1, 0, 0])  # only the image position
    skeleton = SimpleNamespace(
        _fieldnames=names,
        **{name: joint if name == "HandRight" else still for name in names},
    )
    keys = keyframes(skeleton)
    assert (
        len(keys) == K
        and keys == sorted(keys)
        and 0 < keys[0]
        and keys[-1] < len(path) - 1
    ), keys
    assert len(keyshot_frames(keys, len(path))) == K * SHOT
    maps = heatmaps(skeleton)
    assert maps.shape == (len(path), 11, H, W) and abs(maps[0, 0, 32, 58] - 1) < 0.1
    feature = taf(maps, keys)
    assert (
        feature.shape == (77, H, W)
        and np.isfinite(feature).all()
        and feature.max() <= 3.0 + 1e-6
    )
    assert classifier(97, 174)(torch.zeros(2, 97, H, W)).shape == (2, 174)
    print("self-check ok; keyframes", keys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["seg", "masks", "train", "self-check"])
    parser.add_argument("--out", type=Path, default=Path("/results"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, default=0, help="preflight only: use this many items"
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.epochs = args.epochs or {"seg": 20, "train": 100}.get(args.stage)
    {"seg": seg, "masks": masks, "train": train, "self-check": lambda _: self_check()}[
        args.stage
    ](args)
