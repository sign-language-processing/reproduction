"""Train and evaluate the paper's ResNet-18 + LSTM classifier on LSA64.

This is a clean-room implementation: the paper has no released executable code.
It deliberately decodes videos on demand instead of materialising extracted frames.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from simple_video_utils.frames import read_frames_exact
from simple_video_utils.metadata import open_video, video_metadata_from_container
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


# Paper §4.1 and §4.4: each input clip contains 16 RGB frames at 128×128.
IMAGE_SIZE = 128
FRAME_COUNT = 16
# Decision (documented in README): §4.1 requires a random crop but not its
# resize policy. Resize the short side to 144 before a 128×128 crop.
RESIZE_SHORT_SIDE = 144
CLASS_COUNT = 64
# Decision (documented in README): the paper specifies 8/2 signers but not IDs.
HELD_OUT_SIGNERS = (5, 10)
# Decision (documented in README): use ImageNet normalization because the paper
# says ImageNet pre-training but omits the preprocessing constants.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    signer: int
    repetition: int


def parse_sample(path: Path) -> Sample:
    """Parse the official LSA64 `class_signer_repetition.mp4` convention."""
    try:
        class_id, signer, repetition = (int(part) for part in path.stem.split("_"))
    except ValueError as error:
        raise ValueError(f"unexpected LSA64 filename: {path.name}") from error
    if not 1 <= class_id <= CLASS_COUNT or not 1 <= signer <= 10 or not 1 <= repetition <= 5:
        raise ValueError(f"out-of-range LSA64 filename: {path.name}")
    return Sample(path=path, label=class_id - 1, signer=signer, repetition=repetition)


def list_samples(data_root: Path) -> list[Sample]:
    samples = [parse_sample(path) for path in sorted(data_root.rglob("*.mp4"))]
    if len(samples) != 3200:
        raise ValueError(f"expected 3200 LSA64 videos under {data_root}, found {len(samples)}")
    identities = {(sample.label, sample.signer, sample.repetition) for sample in samples}
    if len(identities) != 3200:
        raise ValueError("LSA64 contains duplicate or missing class/signer/repetition identities")
    return samples


def split_samples(samples: list[Sample]) -> tuple[list[Sample], list[Sample]]:
    # Paper §§4.2 and 4.5 specify only an 8/2 learner split and 2,560/640
    # clips; HELD_OUT_SIGNERS is therefore a documented reconstruction choice.
    train = [sample for sample in samples if sample.signer not in HELD_OUT_SIGNERS]
    validation = [sample for sample in samples if sample.signer in HELD_OUT_SIGNERS]
    train_ids = {(sample.label, sample.signer, sample.repetition) for sample in train}
    validation_ids = {(sample.label, sample.signer, sample.repetition) for sample in validation}
    if train_ids & validation_ids or len(train) != 2560 or len(validation) != 640:
        raise ValueError("invalid LSA64 signer split")
    if {sample.signer for sample in train} & set(HELD_OUT_SIGNERS):
        raise ValueError("held-out signer leaked into training split")
    if any(sum(sample.label == label for sample in train) != 40 or sum(sample.label == label for sample in validation) != 10 for label in range(CLASS_COUNT)):
        raise ValueError("LSA64 class counts do not match the 8/2 signer split")
    return train, validation


def uniformly_sample_frames(path: Path) -> Tensor:
    """Decode exactly 16 frames; uniform positions are a documented inference."""
    # DataLoader supplies process-level parallelism; nested codec threads would
    # oversubscribe the assigned CPUs and can deadlock after a worker fork.
    with open_video(str(path), thread_type="NONE") as video:
        metadata = video_metadata_from_container(video)
        if not metadata.nb_frames or metadata.nb_frames < 1:
            raise ValueError(f"video has no decodable frames: {path}")
        indices = np.rint(np.linspace(0, metadata.nb_frames - 1, FRAME_COUNT)).astype(int)
        wanted = set(indices.tolist())
        retained: dict[int, np.ndarray] = {}
        for frame, index in read_frames_exact(video, return_indices=True):
            if index in wanted:
                retained[index] = frame
        if any(index not in retained for index in indices):
            raise ValueError(f"failed to decode all requested frames from {path}")
    return torch.from_numpy(np.stack([retained[index] for index in indices])).permute(0, 3, 1, 2)


def preprocess(video: Tensor, training: bool) -> Tensor:
    """Apply one temporally consistent crop and ImageNet normalization."""
    video = video.float().div_(255)
    _, _, height, width = video.shape
    if height <= width:
        resized_height, resized_width = RESIZE_SHORT_SIDE, round(width * RESIZE_SHORT_SIDE / height)
    else:
        resized_height, resized_width = round(height * RESIZE_SHORT_SIDE / width), RESIZE_SHORT_SIDE
    video = F.interpolate(video, size=(resized_height, resized_width), mode="bilinear", align_corners=False, antialias=True)
    max_top, max_left = resized_height - IMAGE_SIZE, resized_width - IMAGE_SIZE
    if training:
        # Paper §§4.1 and 4.4 require random crops; see README for this crop policy.
        top = int(torch.randint(max_top + 1, ()).item())
        left = int(torch.randint(max_left + 1, ()).item())
    else:
        top, left = max_top // 2, max_left // 2
    video = video[:, :, top : top + IMAGE_SIZE, left : left + IMAGE_SIZE]
    mean = video.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = video.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (video - mean) / std


class LSA64Videos(Dataset[tuple[Tensor, int]]):
    def __init__(self, samples: list[Sample], training: bool) -> None:
        self.samples = samples
        self.training = training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        sample = self.samples[index]
        return preprocess(uniformly_sample_frames(sample.path), self.training), sample.label


class ResNet18LSTM(nn.Module):
    """ImageNet-initialized ResNet-18 frame encoder followed by an LSTM classifier."""
    def __init__(self) -> None:
        super().__init__()
        # Paper §3.5 identifies ResNet-18 and ImageNet pre-training.
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        # Paper §4.4 says to retain/freeze the pretrained hidden-layer parameters.
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        # Paper §4.4 says the modified fully connected layer feeds the LSTM.
        # Decision (documented in README): its width is unreported; use 512.
        self.frame_projection = nn.Linear(backbone.fc.in_features, 512)
        # Decision (documented in README): the paper does not state LSTM depth;
        # use one 512-unit layer matching the modified projection width.
        self.lstm = nn.LSTM(input_size=512, hidden_size=512, batch_first=True)
        self.classifier = nn.Linear(512, CLASS_COUNT)

    def forward(self, video: Tensor) -> Tensor:
        batch, steps, channels, height, width = video.shape
        features = self.frame_projection(self.encoder(video.reshape(batch * steps, channels, height, width)).flatten(1))
        sequence, _ = self.lstm(features.reshape(batch, steps, -1))
        return self.classifier(sequence[:, -1])


def metrics(predictions: list[int], labels: list[int]) -> dict[str, float]:
    prediction_tensor = torch.tensor(predictions)
    label_tensor = torch.tensor(labels)
    confusion = torch.bincount(CLASS_COUNT * label_tensor + prediction_tensor, minlength=CLASS_COUNT**2).reshape(CLASS_COUNT, CLASS_COUNT).float()
    true_positive = confusion.diag()
    precision = true_positive / confusion.sum(0).clamp_min(1)
    recall = true_positive / confusion.sum(1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(torch.finfo(torch.float32).eps)
    return {
        "accuracy": float((prediction_tensor == label_tensor).float().mean().item() * 100),
        "macro_f1": float(f1.mean().item() * 100),
        "macro_precision": float(precision.mean().item() * 100),
        "macro_recall": float(recall.mean().item() * 100),
    }


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader[tuple[Tensor, Tensor]], device: torch.device) -> dict[str, float]:
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    for video, target in loader:
        predictions.extend(model(video.to(device, non_blocking=True)).argmax(1).cpu().tolist())
        labels.extend(target.tolist())
    return metrics(predictions, labels)


def checkpoint(path: Path, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau, progress: dict[str, float]) -> None:
    torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "progress": progress}, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--resume", type=Path)
    arguments = parser.parse_args()

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    # Decision (documented in README): prefer repeatability over cuDNN autotuning.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    samples = list_samples(arguments.data_root)
    train_samples, validation_samples = split_samples(samples)
    loader_options = {"num_workers": arguments.workers, "pin_memory": device.type == "cuda", "persistent_workers": arguments.workers > 0}
    if arguments.workers:
        loader_options["prefetch_factor"] = 2
    train_loader = DataLoader(LSA64Videos(train_samples, training=True), batch_size=arguments.batch_size, shuffle=True, **loader_options)
    validation_loader = DataLoader(LSA64Videos(validation_samples, training=False), batch_size=arguments.batch_size, shuffle=False, **loader_options)

    model = ResNet18LSTM().to(device)
    # Paper §4.4: Adam, learning rate 1e-4, weight decay 5e-4, label smoothing.
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-4, weight_decay=5e-4)
    # Paper §4.4 specifies ReduceLROnPlateau, threshold 1e-4, patience 5, and
    # factor 0.1. It does not name its monitor; train loss avoids adapting to
    # the final held-out set.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", threshold=1e-4, patience=5, factor=0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    first_epoch = 1
    if arguments.resume:
        state = torch.load(arguments.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        first_epoch = int(state["epoch"]) + 1
    if first_epoch > arguments.epochs:
        raise ValueError("resume checkpoint already meets the requested epoch count")

    metrics_path = arguments.output_dir / "metrics.jsonl"
    with metrics_path.open("a" if arguments.resume else "x", encoding="utf-8") as metrics_file:
        for epoch in range(first_epoch, arguments.epochs + 1):
            model.train()
            # Freezing includes BatchNorm running statistics, not only gradients.
            model.encoder.eval()
            losses: list[float] = []
            for video, target in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(video.to(device, non_blocking=True)), target.to(device, non_blocking=True))
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            progress = {"epoch": epoch, "train_loss": float(np.mean(losses)), "learning_rate": optimizer.param_groups[0]["lr"]}
            scheduler.step(progress["train_loss"])
            checkpoint(arguments.output_dir / "last.pt", epoch, model, optimizer, scheduler, progress)
            print(json.dumps(progress, sort_keys=True), flush=True)
        # Paper Table 3 reports fixed epochs; evaluate the declared final epoch,
        # rather than selecting a checkpoint or tuning on the held-out clips.
        result = {**progress, **evaluate(model, validation_loader, device)}
        metrics_file.write(json.dumps(result, sort_keys=True) + "\n")
        metrics_file.flush()
    checkpoint(arguments.output_dir / f"epoch-{arguments.epochs}.pt", arguments.epochs, model, optimizer, scheduler, result)
    run = {"seed": arguments.seed, "held_out_signers": HELD_OUT_SIGNERS, "train_samples": len(train_samples), "validation_samples": len(validation_samples), "epochs": arguments.epochs, "batch_size": arguments.batch_size, "encoder": "ImageNet ResNet-18 frozen per paper §4.4", "scheduler_monitor": "training loss (paper does not specify monitor)", "final": result}
    (arguments.output_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
