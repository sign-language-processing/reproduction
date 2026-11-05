"""Conditional reimplementation of the three ASL Alphabet autoencoders (arXiv:2501.06942v1).

The paper publishes no code. Values marked GUESS are not reported by the paper and are
recorded in reproduction.json (guesses) and README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

CLASSES = [*"ABCDEFGHIJKLMNOPQRSTUVWXYZ", "del", "nothing", "space"]  # 29 classes (Section 2.1)
SIZE = 200  # Section 2.2
LATENT = 64  # Section 3.4, Figure 1


def load_split(root: Path, per_class: int, seed: int, grayscale: bool):
    """500 images per class, then an 80/20 random split (Section 4, Section 5.1)."""
    rng = random.Random(seed)  # GUESS: the paper's seed value and selection procedure
    files = []
    for c in CLASSES:
        names = sorted(p.name for p in (root / c).iterdir())
        files += [f"{c}/{n}" for n in rng.sample(names, per_class)]
    order = torch.randperm(len(files), generator=torch.Generator().manual_seed(seed)).tolist()
    n_train = int(0.8 * len(files))
    train, val = [files[i] for i in order[:n_train]], [files[i] for i in order[n_train:]]

    def tensor(names):
        mode = "L" if grayscale else "RGB"
        arr = np.stack([np.asarray(Image.open(root / n).convert(mode).resize((SIZE, SIZE))) for n in names])
        t = torch.from_numpy(arr)
        return (t.unsqueeze(1) if grayscale else t.permute(0, 3, 1, 2)).contiguous()  # uint8 NCHW

    return train, val, tensor(train), tensor(val)


class Feedforward(nn.Module):
    # Figure 1: flattened 40,000-value input (200x200x1), two hidden layers per side, 64-d latent.
    HIDDEN = (1024, 256)  # GUESS: widths are not reported

    def __init__(self):
        super().__init__()
        a, b = self.HIDDEN
        d = SIZE * SIZE
        self.encoder = nn.Sequential(nn.Linear(d, a), nn.ReLU(), nn.Linear(a, b), nn.ReLU(), nn.Linear(b, LATENT))
        self.decoder = nn.Sequential(nn.Linear(LATENT, b), nn.ReLU(), nn.Linear(b, a), nn.ReLU(), nn.Linear(a, d), nn.Sigmoid())

    def forward(self, x):
        return self.decoder(self.encoder(x.flatten(1))).view_as(x)


def conv_encoder():
    # Section 3.4: three 4x4/stride-2/pad-1 convolutions 3->32->64->128, 200->100->50->25.
    return nn.Sequential(
        nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
        nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
        nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
        nn.Flatten(),
    )


def conv_decoder():
    # Section 3.4: linear layer then transposed convolutions; ReLU except the sigmoid output.
    return nn.Sequential(
        nn.Linear(LATENT, 128 * 25 * 25), nn.ReLU(),  # GUESS: ReLU after the decoder linear layer
        nn.Unflatten(1, (128, 25, 25)),
        nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
        nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
        nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
    )


class Convolutional(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(conv_encoder(), nn.Linear(128 * 25 * 25, LATENT))
        self.decoder = conv_decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Diffusion(nn.Module):
    """Section 3.6: conv encoder -> (mean, log-variance) -> latent diffusion over 100 steps ->
    fully connected noise predictor -> conv decoder from the denoised latent."""

    T = 100  # Section 3.6
    BETAS = (1e-4, 0.02)  # GUESS: linear DDPM schedule; the paper says only "a beta schedule"
    WIDTH = 256  # GUESS: denoiser width; "five primary layers" kept, "dynamically added layers" omitted

    def __init__(self):
        super().__init__()
        self.features = conv_encoder()
        self.mu = nn.Linear(128 * 25 * 25, LATENT)
        self.logvar = nn.Linear(128 * 25 * 25, LATENT)
        w = self.WIDTH
        layers, d = [], LATENT + 1  # latent plus normalised timestep
        for _ in range(5):
            layers += [nn.Linear(d, w), nn.ReLU()]
            d = w
        self.denoiser = nn.Sequential(*layers, nn.Linear(w, LATENT))
        self.decoder = conv_decoder()
        self.register_buffer("abar", torch.cumprod(1 - torch.linspace(*self.BETAS, self.T), 0))

    def forward(self, x, generator=None):
        h = self.features(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.exp(0.5 * logvar) * torch.randn(mu.shape, device=x.device, generator=generator)
        t = torch.randint(0, self.T, (x.shape[0],), device=x.device, generator=generator)
        noise = torch.randn(z.shape, device=x.device, generator=generator)
        a = self.abar[t].unsqueeze(1)
        z_t = a.sqrt() * z + (1 - a).sqrt() * noise
        pred = self.denoiser(torch.cat([z_t, (t.float() / self.T).unsqueeze(1)], 1))
        z0 = (z_t - (1 - a).sqrt() * pred) / a.sqrt()
        return self.decoder(z0), noise, pred


MODELS = {"feedforward": Feedforward, "convolutional": Convolutional, "diffusion": Diffusion}


def losses(model, x, generator=None):
    if isinstance(model, Diffusion):
        recon, noise, pred = model(x, generator)
        # GUESS: reconstruction MSE plus noise-prediction MSE, no KL term (Section 4 names only MSE).
        mse = F.mse_loss(recon, x)
        return mse, mse + F.mse_loss(pred, noise)
    recon = model(x)
    mse = F.mse_loss(recon, x)
    return mse, mse


@torch.no_grad()
def evaluate(model, data, batch, device, seed):
    """MSE over every pixel and channel of the validation set (sum of squared errors / elements)."""
    model.eval()
    g = torch.Generator(device=device).manual_seed(seed)  # fixes the diffusion model's sampled noise
    sse, n = 0.0, 0
    for i in range(0, len(data), batch):
        x = data[i : i + batch].to(device).float() / 255
        recon = model(x, g)[0] if isinstance(model, Diffusion) else model(x)
        sse += F.mse_loss(recon, x, reduction="sum").item()
        n += x.numel()
    return sse / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--per-class", type=int, default=500)  # Section 4
    p.add_argument("--epochs", type=int, default=20)  # GUESS
    p.add_argument("--batch-size", type=int, default=32)  # GUESS
    p.add_argument("--lr", type=float, default=1e-3)  # GUESS: Adam default
    p.add_argument("--seed", type=int, default=42)  # GUESS
    p.add_argument("--max-steps", type=int, default=0, help="preflight only: stop each epoch after N steps")
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    start = time.time()

    random.seed(a.seed), np.random.seed(a.seed), torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grayscale = a.model == "feedforward"  # Figure 1: 40,000-value input
    train_names, val_names, train, val = load_split(a.data_root, a.per_class, a.seed, grayscale)
    split = {"train": train_names, "validation": val_names}
    (a.output_dir / "split.json").write_text(json.dumps(split))

    model = MODELS[a.model]().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    params = sum(t.numel() for t in model.parameters())
    log = []
    for epoch in range(1, a.epochs + 1):
        model.train()
        order = torch.randperm(len(train))
        total, steps = 0.0, 0
        for i in range(0, len(train), a.batch_size):
            x = train[order[i : i + a.batch_size]].to(device).float() / 255
            mse, loss = losses(model, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += mse.item()
            steps += 1
            if a.max_steps and steps >= a.max_steps:
                break
        log.append({"epoch": epoch, "train_mse": total / steps, "steps": steps, "elapsed_s": round(time.time() - start, 1)})
        print(json.dumps(log[-1]), flush=True)

    ckpt = a.output_dir / "model.pt"
    torch.save(model.state_dict(), ckpt)  # Section 4: weights saved after training (final epoch)
    reloaded = MODELS[a.model]().to(device)
    reloaded.load_state_dict(torch.load(ckpt, map_location=device))
    val_mse = evaluate(reloaded, val, 64, device, a.seed)

    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    result = {
        "model": a.model, "validation_mse": val_mse, "validation_images": len(val), "train_images": len(train),
        "grayscale": grayscale, "parameters": params, "epochs": a.epochs, "batch_size": a.batch_size, "lr": a.lr,
        "seed": a.seed, "max_steps": a.max_steps, "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "torch": torch.__version__, "wall_time_s": round(time.time() - start, 1),
        "split_sha256": sha(a.output_dir / "split.json"), "checkpoint_sha256": sha(ckpt), "epochs_log": log,
    }
    (a.output_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "epochs_log"}))


if __name__ == "__main__":
    main()
