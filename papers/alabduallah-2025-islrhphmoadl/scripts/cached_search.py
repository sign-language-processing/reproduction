"""Classifier-only CS-IGWO trials on immutable float32 pretrained features."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from data import load_manifest, sha256, write_json, MANIFEST
from experiment import (Recognizer, seed_all, rows_for, loader, train_epoch,
                        evaluate, environment, save_checkpoint, restore)


class Classifier(nn.Module):
    def __init__(self, model):
        super().__init__()
        # Transfer existing modules; do not redraw the seeded initialization.
        self.gru, self.dropout, self.head = model.gru, model.dropout, model.head

    def stage(self, unfrozen):
        self.train()

    def forward(self, sequence):
        _, hidden = self.gru(sequence)
        return self.head(self.dropout(torch.cat([hidden[0], hidden[1]], dim=1)))


class Features(Dataset):
    def __init__(self, path, rows):
        self.path, self.rows = str(path), rows
        self.array = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self.array is None:
            self.array = np.load(self.path, mmap_mode="r")
            if self.array.shape != (len(self.rows), 49, 3328) or self.array.dtype != np.float32:
                raise ValueError("Feature cache shape/dtype mismatch")
        return torch.from_numpy(self.array[index].copy()), self.rows[index]["label"]


def identity(rows):
    import hashlib
    return {"split_sha256": sha256(MANIFEST),
            "extractor_source_sha256": sha256(Path(__file__).with_name("experiment.py")),
            "cache_source_sha256": sha256(__file__),
            "rows_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()}


@torch.no_grad()
def extract(model, rows, path):
    model.eval()
    array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(rows), 49, 3328))
    offset, started = 0, time.monotonic()
    for x, _ in loader(rows):
        values = model.features(x.cuda(non_blocking=True)).cpu().numpy()
        array[offset:offset + len(values)] = values
        offset += len(values)
    array.flush()
    del array
    return time.monotonic() - started


@torch.no_grad()
def build_cache(output, limit_per_class=None):
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    rows = {s: rows_for(manifest, s, limit_per_class) for s in ("train", "validation")}
    expected = identity(rows)
    if (output / "metadata.json").exists():
        meta = json.loads((output / "metadata.json").read_text())
        if meta["identity"] != expected:
            raise ValueError("Incompatible feature cache identity")
        return meta
    seed_all()
    model = Recognizer().cuda().eval()
    env = environment(output)
    timings, hashes = {}, {}
    for split, records in rows.items():
        temporary = output / f"{split}.partial.npy"
        timings[split] = extract(model, records, temporary)
        path = output / f"{split}.npy"
        temporary.replace(path)
        hashes[split] = sha256(path)
    # Round-trip and direct/cached logits must agree for the same first batch.
    x, _ = next(iter(loader(rows["train"])))
    direct = model.features(x.cuda()).cpu()
    stored = torch.from_numpy(np.load(output / "train.npy", mmap_mode="r")[:len(x)].copy())
    if not torch.equal(direct, stored):
        raise AssertionError("Feature serialization changed values")
    classifier = Classifier(model).eval()
    if not torch.equal(model(x.cuda()), classifier(stored.cuda())):
        raise AssertionError("Cached and direct classifier logits differ")
    meta = {"identity": expected, "environment": env, "rows": rows,
            "extraction_seconds": timings, "array_sha256": hashes,
            "exact_features_and_logits": True}
    write_json(output / "metadata.json", meta)
    del model, classifier
    torch.cuda.empty_cache()
    return meta


def train_candidate(output, cache_path, params, epochs):
    output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((cache_path / "metadata.json").read_text())
    if meta["identity"] != identity(meta["rows"]):
        raise ValueError("Stale cache source/split identity")
    seed_all()
    original = Recognizer().cuda()
    model = Classifier(original)
    del original
    optimizer = torch.optim.SGD(model.parameters(), lr=params["head_lr"], momentum=.9)
    start = 1
    if (output / "last.pt").exists():
        checkpoint = restore(output / "last.pt", model, optimizer)
        if checkpoint["params"] != params:
            raise ValueError("Candidate resume parameters differ")
        start = checkpoint["epoch"] + 1
    train = Features(cache_path / "train.npy", meta["rows"]["train"])
    valid = Features(cache_path / "validation.npy", meta["rows"]["validation"])
    for epoch in range(start, epochs + 1):
        timing = train_epoch(model, optimizer, train, epoch)
        save_checkpoint(output / "last.pt", model, optimizer, epoch, params, None)
        write_json(output / f"epoch-{epoch:02d}.json", timing)
    result = evaluate(model, valid)
    result.update(params=params, epochs=epochs, conditional=True,
                  cache_identity=meta["identity"], checkpoint_sha256=sha256(output / "last.pt"))
    write_json(output / "metrics.json", result)
    return result


def preflight(output):
    if output.exists():
        raise FileExistsError("Use a fresh preflight run name")
    output.mkdir(parents=True)
    cache_path = output / "feature-cache"
    meta = build_cache(cache_path, limit_per_class=5)
    seed_all()
    original = Recognizer().cuda()
    model = Classifier(original)
    del original
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
    train = Features(cache_path / "train.npy", meta["rows"]["train"])
    valid = Features(cache_path / "validation.npy", meta["rows"]["validation"])
    timing = train_epoch(model, optimizer, train, 1, max_steps=20)
    params = {"head_lr": .01, "backbone_multiplier": .1}
    checkpoint = output / "resume.pt"
    save_checkpoint(checkpoint, model, optimizer, 1, params, None)
    first = train_epoch(model, optimizer, train, 2, max_steps=1)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    restore(checkpoint, model, optimizer)
    second = train_epoch(model, optimizer, train, 2, max_steps=1)
    exact = first["mean_loss"] == second["mean_loss"] and all(torch.equal(state[k], v.detach().cpu()) for k, v in model.state_dict().items())
    if not exact:
        raise AssertionError("Cached classifier resume is not exact")
    metrics = evaluate(model, valid)
    extraction = sum(meta["extraction_seconds"].values()) / sum(map(len, meta["rows"].values())) * 40600
    training = timing["steady_step_seconds"] * 7308 * 200
    evaluation = metrics["seconds"] / len(valid) * 4060 * 20
    result = {"kind": "cached_classifier_preflight", "environment": meta["environment"],
              "training": timing, "validation_metrics": metrics,
              "exact_features_and_logits": True, "resume_exact_next_step": exact,
              "checkpoint_sha256": sha256(checkpoint), "cache_metadata_sha256": sha256(cache_path / "metadata.json"),
              "estimate": {"extraction_seconds": extraction, "training_seconds": training,
                           "evaluation_seconds": evaluation,
                           "search_hours_with_25pct_overhead": (extraction + training + evaluation) / 3600 * 1.25,
                           "feature_cache_gib": 40600 * 49 * 3328 * 4 / 2**30,
                           "limits": "Tiny cache can fit in page cache; full shuffled 24.7 GiB cache may have lower I/O throughput. Includes 25% allowance, excludes full final training."}}
    write_json(output / "preflight.json", result)
    return result
