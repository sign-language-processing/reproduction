"""Durable native evaluation batches, enabled only around a final checkpoint test.

Cache files are trusted outputs of this reproduction, not downloaded pickles.
Loss tensors retain their original device/dtype; NumPy predictions/attention are
serialized unchanged. The native evaluator still aggregates every batch in order.
"""
from contextvars import ContextVar
from functools import wraps
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import tempfile

_ACTIVE = ContextVar("how2_evaluation_recovery", default=None)
_OPTIONS = (
    "batch_size", "batch_type", "use_cuda", "sgn_dim", "do_recognition",
    "recognition_loss_weight", "do_translation", "translation_loss_weight",
    "translation_max_output_length", "level", "txt_pad_index",
    "recognition_beam_size", "translation_beam_size", "translation_beam_alpha",
    "dataset_version", "frame_subsampling_ratio",
)
_RECOVERY_KEYS = ("load_model", "repro_resume", "repro_recovery_dir")


def _hash_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".batch-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load(path):
    import torch
    with path.open("rb") as stream:
        checksum = stream.readline().rstrip(b"\n")
        payload = stream.read()
    if hashlib.sha256(payload).hexdigest().encode() != checksum:
        raise RuntimeError("Evaluation cache checksum mismatch: " + str(path))
    # Legacy PyTorch lacks weights_only; newer versions need false for native NumPy arrays.
    options = {"weights_only": False} if "weights_only" in inspect.signature(torch.load).parameters else {}
    return torch.load(io.BytesIO(payload), **options)


def _save(path, record):
    import torch
    stream = io.BytesIO()
    torch.save(record, stream)
    payload = stream.getvalue()
    _atomic_write(path, hashlib.sha256(payload).hexdigest().encode() + b"\n" + payload)


def with_evaluation_recovery(test, cache_root, dataset_identity):
    """Wrap recognition-cache(test); dataset_identity must name verified input hashes.

    Only three training recovery keys are excluded from canonical config identity.
    The selected checkpoint, every scientific config field and actual dataset
    order/vocabulary remain bound. An explicit checkpoint is required.
    """
    if not dataset_identity:
        raise ValueError("Evaluation recovery requires verified dataset identity")
    # Validate and copy the JSON identity before any expensive evaluation starts.
    dataset_identity = json.loads(json.dumps(dataset_identity, sort_keys=True))

    @wraps(test)
    def run(*args, **kwargs):
        import yaml
        call = inspect.signature(test).bind(*args, **kwargs)
        call.apply_defaults()
        checkpoint = call.arguments.get("ckpt")
        if checkpoint is None:
            raise ValueError("Recovery requires the native selected checkpoint path")
        config = yaml.safe_load(Path(call.arguments["cfg_file"]).read_text())
        for key in _RECOVERY_KEYS:
            config.get("training", {}).pop(key, None)
        identity = {
            "format": 1,
            "checkpoint_sha256": _hash_file(checkpoint),
            "config_sha256": _digest(config),
            "dataset_identity": dataset_identity,
            "evaluator_source_sha256": _hash_file(inspect.getsourcefile(inspect.unwrap(test))),
            "recovery_source_sha256": _hash_file(__file__),
        }
        context = {
            "root": Path(cache_root) / _digest(identity), "identity": identity,
            "datasets": {}, "hits": 0, "computed": 0,
        }
        context["root"].mkdir(parents=True, exist_ok=True)
        _atomic_write(context["root"] / "identity.json", json.dumps(identity, indent=2).encode())
        token = _ACTIVE.set(context)
        try:
            return test(*args, **kwargs)
        finally:
            _ACTIVE.reset(token)
            print(json.dumps({"evaluation_recovery": {
                "identity_sha256": _digest(identity),
                "completed_batch_hits": context["hits"],
                "new_completed_batches": context["computed"],
            }}), flush=True)
    return run


def model_for_validation(model, data, arguments):
    """Return the exact native model when recovery is inactive (including training)."""
    context = _ACTIVE.get()
    if context is None:
        return model
    if id(data) not in context["datasets"]:
        records = {"sequence": list(data.sequence)}
        for field in ("txt", "gls"):
            if hasattr(data, field):
                records[field] = [list(tokens) for tokens in getattr(data, field)]
        context["datasets"][id(data)] = _digest(records)
    identity = {name: arguments[name] for name in _OPTIONS}
    identity["dataset_order_sha256"] = context["datasets"][id(data)]
    identity["vocab_sha256"] = _digest({
        name: list(getattr(model, name).itos) for name in ("gls_vocab", "txt_vocab")
    })
    identity["loss_functions"] = {
        name: repr(arguments[name])
        for name in ("recognition_loss_function", "translation_loss_function")
    }
    identity["model_do_recognition"] = model.do_recognition
    identity["model_do_translation"] = model.do_translation
    return _BatchModel(model, context, identity)


class _BatchModel:
    def __init__(self, model, context, identity):
        self.model = model
        self.context = context
        self.root = context["root"] / _digest(identity)
        self.identity = identity
        self.index = -1
        self.record = None
        self.losses = None

    def start_batch(self, batch, reverse_index):
        self.index += 1
        self.path = self.root / ("batch-%06d.pt" % self.index)
        self.batch_identity = {
            "sequence": list(batch.sequence),
            "lengths": batch.sgn_lengths.detach().cpu().tolist(),
            "reverse_index": list(reverse_index),
            "num_seqs": int(batch.num_seqs),
            "num_txt_tokens": None if batch.num_txt_tokens is None else int(batch.num_txt_tokens),
            "num_gls_tokens": None if batch.num_gls_tokens is None else int(batch.num_gls_tokens),
        }
        self.record = _load(self.path) if self.path.exists() else None
        self.losses = None
        if self.record is not None:
            if self.record["candidate"] != self.identity or self.record["batch"] != self.batch_identity:
                raise RuntimeError("Evaluation cache batch/order identity mismatch")
            self.context["hits"] += 1

    def get_loss_for_batch(self, **kwargs):
        if self.record is not None:
            return self.record["losses"]
        self.losses = self.model.get_loss_for_batch(**kwargs)
        return self.losses

    def run_batch(self, **kwargs):
        if self.record is not None:
            return self.record["predictions"]
        if self.losses is None:
            raise RuntimeError("Native evaluation loss must precede decoding")
        predictions = self.model.run_batch(**kwargs)
        _save(self.path, {
            "candidate": self.identity, "batch": self.batch_identity,
            "losses": self.losses, "predictions": predictions,
        })
        self.context["computed"] += 1
        return predictions
