"""Modal entry points for the Table 6 mBART/mT5 gloss-to-text finetuning."""

from __future__ import annotations

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parent.parent
TSLT_COMMIT = "d119fbb642d653a987a2e1b2cd1541c88df7f2ef"
RESULTS_VOLUME = "chen-2026-gloss-pretrained-results"

app = modal.App("chen-2026-gloss-pretrained")

base_image = modal.Image.from_dockerfile(REPOSITORY_ROOT / "Dockerfile", context_dir=REPOSITORY_ROOT)
image = (
    base_image
    .pip_install(
        "transformers==4.46.3",
        "datasets==3.1.0",
        "accelerate==1.1.1",
        "sentencepiece==0.2.0",
        "nltk==3.9.1",
    )
    .add_local_file(ROOT / "patches" / "0001-meteor-tokenize.patch", "/tmp/0001-meteor-tokenize.patch", copy=True)
    .run_commands(
        # Only patch 0001 (meteor.py's nltk API break) applies here; 0002
        # (beam_search.py) is specific to the OpenNMT-py baseline path this
        # image doesn't use.
        f"git clone https://github.com/kayoyin/transformer-slt.git /opt/transformer-slt"
        f" && git -C /opt/transformer-slt checkout {TSLT_COMMIT}"
        f" && git -C /opt/transformer-slt -c user.email=repro@example.invalid -c user.name=repro am /tmp/0001-meteor-tokenize.patch",
        "python -c \"import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet')\"",
        # The NGC 26.04 image's apex build dropped apex.amp; transformers'
        # trainer imports it unconditionally whenever apex is importable at
        # all. We use native bf16, not apex AMP, so removing it is a
        # correctness fix, not a capability loss (preflight on 2026-08-31
        # failed with "ImportError: cannot import name 'amp' from 'apex'"
        # otherwise).
        "pip uninstall -y apex",
        # tools/rouge.py resolves its wordnet DB files via
        # pkg_resources.resource_filename(__name__, ...), i.e. relative to
        # tools/ itself, but the repo ships them at the repo root — a
        # pre-existing repo layout quirk unrelated to our changes. Copying
        # them alongside rouge.py is the minimal fix (preflight on
        # 2026-08-31 failed with FileNotFoundError otherwise).
        "cp /opt/transformer-slt/wordnet_key_value.txt /opt/transformer-slt/wordnet_key_value_special_cases.txt /opt/transformer-slt/tools/",
    )
    .add_local_file(ROOT / "train_finetune.py", "/app/train_finetune.py")
    .add_local_file(ROOT / "select_backtranslation_sentences.py", "/app/select_backtranslation_sentences.py")
)

# Separate, older-stack image for the Transformer-tiny/+Back-translation
# baseline (ref [33]'s vendored OpenNMT-py 1.0.0 + torchtext==0.4.0); see
# this paper's own Dockerfile for why it can't share the mBART/mT5 image.
BASELINE_PYTHON = "/root/miniconda3/bin/python"
baseline_image = (
    modal.Image.from_dockerfile(ROOT / "Dockerfile", context_dir=ROOT, add_python="3.11")
    .add_local_file(ROOT / "prepare_baseline_data.py", "/app/prepare_baseline_data.py")
    .add_local_file(ROOT / "train_baseline.py", "/app/train_baseline.py")
)

datasets = modal.Volume.from_name("datasets", create_if_missing=False)
cache = modal.Volume.from_name("huggingface-cache", create_if_missing=False)
results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)

ENV = {"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"}


@app.function(image=baseline_image, gpu="A100", cpu=2, timeout=5 * 60)
def check_baseline_env() -> str:
    """Import + CUDA smoke test for the OpenNMT-py/torchtext environment."""
    import subprocess

    out = subprocess.run(
        [BASELINE_PYTHON, "-c", "import onmt, torch, torchtext; print(torch.__version__, torchtext.__version__, torch.cuda.is_available())"],
        capture_output=True, text=True,
    )
    result = f"exit={out.returncode}\nSTDOUT:\n{out.stdout}\nSTDERR:\n{out.stderr}"
    print(result)
    return result


@app.function(image=baseline_image, cpu=2, timeout=10 * 60, volumes={"/datasets": datasets, "/results": results})
def prepare_baseline_data(dataset: str) -> dict:
    """Write plain aligned src/tgt files for onmt_preprocess (idempotent)."""
    import subprocess

    output_dir = Path("/results") / "baseline-data" / dataset
    if (output_dir / "test.tgt").exists():
        return {"status": "already prepared", "dir": str(output_dir)}
    subprocess.run(
        [BASELINE_PYTHON, "/app/prepare_baseline_data.py", "--dataset", dataset, "--output-dir", str(output_dir)],
        check=True,
    )
    results.commit()
    return {"status": "prepared", "dir": str(output_dir)}


@app.function(
    image=baseline_image,
    gpu="A100",
    cpu=8,
    timeout=6 * 60 * 60,
    volumes={"/results": results},
)
def train_baseline(dataset: str, seed: int = 42) -> dict:
    """Train + translate the Transformer-tiny baseline for one dataset."""
    import shutil
    import subprocess

    data_dir = Path("/results") / "baseline-data" / dataset
    output_dir = Path("/results") / f"baseline-{dataset}-seed{seed}"
    pred_path = output_dir / "test.pred.txt"
    if pred_path.exists():
        return {"status": "already trained", "pred_path": str(pred_path)}
    if output_dir.exists():
        shutil.rmtree(output_dir)
    subprocess.run(
        [
            BASELINE_PYTHON, "/app/train_baseline.py",
            "--data-dir", str(data_dir), "--output-dir", str(output_dir),
            "--tslt-dir", "/opt/transformer-slt", "--seed", str(seed),
        ],
        check=True,
    )
    results.commit()
    return {"status": "trained", "pred_path": str(pred_path)}


@app.function(image=baseline_image, gpu="A100", cpu=8, timeout=2 * 60 * 60, volumes={"/results": results})
def train_reverse_and_translate(seed: int = 42) -> dict:
    """Train Phoenix text->gloss (reverse) model, translate the selected
    30K in-domain German sentences to synthetic gloss (Section IV-A2)."""
    import shutil
    import subprocess

    data_dir = Path("/results") / "baseline-data" / "phoenix"
    output_dir = Path("/results") / f"baseline-phoenix-reverse-seed{seed}"
    pred_path = output_dir / "test.pred.txt"  # holds translations of the 30K selected sentences
    if pred_path.exists():
        return {"status": "already translated", "pred_path": str(pred_path)}
    if output_dir.exists():
        shutil.rmtree(output_dir)
    selected = Path("/results") / "backtranslation" / "selected_30k_de.txt"
    subprocess.run(
        [
            BASELINE_PYTHON, "/app/train_baseline.py",
            "--data-dir", str(data_dir), "--output-dir", str(output_dir),
            "--tslt-dir", "/opt/transformer-slt", "--seed", str(seed),
            "--reverse", "--translate-only-src", str(selected),
        ],
        check=True,
    )
    results.commit()
    return {"status": "translated", "pred_path": str(pred_path)}


@app.function(image=baseline_image, cpu=2, timeout=10 * 60, volumes={"/results": results})
def augment_phoenix_data(seed: int = 42) -> dict:
    """Combine original Phoenix train pairs with synthetic
    (back-translated gloss, selected German text) pairs."""
    data_dir = Path("/results") / "baseline-data" / "phoenix"
    synthetic_gloss_path = Path("/results") / f"baseline-phoenix-reverse-seed{seed}" / "test.pred.txt"
    selected_text_path = Path("/results") / "backtranslation" / "selected_30k_de.txt"
    output_dir = Path("/results") / "baseline-data" / "phoenix-augmented"
    output_dir.mkdir(parents=True, exist_ok=True)

    orig_src = (data_dir / "train.src").read_text(encoding="utf-8").splitlines()
    orig_tgt = (data_dir / "train.tgt").read_text(encoding="utf-8").splitlines()
    synth_src = synthetic_gloss_path.read_text(encoding="utf-8").splitlines()
    synth_tgt = selected_text_path.read_text(encoding="utf-8").splitlines()
    assert len(synth_src) == len(synth_tgt), f"{len(synth_src)} vs {len(synth_tgt)}"

    (output_dir / "train.src").write_text("\n".join(orig_src + synth_src) + "\n", encoding="utf-8")
    (output_dir / "train.tgt").write_text("\n".join(orig_tgt + synth_tgt) + "\n", encoding="utf-8")
    for split in ("dev", "test"):
        (output_dir / f"{split}.src").write_text((data_dir / f"{split}.src").read_text(encoding="utf-8"), encoding="utf-8")
        (output_dir / f"{split}.tgt").write_text((data_dir / f"{split}.tgt").read_text(encoding="utf-8"), encoding="utf-8")

    results.commit()
    return {"status": "augmented", "original_pairs": len(orig_src), "synthetic_pairs": len(synth_src), "total": len(orig_src) + len(synth_src)}


@app.function(image=baseline_image, cpu=2, timeout=10 * 60, volumes={"/results": results})
def score_baseline(dataset: str, seed: int = 42) -> dict:
    """Score the Transformer-tiny baseline's predictions with ref [33]'s own tools."""
    import subprocess

    output_dir = Path("/results") / f"baseline-{dataset}-seed{seed}"
    data_dir = Path("/results") / "baseline-data" / dataset
    pred_path = output_dir / "test.pred.txt"
    ref_path = data_dir / "test.tgt"

    def run_tool(args):
        out = subprocess.run(args, cwd="/opt/transformer-slt", capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(f"{args} failed:\nSTDOUT:\n{out.stdout}\nSTDERR:\n{out.stderr}")
        return out.stdout

    scores = {}
    for n in (1, 2, 3, 4):
        stdout = run_tool([BASELINE_PYTHON, "/opt/transformer-slt/tools/bleu.py", str(n), str(pred_path), str(ref_path)])
        scores[f"bleu-{n}"] = float(stdout.strip()) * 100
    stdout = run_tool([BASELINE_PYTHON, "/opt/transformer-slt/tools/rouge.py", str(pred_path), str(ref_path)])
    rouge_l_line = next(line for line in stdout.splitlines() if line.strip().startswith("rouge-l:"))
    scores["rouge-l"] = float(rouge_l_line.split("F1:")[1].strip())
    stdout = run_tool([BASELINE_PYTHON, "/opt/transformer-slt/tools/meteor.py", str(pred_path), str(ref_path)])
    scores["meteor"] = float(stdout.strip()) * 100

    (output_dir / "scores.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    results.commit()
    print(json.dumps(scores, indent=2))
    return scores


@app.function(
    image=image,
    gpu=["A100", "L40S"],
    cpu=8,
    timeout=24 * 60 * 60,
    volumes={"/datasets": datasets, "/cache/huggingface": cache, "/results": results},
    env=ENV,
)
def finetune(model: str, dataset: str, seed: int = 42, max_steps: int = 40000, batch_size: int = 16) -> dict:
    """Table 6 mBART/mT5 finetuning + evaluation for one (model, dataset) cell."""
    import shutil
    import subprocess

    output_dir = Path("/results") / f"{model}-{dataset}-seed{seed}"
    scores_path = output_dir / "scores.json"
    if scores_path.exists():
        return json.loads(scores_path.read_text(encoding="utf-8"))
    if output_dir.exists():
        shutil.rmtree(output_dir)  # incomplete from a prior failed attempt
    subprocess.run(
        [
            "python", "/app/train_finetune.py",
            "--model", model, "--dataset", dataset,
            "--output-dir", str(output_dir),
            "--tslt-dir", "/opt/transformer-slt",
            "--seed", str(seed), "--max-steps", str(max_steps), "--batch-size", str(batch_size),
        ],
        check=True,
    )
    results.commit()
    return json.loads(scores_path.read_text(encoding="utf-8"))


@app.function(
    image=image,
    gpu="A100",
    cpu=8,
    timeout=2 * 60 * 60,
    volumes={"/datasets": datasets, "/cache/huggingface": cache, "/results": results},
    env=ENV,
)
def reevaluate(model: str, dataset: str, checkpoint: str, seed: int = 42) -> dict:
    """Re-run final test-set generation+scoring from an already-trained
    checkpoint with the fixed generation config (no_repeat_ngram_size=3),
    without retraining. Overwrites the canonical scores.json in place."""
    import subprocess

    output_dir = Path("/results") / f"{model}-{dataset}-seed{seed}"
    checkpoint_dir = output_dir / "checkpoints" / checkpoint
    subprocess.run(
        [
            "python", "/app/train_finetune.py",
            "--model", model, "--dataset", dataset,
            "--output-dir", str(output_dir),
            "--tslt-dir", "/opt/transformer-slt",
            "--seed", str(seed), "--from-checkpoint", str(checkpoint_dir),
        ],
        check=True,
    )
    results.commit()
    return json.loads((output_dir / "scores.json").read_text(encoding="utf-8"))


@app.function(image=image, gpu="A10G", cpu=2, timeout=10 * 60, volumes={"/cache/huggingface": cache}, env=ENV)
def diagnose_model(model: str) -> dict:
    """Check whether the pretrained checkpoint's embeddings actually load,
    and whether the raw (un-finetuned) model produces coherent generations.
    Full-run eval_loss came out ~ln(vocab_size), the signature of a
    from-scratch-init model, not a finetuned pretrained one."""
    import sys

    import torch

    sys.path.insert(0, "/app")
    from train_finetune import MODELS, MODEL_REVISIONS
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_name, revision = MODELS[model], MODEL_REVISIONS[model]
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    net = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision=revision)
    shared = net.get_input_embeddings().weight
    net2 = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision=revision)
    shared2 = net2.get_input_embeddings().weight
    identical_across_loads = torch.equal(shared, shared2)

    net.to("cuda").eval()
    prompt = "Translate to German: The weather tomorrow is sunny." if "mbart" not in model else "The weather tomorrow is sunny."
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = net.generate(**inputs, max_length=30, num_beams=4)
    generation = tokenizer.decode(out[0], skip_special_tokens=True)

    result = {
        "model": model,
        "embedding_mean_abs": shared.abs().mean().item(),
        "embedding_std": shared.std().item(),
        "identical_embeddings_across_two_independent_loads": identical_across_loads,
        "raw_pretrained_generation": generation,
    }
    print(json.dumps(result, indent=2))
    return result


@app.function(image=image, gpu="A10G", cpu=4, timeout=15 * 60, volumes={"/datasets": datasets, "/cache/huggingface": cache, "/results": results}, env=ENV)
def inspect_checkpoint(model: str, dataset: str, checkpoint: str, n: int = 5) -> None:
    """Generate from a saved in-progress checkpoint on real dev examples,
    printing both raw and skip_special_tokens=True decoded output (dev
    BLEU-4 proxy is stuck at exactly 0.0000 for every mT5 run so far)."""
    import sys

    import torch

    sys.path.insert(0, "/app")
    from train_finetune import MODEL_REVISIONS, MODELS, build_dataset
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    ckpt_dir = f"/results/{model}-{dataset}-seed42/checkpoints/{checkpoint}"
    tokenizer = AutoTokenizer.from_pretrained(MODELS[model], revision=MODEL_REVISIONS[model])
    net = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir).to("cuda").eval()
    net.generation_config.no_repeat_ngram_size = 3
    net.generation_config.num_beams = 4

    dev = build_dataset(dataset, "dev")
    for i in range(n):
        gloss, text = dev["gloss"][i], dev["text"][i]
        inputs = tokenizer(gloss, return_tensors="pt", truncation=True, max_length=128).to("cuda")
        with torch.no_grad():
            out = net.generate(**inputs, max_length=128, num_beams=4, no_repeat_ngram_size=3)
        raw = tokenizer.decode(out[0], skip_special_tokens=False)
        stripped = tokenizer.decode(out[0], skip_special_tokens=True)
        print(json.dumps({"gloss": gloss, "reference": text, "raw_decode": raw, "stripped_decode": stripped}, ensure_ascii=False, indent=2))


@app.function(image=image, gpu="A10G", cpu=4, timeout=15 * 60, volumes={"/datasets": datasets, "/cache/huggingface": cache}, env=ENV)
def trace_loss(model: str, dataset: str, batch_size: int, steps: int = 100) -> None:
    """Per-step loss log via the real Trainer code path (diagnostic only)."""
    import subprocess

    subprocess.run(
        [
            "python", "/app/train_finetune.py",
            "--model", model, "--dataset", dataset,
            "--output-dir", "/tmp/trace", "--tslt-dir", "/opt/transformer-slt",
            "--seed", "42", "--batch-size", str(batch_size), "--max-steps", str(steps),
            "--skip-final-eval",
        ],
        check=True,
    )


@app.function(image=image, cpu=2, timeout=10 * 60, volumes={"/datasets": datasets})
def check_data() -> dict:
    """CPU-only sanity check of all three dataset loaders (no model/GPU)."""
    import sys

    sys.path.insert(0, "/app")
    from train_finetune import LOADERS, DATASET_PATHS

    report = {}
    for dataset_key, loader in LOADERS.items():
        splits = {}
        for split in ("train", "dev", "test"):
            gloss, text = loader(Path(DATASET_PATHS[dataset_key]), split)
            splits[split] = {"count": len(gloss), "sample_gloss": gloss[0], "sample_text": text[0]}
        report[dataset_key] = splits
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


@app.function(image=image, cpu=4, timeout=30 * 60, volumes={"/cache/huggingface": cache, "/results": results}, env=ENV)
def fetch_german_background_corpus(n_sentences: int = 400000) -> dict:
    """Stream a German Common Crawl sample (mc4/de) as the background
    corpus for cross-entropy-difference in-domain selection (Section III-C).
    The paper's own intended source (tagesschau.de) is dead even to its
    authors (footnote 6); this is a documented substitute, not a faithful
    recovery of their exact set."""
    from datasets import load_dataset

    output_path = Path("/results") / "backtranslation" / "background_de.txt"
    if output_path.exists():
        return {"status": "already fetched", "path": str(output_path)}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("allenai/c4", "de", split="train", streaming=True)
    lines = []
    for row in ds:
        for sent in row["text"].split("\n"):
            sent = sent.strip()
            if 20 <= len(sent) <= 300:
                lines.append(sent)
        if len(lines) >= n_sentences:
            break
    output_path.write_text("\n".join(lines[:n_sentences]) + "\n", encoding="utf-8")
    results.commit()
    return {"status": "fetched", "path": str(output_path), "count": len(lines[:n_sentences])}


@app.function(image=image, cpu=4, timeout=30 * 60, volumes={"/results": results})
def select_backtranslation_sentences(n_select: int = 30000) -> dict:
    """Cross-entropy-difference selection of in-domain German sentences
    (approximating Section III-C's 30K set; see select_backtranslation_sentences.py)."""
    import subprocess

    in_domain = Path("/results") / "baseline-data" / "phoenix" / "train.tgt"
    background = Path("/results") / "backtranslation" / "background_de.txt"
    output = Path("/results") / "backtranslation" / "selected_30k_de.txt"
    if output.exists():
        return {"status": "already selected", "path": str(output)}
    out = subprocess.run(
        [
            "python", "/app/select_backtranslation_sentences.py",
            "--in-domain-file", str(in_domain), "--background-file", str(background),
            "--output-file", str(output), "--n-select", str(n_select),
        ],
        capture_output=True, text=True,
    )
    print(out.stdout, out.stderr)
    out.check_returncode()
    results.commit()
    return {"status": "selected", "path": str(output)}


@app.function(image=image, gpu="A100", cpu=8, timeout=20 * 60, volumes={"/datasets": datasets, "/cache/huggingface": cache}, env=ENV)
def measure_throughput(model: str, dataset: str, batch_size: int, steps: int = 30) -> dict:
    """Real-weight, real-data throughput/memory measurement (no eval, no save)."""
    import sys
    import time

    import torch

    sys.path.insert(0, "/app")
    from train_finetune import MODEL_REVISIONS, MODELS, MBART_LANG, build_dataset
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq

    model_name, revision = MODELS[model], MODEL_REVISIONS[model]
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if model == "mbart":
        tokenizer.src_lang = tokenizer.tgt_lang = MBART_LANG[dataset]
    dropout_kwargs = {"dropout": 0.3, "attention_dropout": 0.1} if model == "mbart" else {"dropout_rate": 0.3}
    net = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision=revision, **dropout_kwargs)
    net = net.to("cuda", dtype=torch.bfloat16)
    net.train()

    raw = build_dataset(dataset, "train")

    def preprocess(batch):
        model_inputs = tokenizer(batch["gloss"], max_length=128, truncation=True)
        labels = tokenizer(text_target=batch["text"], max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = raw.map(preprocess, batched=True, remove_columns=["gloss", "text"])
    collator = DataCollatorForSeq2Seq(tokenizer, model=net)
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-5, betas=(0.9, 0.98), eps=1e-6)

    torch.cuda.reset_peak_memory_stats()
    batch_items = [tokenized[i] for i in range(batch_size)]
    batch = collator(batch_items)
    batch = {k: v.to("cuda") for k, v in batch.items()}

    # Warm up (compilation/allocator warmup not representative of steady state).
    for _ in range(3):
        optimizer.zero_grad()
        loss = net(**batch).loss
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    start = time.monotonic()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = net(**batch).loss
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start

    result = {
        "model": model, "dataset": dataset, "batch_size": batch_size, "steps": steps,
        "seconds_per_step": elapsed / steps,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
        "train_examples": len(raw),
    }
    print(json.dumps(result, indent=2))
    return result


@app.function(image=image, gpu="A100", cpu=8, timeout=30 * 60, volumes={"/datasets": datasets, "/cache/huggingface": cache, "/results": results}, env=ENV)
def preflight(model: str, dataset: str) -> dict:
    """Cheap representative preflight: real data/weights, a few steps, tiny eval subset."""
    import subprocess

    output_dir = Path("/results") / f"preflight-{model}-{dataset}"
    subprocess.run(
        [
            "python", "/app/train_finetune.py",
            "--model", model, "--dataset", dataset,
            "--output-dir", str(output_dir),
            "--tslt-dir", "/opt/transformer-slt",
            "--seed", "42", "--max-steps", "30", "--eval-steps", "30", "--batch-size", "4",
            "--eval-limit", "20",
        ],
        check=True,
    )
    results.commit()
    return json.loads((output_dir / "scores.json").read_text(encoding="utf-8"))
