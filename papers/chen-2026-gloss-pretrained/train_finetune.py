"""Finetune mBART25 / mT5-{small,base,large} on gloss-to-text pairs (Table 6).

Reimplements the recipe stated in Section IV-A of the target paper (no
published code exists for this part; preference level 3): label-smoothed
cross-entropy (smoothing 0.2), Adam (beta1=0.9, beta2=0.98, eps=1e-6),
polynomial LR schedule (max LR 3e-5, warmup 2500 updates), dropout 0.3,
attention dropout 0.1. The paper does not state batch size, epoch/step
budget, or a checkpoint-selection rule (see reproduction.json.guesses); this
script selects the best checkpoint by corpus BLEU-4 on the dev split.

Evaluation reuses ref [33]'s bundled nltk-based scoring tools
(tools/bleu.py, tools/rouge.py, tools/meteor.py from the pinned
kayoyin/transformer-slt commit) unmodified, via subprocess, since the target
paper says its Transformer baseline follows [33] and the paper itself names
no metric toolkit.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import subprocess
from pathlib import Path

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import Dataset

MODELS = {
    "mbart": "facebook/mbart-large-cc25",
    "mt5-small": "google/mt5-small",
    "mt5-base": "google/mt5-base",
    "mt5-large": "google/mt5-large",
}
MODEL_REVISIONS = {
    "mbart": "f417e5563320b2cc8aabe4329d986b238809067f",
    "mt5-small": "73fb5dbe4756edadc8fbe8c769b0a109493acf7a",
    "mt5-base": "2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f",
    "mt5-large": "50b7223e98fcd124b0cabb1ec81bc6324c7df107",
}
# mBART is multilingual and needs an explicit language code; gloss and text
# share the target spoken language (Section III-B "same language" framing).
MBART_LANG = {"phoenix": "de_DE", "aslgpc12": "en_XX", "csldaily": "zh_CN"}


def load_phoenix(root: Path, split: str) -> tuple[list[str], list[str]]:
    name = {"train": "train", "dev": "dev", "test": "test"}[split]
    path = root / f"PHOENIX-2014-T.{name}.corpus.csv"
    gloss, text = [], []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="|")
        for row in reader:
            # Section IV-A1: gloss annotations are lowercased before use.
            gloss.append(row["orth"].lower())
            text.append(row["translation"])
    return gloss, text


def load_aslgpc12(root: Path, split: str) -> tuple[list[str], list[str]]:
    gloss = (root / f"{split}.gloss").read_text(encoding="utf-8").splitlines()
    text = (root / f"{split}.en").read_text(encoding="utf-8").splitlines()
    return gloss, text


def load_csldaily(root: Path, split: str) -> tuple[list[str], list[str]]:
    with (root / "csl2020ct_v2.pkl").open("rb") as fh:
        data = pickle.load(fh)
    split_by_name = {}
    with (root / "split_1.txt").open(encoding="utf-8") as fh:
        next(fh)  # header: name|split
        for line in fh:
            name, split_name = line.strip().split("|")
            split_by_name[name] = split_name
    gloss, text = [], []
    for entry in data["info"]:
        if split_by_name.get(entry["name"]) != split:
            continue
        gloss.append(" ".join(entry["label_gloss"]))
        # Character-level target: join without spaces (paper Table 4b shows
        # unsegmented Chinese text), matching Zhou et al. 2021's convention
        # for this exact corpus.
        text.append("".join(entry["label_char"]))
    return gloss, text


LOADERS = {"phoenix": load_phoenix, "aslgpc12": load_aslgpc12, "csldaily": load_csldaily}
DATASET_PATHS = {
    "phoenix": "/datasets/rwth-phoenix-2014-t/annotations",
    "aslgpc12": "/datasets/aslg-pc12",
    "csldaily": "/datasets/csl-daily/sentence_label",
}


def build_dataset(dataset_key: str, split: str) -> Dataset:
    gloss, text = LOADERS[dataset_key](Path(DATASET_PATHS[dataset_key]), split)
    assert len(gloss) == len(text) and len(gloss) > 0, f"{dataset_key}/{split}: empty or misaligned"
    return Dataset.from_dict({"gloss": gloss, "text": text})


def space_join_chars(line: str, dataset_key: str) -> str:
    # CSL-Daily's Chinese target text has no word boundaries; score at
    # character granularity by whitespace-separating characters, matching
    # community convention (Zhou et al. 2021) for this exact benchmark.
    return " ".join(line) if dataset_key == "csldaily" else line


def _run_tool(args: list[str], tslt_dir: Path) -> str:
    out = subprocess.run(args, cwd=tslt_dir, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{args} failed:\nSTDOUT:\n{out.stdout}\nSTDERR:\n{out.stderr}")
    return out.stdout


def run_tslt_scoring(pred_path: Path, ref_path: Path, tslt_dir: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    for n in (1, 2, 3, 4):
        stdout = _run_tool(["python", str(tslt_dir / "tools" / "bleu.py"), str(n), str(pred_path), str(ref_path)], tslt_dir)
        scores[f"bleu-{n}"] = float(stdout.strip()) * 100
    stdout = _run_tool(["python", str(tslt_dir / "tools" / "rouge.py"), str(pred_path), str(ref_path)], tslt_dir)
    # rouge.py prints one line per metric, e.g. "\trouge-l:\tP: xx.xx\tR: xx.xx\tF1: xx.xx".
    rouge_l_line = next(line for line in stdout.splitlines() if line.strip().startswith("rouge-l:"))
    scores["rouge-l"] = float(rouge_l_line.split("F1:")[1].strip())
    stdout = _run_tool(["python", str(tslt_dir / "tools" / "meteor.py"), str(pred_path), str(ref_path)], tslt_dir)
    scores["meteor"] = float(stdout.strip()) * 100
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--dataset", choices=LOADERS, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tslt-dir", default="/opt/transformer-slt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=40000, help="Fixed step budget, uniform across datasets (paper states no epoch/step count; warmup_steps=2500 must stay a small fraction of this).")
    parser.add_argument("--eval-steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--eval-limit", type=int, default=None, help="Truncate the test split (preflight only).")
    parser.add_argument("--skip-final-eval", action="store_true", help="Stop after training; skip predict+scoring (loss-tracing diagnostic only).")
    parser.add_argument("--from-checkpoint", default=None, help="Load weights from this local checkpoint dir and skip training entirely (re-eval only, e.g. after a generation-config fix).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = MODELS[args.model]
    revision = MODEL_REVISIONS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if args.model == "mbart":
        lang = MBART_LANG[args.dataset]
        tokenizer.src_lang = lang
        tokenizer.tgt_lang = lang
    # BartConfig (mBART) exposes separate dropout/attention_dropout; T5Config
    # (mT5) has a single dropout_rate covering both (no attention_dropout arg).
    dropout_kwargs = (
        {"dropout": 0.3, "attention_dropout": 0.1} if args.model == "mbart"
        else {"dropout_rate": 0.3}
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.from_checkpoint or model_name,
        revision=None if args.from_checkpoint else revision,
        **dropout_kwargs,
    )
    # Beam search without an anti-repetition constraint degenerates into
    # looping a high-frequency n-gram (observed directly: mT5 checkpoints
    # produced fluent German stuck repeating "wettervorhersage fur morgen").
    # Neither the paper nor ref [33] states decoding settings for this
    # inference path; no_repeat_ngram_size=3 is a standard, documented
    # default (see reproduction.json.guesses), not a paper-specified value.
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.num_beams = 4

    raw = {split: build_dataset(args.dataset, split) for split in ("train", "dev", "test")}
    if args.eval_limit is not None:
        raw["test"] = raw["test"].select(range(min(args.eval_limit, len(raw["test"]))))

    def preprocess(batch):
        model_inputs = tokenizer(batch["gloss"], max_length=args.max_length, truncation=True)
        labels = tokenizer(text_target=batch["text"], max_length=args.max_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = {split: ds.map(preprocess, batched=True, remove_columns=["gloss", "text"]) for split, ds in raw.items()}
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        seed=args.seed,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        label_smoothing_factor=0.2,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        learning_rate=3e-5,
        lr_scheduler_type="polynomial",
        warmup_steps=2500,
        eval_strategy="no" if args.skip_final_eval else "steps",
        eval_steps=args.eval_steps,
        save_strategy="no" if args.skip_final_eval else "steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=args.max_length,
        load_best_model_at_end=not args.skip_final_eval,
        metric_for_best_model=None if args.skip_final_eval else "bleu4",
        greater_is_better=True,
        bf16=True,
        logging_steps=1 if args.skip_final_eval else 500,
        report_to=[],
    )

    def compute_metrics(eval_pred):
        # Cheap dev-time proxy (nltk corpus BLEU-4, lowercased/whitespace)
        # for load_best_model_at_end; the reported Table 6 numbers instead
        # come from run_tslt_scoring on the final test predictions below.
        from nltk.translate.bleu_score import corpus_bleu

        predictions, labels = eval_pred
        # The eval loop pads variable-length generated sequences with -100
        # (the loss-ignore sentinel it reuses for label padding too), not
        # pad_token_id; decoding -100 directly overflows the tokenizer's
        # unsigned id conversion.
        predictions[predictions == -100] = tokenizer.pad_token_id
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels[labels == -100] = tokenizer.pad_token_id
        references = tokenizer.batch_decode(labels, skip_special_tokens=True)
        hyps = [p.lower().split() for p in predictions]
        refs = [[r.lower().split()] for r in references]
        return {"bleu4": corpus_bleu(refs, hyps, weights=[0.25] * 4)}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["dev"],
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    if not args.from_checkpoint:
        trainer.train()
    if args.skip_final_eval:
        return

    predictions = trainer.predict(tokenized["test"], max_length=args.max_length)
    decoded = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
    decoded = [space_join_chars(line, args.dataset) for line in decoded]
    test_text = [space_join_chars(line, args.dataset) for line in raw["test"]["text"]]

    pred_path = output_dir / "test.pred.txt"
    ref_path = output_dir / "test.ref.txt"
    pred_path.write_text("\n".join(decoded) + "\n", encoding="utf-8")
    ref_path.write_text("\n".join(test_text) + "\n", encoding="utf-8")

    scores = run_tslt_scoring(pred_path, ref_path, Path(args.tslt_dir))
    (output_dir / "scores.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
