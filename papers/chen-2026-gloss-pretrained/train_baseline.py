"""Train/eval the Transformer-tiny baseline (ref [33]'s OpenNMT-py recipe).

Table 5 is captioned by the paper as hyperparameters "used in Section III"
only; Section IV-A2 (Table 6) just says the baseline "follows [33]" without
restating numbers. This invokes ref [33]'s own documented Sample Usage
recipe (README.md) directly and unmodified -- the least-invasive reading of
"follow [33]" -- rather than guessing/porting Section III's numbers.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

PYTHON = "/root/miniconda3/bin/python"
ONMT_PREPROCESS = "/root/miniconda3/bin/onmt_preprocess"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Dir with {train,dev,test}.{src,tgt} from prepare_baseline_data.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tslt-dir", default="/opt/transformer-slt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reverse", action="store_true", help="Train text->gloss instead of gloss->text (for back-translation).")
    parser.add_argument("--translate-only-src", default=None, help="If set, skip training/using data-dir's test.src; translate this file instead (for back-translation).")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tslt = Path(args.tslt_dir)

    src_suffix, tgt_suffix = ("tgt", "src") if args.reverse else ("src", "tgt")
    preprocessed = output_dir / "data"
    run([
        ONMT_PREPROCESS,
        "-train_src", str(data_dir / f"train.{src_suffix}"), "-train_tgt", str(data_dir / f"train.{tgt_suffix}"),
        "-valid_src", str(data_dir / f"dev.{src_suffix}"), "-valid_tgt", str(data_dir / f"dev.{tgt_suffix}"),
        "-save_data", str(preprocessed), "-lower",
    ])

    model_prefix = output_dir / "model"
    run([
        PYTHON, str(tslt / "train.py"),
        "-data", str(preprocessed), "-save_model", str(model_prefix), "-keep_checkpoint", "1",
        "-layers", "2", "-rnn_size", "512", "-word_vec_size", "512", "-transformer_ff", "2048", "-heads", "8",
        "-encoder_type", "transformer", "-decoder_type", "transformer", "-position_encoding",
        "-max_generator_batches", "2", "-dropout", "0.1",
        "-early_stopping", "3", "-early_stopping_criteria", "accuracy", "ppl",
        "-batch_size", "2048", "-accum_count", "3", "-batch_type", "tokens", "-normalization", "tokens",
        "-optim", "adam", "-adam_beta2", "0.998", "-decay_method", "noam", "-warmup_steps", "3000", "-learning_rate", "0.5",
        "-max_grad_norm", "0", "-param_init", "0", "-param_init_glorot",
        "-label_smoothing", "0.1", "-valid_steps", "100", "-save_checkpoint_steps", "100",
        "-world_size", "1", "-gpu_ranks", "0", "-seed", str(args.seed),
    ])

    checkpoints = sorted(output_dir.glob("model_step_*.pt"))
    assert checkpoints, f"no checkpoint saved in {output_dir}"
    best_checkpoint = checkpoints[-1]

    translate_src = Path(args.translate_only_src) if args.translate_only_src else data_dir / f"test.{src_suffix}"
    pred_path = output_dir / "test.pred.txt"
    run([
        PYTHON, str(tslt / "translate.py"),
        "-model", str(best_checkpoint), "-src", str(translate_src),
        "-output", str(pred_path), "-gpu", "0", "-replace_unk", "-beam_size", "4",
    ])
    print(f"wrote predictions to {pred_path}, checkpoint {best_checkpoint}")


if __name__ == "__main__":
    main()
