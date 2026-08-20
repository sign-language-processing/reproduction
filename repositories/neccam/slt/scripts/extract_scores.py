#!/usr/bin/env python3
"""Extract JSON metrics from upstream result pickles in the author environment."""

import argparse
import io
import json
import pickle
from pathlib import Path

import torch


def load_pickle(path: Path):
    """Load upstream result pickles without requiring their original CUDA device."""
    original_loader = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda data: torch.load(
        io.BytesIO(data), map_location="cpu"
    )
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    finally:
        torch.storage._load_from_bytes = original_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--result-stem", default="best.IT_*")
    args = parser.parse_args()

    test_result_files = sorted(
        args.model_dir.glob(f"{args.result_stem}.test_results.pkl")
    )
    dev_result_files = sorted(
        args.model_dir.glob(f"{args.result_stem}.dev_results.pkl")
    )
    if not test_result_files or not dev_result_files:
        raise FileNotFoundError(f"no test result pickle under {args.model_dir}")
    test_result = load_pickle(test_result_files[-1])
    dev_result = load_pickle(dev_result_files[-1])

    best_recognition_beam, best_dev_recognition = min(
        dev_result["recognition_results"].items(),
        key=lambda pair: pair[1]["valid_scores"]["wer"],
    )
    translation_results = [
        (beam, alpha, item)
        for beam, beam_results in dev_result["translation_results"].items()
        for alpha, item in beam_results.items()
    ]
    best_translation_beam, best_translation_alpha, best_dev_translation = max(
        translation_results,
        key=lambda result: result[2]["valid_scores"]["bleu"],
    )
    test_scores = test_result["valid_scores"]
    result = {
        "dev_wer": best_dev_recognition["valid_scores"]["wer"],
        "dev_bleu4": best_dev_translation["valid_scores"]["bleu"],
        "dev_recognition_beam_size": best_recognition_beam,
        "dev_translation_beam_size": best_translation_beam,
        "dev_translation_beam_alpha": best_translation_alpha,
        "test_wer": test_scores["wer"],
        "test_bleu4": test_scores["bleu"],
        "test_bleu_scores": test_scores["bleu_scores"],
        "test_wer_scores": test_scores["wer_scores"],
        "test_chrf": test_scores["chrf"],
        "test_rouge": test_scores["rouge"],
        "raw_metric_files": [
            str(dev_result_files[-1]),
            str(test_result_files[-1]),
        ],
    }
    print(json.dumps(result, default=lambda value: value.item()))


if __name__ == "__main__":
    main()
