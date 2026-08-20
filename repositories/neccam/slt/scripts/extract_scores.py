#!/usr/bin/env python3
"""Extract JSON metrics from upstream result pickles in the author environment."""

import argparse
import json
import pickle
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()

    test_result_files = sorted(args.model_dir.glob("best.IT_*.test_results.pkl"))
    dev_result_files = sorted(args.model_dir.glob("best.IT_*.dev_results.pkl"))
    if not test_result_files or not dev_result_files:
        raise FileNotFoundError(f"no test result pickle under {args.model_dir}")
    with test_result_files[-1].open("rb") as handle:
        test_result = pickle.load(handle)
    with dev_result_files[-1].open("rb") as handle:
        dev_result = pickle.load(handle)

    best_dev_recognition = min(
        dev_result["recognition_results"].values(),
        key=lambda item: item["valid_scores"]["wer"],
    )
    translation_results = [
        item
        for beam_results in dev_result["translation_results"].values()
        for item in beam_results.values()
    ]
    best_dev_translation = max(
        translation_results,
        key=lambda item: item["valid_scores"]["bleu"],
    )
    test_scores = test_result["valid_scores"]
    result = {
        "dev_wer": best_dev_recognition["valid_scores"]["wer"],
        "dev_bleu4": best_dev_translation["valid_scores"]["bleu"],
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
