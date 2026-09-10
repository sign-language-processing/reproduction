"""Cross-entropy-difference in-domain sentence selection (Moore & Lewis 2010,
ref [31] in the paper), approximating Section III-C's 30K in-domain German
sentences. The paper's own intended source (tagesschau.de) is dead even to
its authors (footnote 6); this substitutes a background sample from mc4/de
(Common Crawl-derived, matching the paper's stated source family) and scores
it against Phoenix2014T's own training text as the in-domain seed. A
word-level unigram LM (add-1 smoothed) is used rather than a full n-gram LM,
a documented simplification of the original method.
"""
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path


def tokenize(line: str) -> list[str]:
    return line.lower().split()


def build_unigram_lm(lines: list[str]) -> tuple[Counter, int]:
    counts: Counter = Counter()
    for line in lines:
        counts.update(tokenize(line))
    total = sum(counts.values())
    return counts, total


def cross_entropy(tokens: list[str], counts: Counter, total: int, vocab_size: int) -> float:
    if not tokens:
        return 0.0
    log_prob = 0.0
    for tok in tokens:
        # add-1 (Laplace) smoothing
        p = (counts.get(tok, 0) + 1) / (total + vocab_size)
        log_prob += math.log(p)
    return -log_prob / len(tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-domain-file", required=True, help="In-domain seed text, one sentence per line")
    parser.add_argument("--background-file", required=True, help="Large background corpus, one sentence per line")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--n-select", type=int, default=30000)
    args = parser.parse_args()

    in_domain_lines = Path(args.in_domain_file).read_text(encoding="utf-8").splitlines()
    background_lines = Path(args.background_file).read_text(encoding="utf-8").splitlines()

    in_counts, in_total = build_unigram_lm(in_domain_lines)
    bg_counts, bg_total = build_unigram_lm(background_lines)
    vocab = set(in_counts) | set(bg_counts)
    vocab_size = len(vocab)

    scored = []
    for line in background_lines:
        tokens = tokenize(line)
        if not (3 <= len(tokens) <= 60):
            continue
        h_in = cross_entropy(tokens, in_counts, in_total, vocab_size)
        h_bg = cross_entropy(tokens, bg_counts, bg_total, vocab_size)
        scored.append((h_in - h_bg, line))

    scored.sort(key=lambda x: x[0])  # most in-domain-like (lowest diff) first
    selected = [line for _, line in scored[: args.n_select]]

    Path(args.output_file).write_text("\n".join(selected) + "\n", encoding="utf-8")
    print(f"selected {len(selected)} sentences from {len(background_lines)} candidates")
    print(f"score range: {scored[0][0]:.3f} .. {scored[args.n_select - 1][0]:.3f}")


if __name__ == "__main__":
    main()
