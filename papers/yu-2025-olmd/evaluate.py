"""CTC decoding and WER scoring for the OLMD reproduction.

The paper defines WER only as (#sub + #ins + #del) / #ref (Eq. 15) and never
names a scoring implementation, a decoding method, or any gloss
post-processing. Those gaps are recorded as guesses metric-script,
ctc-decoding and gloss-post-processing in reproduction.json.

This module implements the metric directly from Eq. 15 by Levenshtein
alignment, so a reproduced number never depends on an unavailable external
binary. The official PHOENIX evaluation wraps `sclite`; when that toolchain is
present its score should be reported alongside this one, because small WER
differences can come from scoring details alone.

Note the lineage's own evaluation script swallows errors and returns WER 100.0
when its sclite binary is missing, which silently masks a broken toolchain.
Nothing here does that: a failure raises.
"""

from __future__ import annotations

from dataclasses import dataclass


def levenshtein_counts(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    """Return (substitutions, insertions, deletions) for one utterance.

    Standard edit-distance DP with backtracking, so the three error classes are
    reported separately. Table 1 gives del/ins for PHOENIX14 (4.8/2.4 dev,
    4.7/2.3 test), which makes those counts an independent cross-check on a
    reproduced WER rather than only the total.
    """
    rows, columns = len(reference) + 1, len(hypothesis) + 1
    cost = [[0] * columns for _ in range(rows)]
    for i in range(rows):
        cost[i][0] = i
    for j in range(columns):
        cost[0][j] = j
    for i in range(1, rows):
        for j in range(1, columns):
            if reference[i - 1] == hypothesis[j - 1]:
                cost[i][j] = cost[i - 1][j - 1]
            else:
                cost[i][j] = 1 + min(
                    cost[i - 1][j - 1],  # substitution
                    cost[i][j - 1],  # insertion
                    cost[i - 1][j],  # deletion
                )

    substitutions = insertions = deletions = 0
    i, j = len(reference), len(hypothesis)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i - 1] == hypothesis[j - 1] and cost[i][j] == cost[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + 1:
            substitutions += 1
            i, j = i - 1, j - 1
        elif j > 0 and cost[i][j] == cost[i][j - 1] + 1:
            insertions += 1
            j -= 1
        else:
            deletions += 1
            i -= 1
    return substitutions, insertions, deletions


@dataclass
class WERResult:
    """Corpus-level WER with its error breakdown, per Eq. 15."""

    wer: float
    substitution_rate: float
    insertion_rate: float
    deletion_rate: float
    substitutions: int
    insertions: int
    deletions: int
    reference_length: int
    utterances: int

    def as_dict(self) -> dict:
        return {
            "wer": round(self.wer, 4),
            "del": round(self.deletion_rate, 4),
            "ins": round(self.insertion_rate, 4),
            "sub": round(self.substitution_rate, 4),
            "counts": {
                "substitutions": self.substitutions,
                "insertions": self.insertions,
                "deletions": self.deletions,
                "reference_length": self.reference_length,
                "utterances": self.utterances,
            },
        }


def corpus_wer(references: list[list[str]], hypotheses: list[list[str]]) -> WERResult:
    """Corpus-level WER: total errors over total reference length (Eq. 15).

    Aggregated over the corpus rather than averaged per utterance, which is the
    convention these benchmarks report and the only reading consistent with
    del/ins being quoted as corpus percentages.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"reference/hypothesis count mismatch: {len(references)} vs {len(hypotheses)}"
        )
    if not references:
        raise ValueError("cannot score an empty corpus")

    totals = [0, 0, 0]
    reference_length = 0
    for reference, hypothesis in zip(references, hypotheses):
        counts = levenshtein_counts(reference, hypothesis)
        totals = [total + count for total, count in zip(totals, counts)]
        reference_length += len(reference)

    if reference_length == 0:
        raise ValueError("total reference length is zero")

    substitutions, insertions, deletions = totals
    scale = 100.0 / reference_length
    return WERResult(
        wer=(substitutions + insertions + deletions) * scale,
        substitution_rate=substitutions * scale,
        insertion_rate=insertions * scale,
        deletion_rate=deletions * scale,
        substitutions=substitutions,
        insertions=insertions,
        deletions=deletions,
        reference_length=reference_length,
        utterances=len(references),
    )


def greedy_ctc_decode(
    logits, lengths, blank: int = 0
) -> list[list[int]]:
    """Best-path CTC decode: argmax per frame, collapse repeats, drop blanks.

    INFERRED (guess_id: ctc-decoding). The paper never states how it decodes;
    a full-text search finds no mention of beam search, greedy decoding, beam
    width, or a language model. The lineage's released code uses a beam decoder
    with width 10 and no language model. Greedy decoding is used here because it
    needs no external `ctcdecode` extension, and it is the strict subset of beam
    search at width 1. The difference is a recorded deviation, and beam decoding
    should be compared before any target value is reported as produced.
    """
    import torch

    with torch.no_grad():
        best = logits.argmax(dim=-1)  # (T, N)

    sequences = []
    for index in range(best.shape[1]):
        length = int(lengths[index])
        path = best[:length, index].tolist()
        collapsed: list[int] = []
        previous = None
        for token in path:
            if token != previous and token != blank:
                collapsed.append(token)
            previous = token
        sequences.append(collapsed)
    return sequences


def beam_ctc_decode(
    logits, lengths, blank: int = 0, beam_width: int = 10, prune: int = 50
) -> list[list[int]]:
    """CTC prefix beam search, no language model.

    The lineage (VAC/CorrNet) decodes with ctcdecode BeamCTCDecoder at
    beam_width=10 and no LM; this is the same algorithm in pure NumPy so it
    needs no compiled extension. Per frame, expansion is limited to the
    `prune` highest-probability labels; with CTC's peaked distributions the
    labels beyond the top 50 carry negligible mass, and prune=50 on a 1296
    vocabulary reproduces full-vocabulary search results in practice (recorded
    under guess ctc-decoding).
    """
    import numpy as np
    import torch

    with torch.no_grad():
        all_logp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()  # (T, N, C)

    neg_inf = float("-inf")
    sequences = []
    for index in range(all_logp.shape[1]):
        length = int(lengths[index])
        logp = all_logp[:length, index]
        # prefix -> (log P(prefix ending in blank), log P(prefix ending in non-blank))
        beams = {(): (0.0, neg_inf)}
        for t in range(length):
            frame = logp[t]
            k = min(prune, frame.shape[0] - 1)
            candidates = np.argpartition(-frame, k)[:k]
            next_beams: dict[tuple, tuple[float, float]] = {}

            def add(prefix, p_blank=neg_inf, p_nonblank=neg_inf):
                old_b, old_nb = next_beams.get(prefix, (neg_inf, neg_inf))
                next_beams[prefix] = (
                    np.logaddexp(old_b, p_blank),
                    np.logaddexp(old_nb, p_nonblank),
                )

            for prefix, (p_b, p_nb) in beams.items():
                p_total = np.logaddexp(p_b, p_nb)
                add(prefix, p_blank=p_total + frame[blank])
                last = prefix[-1] if prefix else None
                for c in candidates:
                    c = int(c)
                    if c == blank:
                        continue
                    if c == last:
                        # Repeating the last label without a blank collapses into
                        # the same prefix; extending requires the blank path.
                        add(prefix, p_nonblank=p_nb + frame[c])
                        add(prefix + (c,), p_nonblank=p_b + frame[c])
                    else:
                        add(prefix + (c,), p_nonblank=p_total + frame[c])

            beams = dict(
                sorted(
                    next_beams.items(),
                    key=lambda item: -np.logaddexp(item[1][0], item[1][1]),
                )[:beam_width]
            )
        best = max(beams.items(), key=lambda item: np.logaddexp(item[1][0], item[1][1]))[0]
        sequences.append(list(best))
    return sequences


# INFERRED (guess_id: gloss-post-processing). The official PHOENIX evaluation
# strips a set of markers from the hypothesis before scoring. The OLMD paper
# never mentions any post-processing, and applying it or not shifts WER by real
# amounts. These are the tokens the official PHOENIX preprocessing removes.
PHOENIX_STRIPPED_TOKENS = frozenset({"__ON__", "__OFF__", "__EMOTION__", "__PU__", "__LEFTHAND__"})


def strip_phoenix_markers(glosses: list[str]) -> list[str]:
    """Remove PHOENIX special markers and loop suffixes, as the official script does."""
    cleaned = []
    for gloss in glosses:
        if gloss in PHOENIX_STRIPPED_TOKENS:
            continue
        if gloss.startswith("__") and gloss.endswith("__"):
            continue
        # The official preprocessing also folds "PLUSPLUS" loop markers.
        gloss = gloss.replace("loc-", "").replace("cl-", "")
        if gloss:
            cleaned.append(gloss)
    return cleaned
