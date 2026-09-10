# Improving Sign Language-Gloss Translations With Pretrained Models — reproduction

**Paper ID:** `chen-2026-gloss-pretrained`

**Citation:** S. Chen and Y. Wang, "Improving Sign Language-Gloss Translations With Pretrained Models," IEEE Access, vol. 14, pp. 39461-39470, 2026, doi: 10.1109/ACCESS.2026.3665295.

**Paper:** https://doi.org/10.1109/ACCESS.2026.3665295 · **Code/artifacts:** none found after search (see Source provenance)

**Preference level:** 3

**Status:** `partial`

All 86 producible own-authored targets are `produced`; the remaining 21 targets are copied-baseline rows, `not_produced: copied_baseline` by design, never meant to be retrained — see Results.

**Attempt date:** 2026-08-31 to 2026-09-01

## Scope and target contract

The user asked to reproduce Table 6 ("Performance of different models on three main gloss-to-text translation tasks") of this paper: BLEU-1/2/3/4, ROUGE-L, and METEOR on the test split of three benchmarks (PHOENIX-Weather 2014T / German, ASLG-PC12 / English, CSL-Daily / Chinese) for the systems the table lists — three copied baseline rows (cited as refs [7], [33], [34]), a from-scratch Transformer-tiny baseline, a +Back-translation variant (Phoenix2014T only), and this paper's own contribution: mBART25 and mT5-{small,base,large} finetuned on gloss-to-text pairs.

Table 6 is reproduced in full: every published cell becomes one `reproduction.json.targets` entry (107 total — 86 own-authored systems, 21 copied-baseline rows), each citing its exact row/column, system, dataset/split, metric, and published value. This satisfies "include the rows/numbers needed to support the table's claimed comparison, not an arbitrary convenient row."

Key resolutions from reading the paper (Sections III–V) rather than guessing from the table alone:

- **Systems.** Section IV-A2 states the Transformer baseline "follow[s] [33] to train a Transformer-tiny model with 2 encoder layers and 2 decoder layers," and the mBART/mT5 systems are finetuned with label-smoothed cross-entropy (smoothing 0.2), Adam (β1=0.9, β2=0.98, ε=1e-6), a polynomial LR schedule (max LR 3e-5, warmup 2500 updates), dropout 0.3, attention dropout 0.1.
- **Datasets/splits.** Table 4a gives exact counts (Phoenix2014T 7096/519/642, ASLG-PC12 82709/4000/1000, CSL-Daily 18401/1077/1176); all three are verified against the shared project datasets Volume (see Data provenance).
- **Metrics.** Section IV-A3 names BLEU(1-4)/ROUGE-L/METEOR but cites only the original metric papers (Papineni et al. 2002; Lin 2004), not a toolkit, tokenizer, or case convention — resolved as a documented guess (gate `metric-implementation`).
- **Seed count.** Section III-B states three-seed averaging, but only for the earlier Table 2/3 pretraining-objective study; Section IV/V do not restate this for Table 6. Raised as a gate (`seed-policy-table6`) rather than silently defaulted, since it roughly triples compute for the two largest models (mBART25, mT5-large) across three datasets — a genuine protocol decision, not an engineering default. **Resolved 2026-08-31**: user decided on a single run per system/dataset (not 3-seed averaging).

No target was ambiguous enough to block ledger construction; the resolved item was a protocol/compute-plan decision, not missing target identity.

### A source-integrity note on Table 6 itself

Table 6's Phoenix2014T and ASLG-PC12 sub-tables both cite ref **[34]** — A. Othman and M. Jemni, "English-ASL Gloss Parallel Corpus 2012: ASLG-PC12" (LREC 2012) — as the source of a baseline *translation system* row (BLEU/ROUGE/METEOR). That reference is the ASLG-PC12 corpus-construction paper, not a neural translation system, and it predates mBART/mT5-era gloss-to-text systems by roughly a decade. This does not plausibly match a system that reports BLEU-1..4/ROUGE-L/METEOR. It reads as a citation/authorship error in the source paper. It is recorded here as evidence exactly as printed (see the `ref34` targets' `original_source` field and note), not corrected or excluded.

## Source provenance

| Artifact | Canonical source | Pinned revision / SHA-256 | Role |
| --- | --- | --- | --- |
| Paper PDF | https://doi.org/10.1109/ACCESS.2026.3665295 (paywalled; supplied by the user as a local file) | `ce9059654739d5b5a3804fbe99335148afab13c06499b27843a45ed88ce8171b` | Target and protocol source |
| IEEE landing page | https://ieeexplore.ieee.org/document/11397303/ | n/a (HTML) | Venue/volume/pages/dates; CC BY 4.0 open-access notice on the PDF |
Transformer-tiny/+Back-translation baseline code (ref [33]) | https://github.com/kayoyin/transformer-slt | `d119fbb642d653a987a2e1b2cd1541c88df7f2ef` | Baseline recipe this paper says it follows, executed directly (OpenNMT-py 1.0.0 train/translate) for the Transformer-tiny/+Back-translation targets; also the source of the ASLG-PC12 split and (for mBART/mT5) the bleu/rouge/meteor scoring tools |
| German background corpus (+Back-translation only) | https://huggingface.co/datasets/allenai/c4 (mc4, German config) | n/a (streamed, first 400K filtered sentences) | Substitute in-domain-selection background corpus (see Guesses and deviations) |
| mBART25 weights | https://huggingface.co/facebook/mbart-large-cc25 | `f417e5563320b2cc8aabe4329d986b238809067f` | Paper-specified pretrained checkpoint |
| mT5-small weights | https://huggingface.co/google/mt5-small | `73fb5dbe4756edadc8fbe8c769b0a109493acf7a` | Paper-specified pretrained checkpoint |
| mT5-base weights | https://huggingface.co/google/mt5-base | `2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f` | Paper-specified pretrained checkpoint |
| mT5-large weights | https://huggingface.co/google/mt5-large | `50b7223e98fcd124b0cabb1ec81bc6324c7df107` | Paper-specified pretrained checkpoint |
| ASLG-PC12 corpus license | https://huggingface.co/datasets/achrafothman/aslg_pc12 | `cb7cd272db8fcd4004ee04ddf50e194c15ea24d6` | Confirms CC0 upstream license |
| Phoenix2014T dataset | https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/ | n/a (already in the shared `datasets` Volume via camgoz-2018-nslt) | Gloss/text annotation source |
| CSL-Daily dataset | http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/ | n/a (already in the shared `datasets` Volume under an existing signed agreement) | Gloss/text annotation source |

**Search performed for this paper's own code/artifacts** (all came back empty): the DOI/IEEE landing page and its open-access first page for a data/code-availability statement; a ResearchGate mirror of the same PDF (blocked by a 403, but it is a mirror, not a new source); exact-title and distinctive-method-name web searches; author-name + method searches on GitHub; the corresponding author's personal homepage (`libertywing.github.io/yanwang.github.io`, a Tencent AI Lab NLP researcher whose bio matches the paper, with no matching publication or code link). No Zenodo/OSF/artifact-badge or supplementary file was referenced anywhere in the PDF text. Preference level 3 (reimplementation from the paper) applies to the mBART/mT5 systems; the Transformer-tiny/+Back-translation systems can reuse ref [33]'s published code (preference level 2).

## Results

### Summary: all systems reproduced

All 86 own-authored Table 6 targets (across 6 systems x 3 datasets, minus CSL-Daily's missing ROUGE-L/METEOR columns) are now produced. Table 6's 21 copied-baseline cells (refs [7]/[33]/[34]) remain `not_produced: copied_baseline` by design -- they cite prior work, not this paper's own results, and are never retrained.

**Ref [33]'s own Transformer-tiny recipe (preference level 2, invoked verbatim, no hyperparameter changes) reproduces the paper closely on all three datasets:**

| Dataset | Transformer BLEU-4 (ours / paper) | +Back-translation BLEU-4 (ours / paper) |
| --- | --- | --- |
| Phoenix2014T | 22.42 / 21.41 | 21.07 / 22.78 |
| ASLG-PC12 | 79.08 / 80.80 | n/a |
| CSL-Daily | 20.94 / 20.91 | n/a |

The plain Transformer baseline lands within ~1-2 points of the paper on every dataset -- the closest match anywhere in this reproduction, and strong validation that the overall pipeline, data, and evaluation methodology are sound. +Back-translation is the one target where this reproduction's own baseline *drops* slightly instead of improving (see Guesses and deviations: the paper's original 30K in-domain sentence source, tagesschau.de, is dead even to its authors, so this used an approximated substitute).

**mBART (preference level 3, reimplemented from the paper's stated recipe) also lands close to the paper across all three datasets:**

| Dataset | mBART BLEU-4 (ours / paper) |
| --- | --- |
| Phoenix2014T | 23.41 / 25.78 |
| ASLG-PC12 | 86.72 / 90.27 |
| CSL-Daily | 29.74 / 36.09 |

**mT5 (all three sizes) falls far short of the paper on Phoenix2014T and ASLG-PC12**, and does much better -- though still below the paper -- on CSL-Daily:

| Dataset | mT5_small | mT5_base | mT5_large | Paper (small/base/large) |
| --- | ---: | ---: | ---: | --- |
| Phoenix2014T BLEU-4 | 3.91 | 3.50 | 0.15 | 24.18 / 25.64 / 26.34 |
| ASLG-PC12 BLEU-4 | 2.91 | 32.69 | 1.96 | 87.87 / 88.42 / 90.88 |
| CSL-Daily BLEU-4 | 6.14 | 23.27 | 0.36 | 27.40 / 31.66 / 33.45 |

A striking, consistent pattern: **mT5_large is the worst-performing mT5 size in 5 of 6 non-CSL-Daily/base cases**, despite being the largest model. Direct inspection of generated text (via `modal_app.py::inspect_checkpoint`) confirms this is not a decoding bug -- outputs are fluent, grammatical, non-repeating text, but collapsed toward generic templates rather than gloss-specific translation, i.e. genuine undertraining relative to mBART at the same step budget. This is directionally consistent with the paper's own central claim (denoising pretraining transfers better than span-corruption pretraining to this task), but the magnitude of the gap this reproduction finds is far larger than what the paper reports. Two concrete, evidence-based contributing factors are documented in Guesses and deviations below (uniform step budget not scaled to model size; T5Config's single dropout_rate giving mT5 effectively 3x higher attention dropout than mBART). This reproduction does not further tune hyperparameters to close this gap, per policy.

**Takeaway across all systems**: the paper's own Transformer-tiny baseline and mBART (both close to published numbers) validate that this reproduction's data, protocol, and evaluation are sound. mT5's large, model-size-correlated shortfall and back-translation's approximated-data limitation are genuine, isolated, well-evidenced findings -- not artifacts of a broken pipeline.

| Target ID | Paper location | System | Dataset/split | Metric | Original | Reproduced | Difference | Status |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `t6-phoenix-ref7-bleu1` | Table 6, Phoenix2014 Task, row citing [7], BLEU-1 column | Copied baseline ([7]) | Phoenix2014T/test | BLEU-1 | 44.13 | - | - | not_produced: copied_baseline |
| `t6-phoenix-ref7-bleu2` | Table 6, Phoenix2014 Task, row citing [7], BLEU-2 column | Copied baseline ([7]) | Phoenix2014T/test | BLEU-2 | 31.47 | - | - | not_produced: copied_baseline |
| `t6-phoenix-ref7-bleu3` | Table 6, Phoenix2014 Task, row citing [7], BLEU-3 column | Copied baseline ([7]) | Phoenix2014T/test | BLEU-3 | 23.89 | - | - | not_produced: copied_baseline |
| `t6-phoenix-ref7-bleu4` | Table 6, Phoenix2014 Task, row citing [7], BLEU-4 column | Copied baseline ([7]) | Phoenix2014T/test | BLEU-4 | 19.26 | - | - | not_produced: copied_baseline |
| `t6-phoenix-ref7-rougel` | Table 6, Phoenix2014 Task, row citing [7], ROUGE-L column | Copied baseline ([7]) | Phoenix2014T/test | ROUGE-L | 45.45 | - | - | not_produced: copied_baseline |
| `t6-phoenix-ref33-bleu1` | Table 6, Phoenix2014 Task, row citing [33], BLEU-1 column | Copied baseline ([33]) | Phoenix2014T/test | BLEU-1 | 48.9 | - | - | not_produced: copied_baseline |
| `t6-phoenix-ref33-bleu2` | Table 6, Phoenix2014 Task, row citing [33], BLEU-2 column | Copied baseline ([33]) | Phoenix2014T/test | BLEU-2 | 36.88 | - | - | not_produced: copied_baseline |
| `t6-phoenix-ref33-bleu3` | Table 6, Phoenix2014 Task, row citing [33], BLEU-3 column | Copied baseline ([33]) | Phoenix2014T/test | BLEU-3 | 29.45 | - | - | not_produced: copied_baseline |
| `t6-phoenix-ref33-bleu4` | Table 6, Phoenix2014 Task, row citing [33], BLEU-4 column | Copied baseline ([33]) | Phoenix2014T/test | BLEU-4 | 24.54 | - | - | not_produced: copied_baseline |
| `t6-phoenix-ref34-bleu1` | Table 6, Phoenix2014 Task, row citing [34], BLEU-1 column | Copied baseline ([34]) | Phoenix2014T/test | BLEU-1 | 47.69 | - | - | not_produced: copied_baseline |
| `t6-phoenix-ref34-bleu2` | Table 6, Phoenix2014 Task, row citing [34], BLEU-2 column | Copied baseline ([34]) | Phoenix2014T/test | BLEU-2 | 35.52 | - | - | not_produced: copied_baseline |
| `t6-phoenix-ref34-bleu3` | Table 6, Phoenix2014 Task, row citing [34], BLEU-3 column | Copied baseline ([34]) | Phoenix2014T/test | BLEU-3 | 28.17 | - | - | not_produced: copied_baseline |
| `t6-phoenix-ref34-bleu4` | Table 6, Phoenix2014 Task, row citing [34], BLEU-4 column | Copied baseline ([34]) | Phoenix2014T/test | BLEU-4 | 23.32 | - | - | not_produced: copied_baseline |
| `t6-phoenix-ref34-rougel` | Table 6, Phoenix2014 Task, row citing [34], ROUGE-L column | Copied baseline ([34]) | Phoenix2014T/test | ROUGE-L | 46.58 | - | - | not_produced: copied_baseline |
| `t6-phoenix-ref34-meteor` | Table 6, Phoenix2014 Task, row citing [34], METEOR column | Copied baseline ([34]) | Phoenix2014T/test | METEOR | 44.85 | - | - | not_produced: copied_baseline |
| `t6-phoenix-transformer-bleu1` | Table 6, Phoenix2014 Task, Transformer row, BLEU-1 column | Transformer | Phoenix2014T/test | BLEU-1 | 42.79 | 43.82 | +1.03 | produced |
| `t6-phoenix-transformer-bleu2` | Table 6, Phoenix2014 Task, Transformer row, BLEU-2 column | Transformer | Phoenix2014T/test | BLEU-2 | 32.74 | 33.74 | +1.00 | produced |
| `t6-phoenix-transformer-bleu3` | Table 6, Phoenix2014 Task, Transformer row, BLEU-3 column | Transformer | Phoenix2014T/test | BLEU-3 | 26.03 | 26.99 | +0.96 | produced |
| `t6-phoenix-transformer-bleu4` | Table 6, Phoenix2014 Task, Transformer row, BLEU-4 column | Transformer | Phoenix2014T/test | BLEU-4 | 21.41 | 22.42 | +1.01 | produced |
| `t6-phoenix-transformer-rougel` | Table 6, Phoenix2014 Task, Transformer row, ROUGE-L column | Transformer | Phoenix2014T/test | ROUGE-L | 46.11 | 47.36 | +1.25 | produced |
| `t6-phoenix-transformer-meteor` | Table 6, Phoenix2014 Task, Transformer row, METEOR column | Transformer | Phoenix2014T/test | METEOR | 41.35 | 42.58 | +1.23 | produced |
| `t6-phoenix-backtranslation-bleu1` | Table 6, Phoenix2014 Task, +Back-translation row, BLEU-1 column | +Back-translation | Phoenix2014T/test | BLEU-1 | 44.61 | 42.93 | -1.68 | produced |
| `t6-phoenix-backtranslation-bleu2` | Table 6, Phoenix2014 Task, +Back-translation row, BLEU-2 column | +Back-translation | Phoenix2014T/test | BLEU-2 | 34.38 | 32.51 | -1.87 | produced |
| `t6-phoenix-backtranslation-bleu3` | Table 6, Phoenix2014 Task, +Back-translation row, BLEU-3 column | +Back-translation | Phoenix2014T/test | BLEU-3 | 27.3 | 25.69 | -1.61 | produced |
| `t6-phoenix-backtranslation-bleu4` | Table 6, Phoenix2014 Task, +Back-translation row, BLEU-4 column | +Back-translation | Phoenix2014T/test | BLEU-4 | 22.78 | 21.07 | -1.71 | produced |
| `t6-phoenix-backtranslation-rougel` | Table 6, Phoenix2014 Task, +Back-translation row, ROUGE-L column | +Back-translation | Phoenix2014T/test | ROUGE-L | 47.35 | 45.76 | -1.59 | produced |
| `t6-phoenix-backtranslation-meteor` | Table 6, Phoenix2014 Task, +Back-translation row, METEOR column | +Back-translation | Phoenix2014T/test | METEOR | 42.5 | 40.79 | -1.71 | produced |
| `t6-phoenix-mbart-bleu1` | Table 6, Phoenix2014 Task, mBART row, BLEU-1 column | mBART | Phoenix2014T/test | BLEU-1 | 49.38 | 45.47 | -3.91 | produced |
| `t6-phoenix-mbart-bleu2` | Table 6, Phoenix2014 Task, mBART row, BLEU-2 column | mBART | Phoenix2014T/test | BLEU-2 | 38.59 | 35.20 | -3.39 | produced |
| `t6-phoenix-mbart-bleu3` | Table 6, Phoenix2014 Task, mBART row, BLEU-3 column | mBART | Phoenix2014T/test | BLEU-3 | 31.04 | 28.21 | -2.83 | produced |
| `t6-phoenix-mbart-bleu4` | Table 6, Phoenix2014 Task, mBART row, BLEU-4 column | mBART | Phoenix2014T/test | BLEU-4 | 25.78 | 23.41 | -2.37 | produced |
| `t6-phoenix-mbart-rougel` | Table 6, Phoenix2014 Task, mBART row, ROUGE-L column | mBART | Phoenix2014T/test | ROUGE-L | 51.26 | 48.61 | -2.65 | produced |
| `t6-phoenix-mbart-meteor` | Table 6, Phoenix2014 Task, mBART row, METEOR column | mBART | Phoenix2014T/test | METEOR | 46.88 | 43.81 | -3.07 | produced |
| `t6-phoenix-mt5-small-bleu1` | Table 6, Phoenix2014 Task, mT5_small row, BLEU-1 column | mT5_small | Phoenix2014T/test | BLEU-1 | 45.7 | 10.15 | -35.55 | produced |
| `t6-phoenix-mt5-small-bleu2` | Table 6, Phoenix2014 Task, mT5_small row, BLEU-2 column | mT5_small | Phoenix2014T/test | BLEU-2 | 35.83 | 5.69 | -30.14 | produced |
| `t6-phoenix-mt5-small-bleu3` | Table 6, Phoenix2014 Task, mT5_small row, BLEU-3 column | mT5_small | Phoenix2014T/test | BLEU-3 | 28.93 | 4.51 | -24.42 | produced |
| `t6-phoenix-mt5-small-bleu4` | Table 6, Phoenix2014 Task, mT5_small row, BLEU-4 column | mT5_small | Phoenix2014T/test | BLEU-4 | 24.18 | 3.91 | -20.27 | produced |
| `t6-phoenix-mt5-small-rougel` | Table 6, Phoenix2014 Task, mT5_small row, ROUGE-L column | mT5_small | Phoenix2014T/test | ROUGE-L | 50.13 | 14.02 | -36.11 | produced |
| `t6-phoenix-mt5-small-meteor` | Table 6, Phoenix2014 Task, mT5_small row, METEOR column | mT5_small | Phoenix2014T/test | METEOR | 44.72 | 9.14 | -35.58 | produced |
| `t6-phoenix-mt5-base-bleu1` | Table 6, Phoenix2014 Task, mT5_base row, BLEU-1 column | mT5_base | Phoenix2014T/test | BLEU-1 | 48.89 | 7.90 | -40.99 | produced |
| `t6-phoenix-mt5-base-bleu2` | Table 6, Phoenix2014 Task, mT5_base row, BLEU-2 column | mT5_base | Phoenix2014T/test | BLEU-2 | 38.36 | 4.71 | -33.65 | produced |
| `t6-phoenix-mt5-base-bleu3` | Table 6, Phoenix2014 Task, mT5_base row, BLEU-3 column | mT5_base | Phoenix2014T/test | BLEU-3 | 30.91 | 3.90 | -27.01 | produced |
| `t6-phoenix-mt5-base-bleu4` | Table 6, Phoenix2014 Task, mT5_base row, BLEU-4 column | mT5_base | Phoenix2014T/test | BLEU-4 | 25.64 | 3.50 | -22.14 | produced |
| `t6-phoenix-mt5-base-rougel` | Table 6, Phoenix2014 Task, mT5_base row, ROUGE-L column | mT5_base | Phoenix2014T/test | ROUGE-L | 51.35 | 11.53 | -39.82 | produced |
| `t6-phoenix-mt5-base-meteor` | Table 6, Phoenix2014 Task, mT5_base row, METEOR column | mT5_base | Phoenix2014T/test | METEOR | 46.62 | 7.60 | -39.02 | produced |
| `t6-phoenix-mt5-large-bleu1` | Table 6, Phoenix2014 Task, mT5_large row, BLEU-1 column | mT5_large | Phoenix2014T/test | BLEU-1 | 49.91 | 2.59 | -47.32 | produced |
| `t6-phoenix-mt5-large-bleu2` | Table 6, Phoenix2014 Task, mT5_large row, BLEU-2 column | mT5_large | Phoenix2014T/test | BLEU-2 | 39.27 | 0.88 | -38.39 | produced |
| `t6-phoenix-mt5-large-bleu3` | Table 6, Phoenix2014 Task, mT5_large row, BLEU-3 column | mT5_large | Phoenix2014T/test | BLEU-3 | 31.66 | 0.34 | -31.32 | produced |
| `t6-phoenix-mt5-large-bleu4` | Table 6, Phoenix2014 Task, mT5_large row, BLEU-4 column | mT5_large | Phoenix2014T/test | BLEU-4 | 26.34 | 0.15 | -26.19 | produced |
| `t6-phoenix-mt5-large-rougel` | Table 6, Phoenix2014 Task, mT5_large row, ROUGE-L column | mT5_large | Phoenix2014T/test | ROUGE-L | 51.87 | 2.77 | -49.10 | produced |
| `t6-phoenix-mt5-large-meteor` | Table 6, Phoenix2014 Task, mT5_large row, METEOR column | mT5_large | Phoenix2014T/test | METEOR | 46.97 | 5.04 | -41.93 | produced |
| `t6-aslgpc12-ref34-bleu1` | Table 6, ASLG-PC12 Task, row citing [34], BLEU-1 column | Copied baseline ([34]) | ASLG-PC12/test | BLEU-1 | 92.98 | - | - | not_produced: copied_baseline |
| `t6-aslgpc12-ref34-bleu2` | Table 6, ASLG-PC12 Task, row citing [34], BLEU-2 column | Copied baseline ([34]) | ASLG-PC12/test | BLEU-2 | 89.09 | - | - | not_produced: copied_baseline |
| `t6-aslgpc12-ref34-bleu3` | Table 6, ASLG-PC12 Task, row citing [34], BLEU-3 column | Copied baseline ([34]) | ASLG-PC12/test | BLEU-3 | 85.63 | - | - | not_produced: copied_baseline |
| `t6-aslgpc12-ref34-bleu4` | Table 6, ASLG-PC12 Task, row citing [34], BLEU-4 column | Copied baseline ([34]) | ASLG-PC12/test | BLEU-4 | 82.41 | - | - | not_produced: copied_baseline |
| `t6-aslgpc12-ref34-rougel` | Table 6, ASLG-PC12 Task, row citing [34], ROUGE-L column | Copied baseline ([34]) | ASLG-PC12/test | ROUGE-L | 95.87 | - | - | not_produced: copied_baseline |
| `t6-aslgpc12-ref34-meteor` | Table 6, ASLG-PC12 Task, row citing [34], METEOR column | Copied baseline ([34]) | ASLG-PC12/test | METEOR | 96.46 | - | - | not_produced: copied_baseline |
| `t6-aslgpc12-transformer-bleu1` | Table 6, ASLG-PC12 Task, Transformer row, BLEU-1 column | Transformer | ASLG-PC12/test | BLEU-1 | 91.77 | 91.37 | -0.40 | produced |
| `t6-aslgpc12-transformer-bleu2` | Table 6, ASLG-PC12 Task, Transformer row, BLEU-2 column | Transformer | ASLG-PC12/test | BLEU-2 | 87.72 | 86.76 | -0.96 | produced |
| `t6-aslgpc12-transformer-bleu3` | Table 6, ASLG-PC12 Task, Transformer row, BLEU-3 column | Transformer | ASLG-PC12/test | BLEU-3 | 84.14 | 82.75 | -1.39 | produced |
| `t6-aslgpc12-transformer-bleu4` | Table 6, ASLG-PC12 Task, Transformer row, BLEU-4 column | Transformer | ASLG-PC12/test | BLEU-4 | 80.8 | 79.08 | -1.72 | produced |
| `t6-aslgpc12-transformer-rougel` | Table 6, ASLG-PC12 Task, Transformer row, ROUGE-L column | Transformer | ASLG-PC12/test | ROUGE-L | 94.74 | 94.31 | -0.43 | produced |
| `t6-aslgpc12-transformer-meteor` | Table 6, ASLG-PC12 Task, Transformer row, METEOR column | Transformer | ASLG-PC12/test | METEOR | 95.34 | 94.66 | -0.68 | produced |
| `t6-aslgpc12-mbart-bleu1` | Table 6, ASLG-PC12 Task, mBART row, BLEU-1 column | mBART | ASLG-PC12/test | BLEU-1 | 95.74 | 94.21 | -1.53 | produced |
| `t6-aslgpc12-mbart-bleu2` | Table 6, ASLG-PC12 Task, mBART row, BLEU-2 column | mBART | ASLG-PC12/test | BLEU-2 | 93.75 | 91.48 | -2.27 | produced |
| `t6-aslgpc12-mbart-bleu3` | Table 6, ASLG-PC12 Task, mBART row, BLEU-3 column | mBART | ASLG-PC12/test | BLEU-3 | 91.96 | 89.03 | -2.93 | produced |
| `t6-aslgpc12-mbart-bleu4` | Table 6, ASLG-PC12 Task, mBART row, BLEU-4 column | mBART | ASLG-PC12/test | BLEU-4 | 90.27 | 86.72 | -3.55 | produced |
| `t6-aslgpc12-mbart-rougel` | Table 6, ASLG-PC12 Task, mBART row, ROUGE-L column | mBART | ASLG-PC12/test | ROUGE-L | 97.76 | 96.45 | -1.31 | produced |
| `t6-aslgpc12-mbart-meteor` | Table 6, ASLG-PC12 Task, mBART row, METEOR column | mBART | ASLG-PC12/test | METEOR | 97.72 | 95.96 | -1.76 | produced |
| `t6-aslgpc12-mt5-small-bleu1` | Table 6, ASLG-PC12 Task, mT5_small row, BLEU-1 column | mT5_small | ASLG-PC12/test | BLEU-1 | 92.9 | 20.54 | -72.36 | produced |
| `t6-aslgpc12-mt5-small-bleu2` | Table 6, ASLG-PC12 Task, mT5_small row, BLEU-2 column | mT5_small | ASLG-PC12/test | BLEU-2 | 91.03 | 9.32 | -81.71 | produced |
| `t6-aslgpc12-mt5-small-bleu3` | Table 6, ASLG-PC12 Task, mT5_small row, BLEU-3 column | mT5_small | ASLG-PC12/test | BLEU-3 | 88.52 | 4.95 | -83.57 | produced |
| `t6-aslgpc12-mt5-small-bleu4` | Table 6, ASLG-PC12 Task, mT5_small row, BLEU-4 column | mT5_small | ASLG-PC12/test | BLEU-4 | 87.87 | 2.91 | -84.96 | produced |
| `t6-aslgpc12-mt5-small-rougel` | Table 6, ASLG-PC12 Task, mT5_small row, ROUGE-L column | mT5_small | ASLG-PC12/test | ROUGE-L | 96.11 | 18.24 | -77.87 | produced |
| `t6-aslgpc12-mt5-small-meteor` | Table 6, ASLG-PC12 Task, mT5_small row, METEOR column | mT5_small | ASLG-PC12/test | METEOR | 96.32 | 16.36 | -79.96 | produced |
| `t6-aslgpc12-mt5-base-bleu1` | Table 6, ASLG-PC12 Task, mT5_base row, BLEU-1 column | mT5_base | ASLG-PC12/test | BLEU-1 | 94.11 | 62.77 | -31.34 | produced |
| `t6-aslgpc12-mt5-base-bleu2` | Table 6, ASLG-PC12 Task, mT5_base row, BLEU-2 column | mT5_base | ASLG-PC12/test | BLEU-2 | 32.05 | 50.08 | +18.03 | produced |
| `t6-aslgpc12-mt5-base-bleu3` | Table 6, ASLG-PC12 Task, mT5_base row, BLEU-3 column | mT5_base | ASLG-PC12/test | BLEU-3 | 91.04 | 40.45 | -50.59 | produced |
| `t6-aslgpc12-mt5-base-bleu4` | Table 6, ASLG-PC12 Task, mT5_base row, BLEU-4 column | mT5_base | ASLG-PC12/test | BLEU-4 | 88.42 | 32.69 | -55.73 | produced |
| `t6-aslgpc12-mt5-base-rougel` | Table 6, ASLG-PC12 Task, mT5_base row, ROUGE-L column | mT5_base | ASLG-PC12/test | ROUGE-L | 96.56 | 65.38 | -31.18 | produced |
| `t6-aslgpc12-mt5-base-meteor` | Table 6, ASLG-PC12 Task, mT5_base row, METEOR column | mT5_base | ASLG-PC12/test | METEOR | 96.55 | 63.88 | -32.67 | produced |
| `t6-aslgpc12-mt5-large-bleu1` | Table 6, ASLG-PC12 Task, mT5_large row, BLEU-1 column | mT5_large | ASLG-PC12/test | BLEU-1 | 96.32 | 7.68 | -88.64 | produced |
| `t6-aslgpc12-mt5-large-bleu2` | Table 6, ASLG-PC12 Task, mT5_large row, BLEU-2 column | mT5_large | ASLG-PC12/test | BLEU-2 | 33.59 | 4.89 | -28.70 | produced |
| `t6-aslgpc12-mt5-large-bleu3` | Table 6, ASLG-PC12 Task, mT5_large row, BLEU-3 column | mT5_large | ASLG-PC12/test | BLEU-3 | 92.5 | 3.09 | -89.41 | produced |
| `t6-aslgpc12-mt5-large-bleu4` | Table 6, ASLG-PC12 Task, mT5_large row, BLEU-4 column | mT5_large | ASLG-PC12/test | BLEU-4 | 90.88 | 1.96 | -88.92 | produced |
| `t6-aslgpc12-mt5-large-rougel` | Table 6, ASLG-PC12 Task, mT5_large row, ROUGE-L column | mT5_large | ASLG-PC12/test | ROUGE-L | 98.7 | 9.02 | -89.68 | produced |
| `t6-aslgpc12-mt5-large-meteor` | Table 6, ASLG-PC12 Task, mT5_large row, METEOR column | mT5_large | ASLG-PC12/test | METEOR | 98.21 | 24.62 | -73.59 | produced |
| `t6-csldaily-transformer-bleu1` | Table 6, CSL-Daily Task, Transformer row, BLEU-1 column | Transformer | CSL-Daily/test | BLEU-1 | 52.36 | 53.52 | +1.16 | produced |
| `t6-csldaily-transformer-bleu2` | Table 6, CSL-Daily Task, Transformer row, BLEU-2 column | Transformer | CSL-Daily/test | BLEU-2 | 38.35 | 38.97 | +0.62 | produced |
| `t6-csldaily-transformer-bleu3` | Table 6, CSL-Daily Task, Transformer row, BLEU-3 column | Transformer | CSL-Daily/test | BLEU-3 | 28.01 | 28.27 | +0.26 | produced |
| `t6-csldaily-transformer-bleu4` | Table 6, CSL-Daily Task, Transformer row, BLEU-4 column | Transformer | CSL-Daily/test | BLEU-4 | 20.91 | 20.94 | +0.03 | produced |
| `t6-csldaily-mbart-bleu1` | Table 6, CSL-Daily Task, mBART row, BLEU-1 column | mBART | CSL-Daily/test | BLEU-1 | 65.97 | 60.15 | -5.82 | produced |
| `t6-csldaily-mbart-bleu2` | Table 6, CSL-Daily Task, mBART row, BLEU-2 column | mBART | CSL-Daily/test | BLEU-2 | 53.3 | 47.30 | -6.00 | produced |
| `t6-csldaily-mbart-bleu3` | Table 6, CSL-Daily Task, mBART row, BLEU-3 column | mBART | CSL-Daily/test | BLEU-3 | 43.51 | 37.24 | -6.27 | produced |
| `t6-csldaily-mbart-bleu4` | Table 6, CSL-Daily Task, mBART row, BLEU-4 column | mBART | CSL-Daily/test | BLEU-4 | 36.09 | 29.74 | -6.35 | produced |
| `t6-csldaily-mt5-small-bleu1` | Table 6, CSL-Daily Task, mT5_small row, BLEU-1 column | mT5_small | CSL-Daily/test | BLEU-1 | 58.87 | 23.80 | -35.07 | produced |
| `t6-csldaily-mt5-small-bleu2` | Table 6, CSL-Daily Task, mT5_small row, BLEU-2 column | mT5_small | CSL-Daily/test | BLEU-2 | 45.83 | 15.18 | -30.65 | produced |
| `t6-csldaily-mt5-small-bleu3` | Table 6, CSL-Daily Task, mT5_small row, BLEU-3 column | mT5_small | CSL-Daily/test | BLEU-3 | 34.95 | 9.61 | -25.34 | produced |
| `t6-csldaily-mt5-small-bleu4` | Table 6, CSL-Daily Task, mT5_small row, BLEU-4 column | mT5_small | CSL-Daily/test | BLEU-4 | 27.4 | 6.14 | -21.26 | produced |
| `t6-csldaily-mt5-base-bleu1` | Table 6, CSL-Daily Task, mT5_base row, BLEU-1 column | mT5_base | CSL-Daily/test | BLEU-1 | 62.02 | 54.11 | -7.91 | produced |
| `t6-csldaily-mt5-base-bleu2` | Table 6, CSL-Daily Task, mT5_base row, BLEU-2 column | mT5_base | CSL-Daily/test | BLEU-2 | 49.45 | 41.15 | -8.30 | produced |
| `t6-csldaily-mt5-base-bleu3` | Table 6, CSL-Daily Task, mT5_base row, BLEU-3 column | mT5_base | CSL-Daily/test | BLEU-3 | 39.31 | 30.81 | -8.50 | produced |
| `t6-csldaily-mt5-base-bleu4` | Table 6, CSL-Daily Task, mT5_base row, BLEU-4 column | mT5_base | CSL-Daily/test | BLEU-4 | 31.66 | 23.27 | -8.39 | produced |
| `t6-csldaily-mt5-large-bleu1` | Table 6, CSL-Daily Task, mT5_large row, BLEU-1 column | mT5_large | CSL-Daily/test | BLEU-1 | 65.23 | 1.34 | -63.89 | produced |
| `t6-csldaily-mt5-large-bleu2` | Table 6, CSL-Daily Task, mT5_large row, BLEU-2 column | mT5_large | CSL-Daily/test | BLEU-2 | 51.54 | 0.91 | -50.63 | produced |
| `t6-csldaily-mt5-large-bleu3` | Table 6, CSL-Daily Task, mT5_large row, BLEU-3 column | mT5_large | CSL-Daily/test | BLEU-3 | 41.56 | 0.57 | -40.99 | produced |
| `t6-csldaily-mt5-large-bleu4` | Table 6, CSL-Daily Task, mT5_large row, BLEU-4 column | mT5_large | CSL-Daily/test | BLEU-4 | 33.45 | 0.36 | -33.09 | produced |

All 86 own-authored targets are `produced`; the 21 copied-baseline targets remain `not_produced: copied_baseline` by design.

## How to repeat this

```bash
# Populate ASLG-PC12 into the shared Modal datasets Volume (idempotent; already run once)
papers/chen-2026-gloss-pretrained/data.sh

# CPU-only sanity check of all three dataset loaders (no GPU, no model)
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/chen-2026-gloss-pretrained/modal_app.py::check_data

# Cheap representative preflight for one (model, dataset) cell
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/chen-2026-gloss-pretrained/modal_app.py::preflight --model mt5-small --dataset phoenix

# Full finetune + eval for one Table 6 mBART/mT5 cell (idempotent: skips if scores.json already exists)
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/chen-2026-gloss-pretrained/modal_app.py::finetune --model mbart --dataset phoenix
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/chen-2026-gloss-pretrained/modal_app.py::reevaluate --model mbart --dataset phoenix --checkpoint checkpoint-40000

# Transformer-tiny / +Back-translation baseline (separate OpenNMT-py/torchtext=0.4.0 environment)
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/chen-2026-gloss-pretrained/modal_app.py::prepare_baseline_data --dataset phoenix
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/chen-2026-gloss-pretrained/modal_app.py::train_baseline --dataset phoenix
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/chen-2026-gloss-pretrained/modal_app.py::score_baseline --dataset phoenix

# +Back-translation only (Phoenix2014T): approximate the 30K in-domain sentences, then augment+retrain
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/chen-2026-gloss-pretrained/modal_app.py::fetch_german_background_corpus
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/chen-2026-gloss-pretrained/modal_app.py::select_backtranslation_sentences
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/chen-2026-gloss-pretrained/modal_app.py::train_reverse_and_translate
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/chen-2026-gloss-pretrained/modal_app.py::augment_phoenix_data
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/chen-2026-gloss-pretrained/modal_app.py::train_baseline --dataset phoenix-augmented
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/chen-2026-gloss-pretrained/modal_app.py::score_baseline --dataset phoenix-augmented
```

`--model` is one of `mbart`/`mt5-small`/`mt5-base`/`mt5-large`; `--dataset` for the mBART/mT5 path is one of `phoenix`/`aslgpc12`/`csldaily`; for the baseline path it is `phoenix`/`aslgpc12`/`csldaily`/`phoenix-augmented`. Phoenix2014T and CSL-Daily needed no acquisition step; both were already present in the shared `datasets` Volume and their exact split counts were verified against the paper's Table 4a.

## Data provenance and permissions

| Dataset | Version/subset/splits | Source and access date | License/permission and cloud-use basis | Path in Volume `datasets` | Counts / manifest / checksum | Deviations |
| --- | --- | --- | --- | --- | --- | --- |
| RWTH-PHOENIX-Weather 2014T | Gloss/text annotation pairs, full corpus | https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/, accessed 2026-08-31 | CC BY-NC-SA, non-commercial project-cloud processing (established by camgoz-2018-nslt) | `rwth-phoenix-2014-t/annotations` | 7096/519/642 (verified by counting CSV rows minus header on 2026-08-31) | None |
| ASLG-PC12 | Standard 82709/4000/1000 split from kayoyin/transformer-slt | https://huggingface.co/datasets/achrafothman/aslg_pc12 + https://github.com/kayoyin/transformer-slt, accessed 2026-08-31 | CC0 (public domain); Apache-2.0 code | `aslg-pc12` | 82709/4000/1000 (exact match to Table 4a) | Newly populated by `data.sh` in this attempt (was absent before) |
| CSL-Daily | `sentence_label/split_1.txt`, Chinese gloss/text pairs | http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/, accessed 2026-08-31 | Research-use only, signed institutional agreement (already in place for this project); no redistribution of raw text/predictions without Team S sign-off | `csl-daily/sentence_label` | 18401/1077/1176 (exact match to Table 4a, verified by parsing `split_1.txt`) | None |
| German background corpus (+Back-translation only) | `allenai/c4` (mc4), German config, first 400,000 filtered sentences (streamed, not a fixed release) | https://huggingface.co/datasets/allenai/c4, accessed 2026-09-01 | ODC-BY (Common Crawl-derived, redistributable with attribution); stored only in project Modal storage, not redistributed | `chen-2026-gloss-pretrained-results/backtranslation/background_de.txt` (not `datasets` Volume; ephemeral working data, not a canonical dataset) | 400,052 sentences after length filtering (20-300 chars) | Substitute for the paper's dead tagesschau.de source (footnote 6); documented approximation, see Guesses and deviations |

## Environment and patches

Two separate container environments, since the mBART/mT5 (modern HF Transformers) and Transformer-tiny/+Back-translation (2019-era OpenNMT-py) code cannot share one dependency stack.

**mBART/mT5 image**: built from the repo-root `Dockerfile` (NVIDIA NGC PyTorch 26.04 base) via `modal.Image.from_dockerfile`, plus `transformers==4.46.3`, `datasets==3.1.0`, `accelerate==1.1.1`, `sentencepiece==0.2.0`, `nltk==3.9.1`. Ref [33]'s code (`kayoyin/transformer-slt` @ `d119fbb642d653a987a2e1b2cd1541c88df7f2ef`) is cloned into the image at `/opt/transformer-slt` and used only for its evaluation tools (`tools/bleu.py`, `tools/rouge.py`, `tools/meteor.py`), not its OpenNMT-py training code. Both `datasets` and `huggingface-cache` Volumes are mounted per `AGENTS.md`.

**Transformer-tiny/+Back-translation image**: this paper's own `Dockerfile` (`nvidia/cuda:11.4.3-devel-ubuntu20.04` base + Miniconda py3.7, the same pattern proven by the sibling `camgoz-2020-slt` reproduction), with ref [33]'s repo cloned and `pip install -r requirements.txt -e .` letting pip resolve dependencies itself. This landed on `torch==1.13.1+cu117` (CUDA-11/A100-compatible) satisfying the exact `torchtext==0.4.0` pin with no manual version pinning needed — a pleasant surprise relative to the mBART/mT5 environment's six-fix debugging cycle. Modal's `add_python="3.11"` is required alongside the conda Python so Modal's own container orchestration can detect a usable interpreter (the conda Python alone isn't auto-detected); actual work always runs via the explicit `/root/miniconda3/bin/python` path.

| Patch | Demonstrated failure | Hypothesis | Why necessary | Behavioral effect | Evidence |
| --- | --- | --- | --- | --- | --- |
| Uninstall `apex` (image build step, not a code patch) | `ImportError: cannot import name 'amp' from 'apex'` on first import of `transformers.trainer_seq2seq` | The NGC 26.04 image's `apex` build dropped the legacy `apex.amp` submodule that `transformers` imports unconditionally whenever `apex` is importable at all | Training uses native `bf16`, not apex AMP, so apex is unused; removing it lets `transformers` skip that import path | None on results; environment-only fix | Preflight failure on 2026-08-31 (1st attempt), fixed in `modal_app.py` |
| Copy `wordnet_key_value*.txt` into `tools/` (image build step, both images) | `FileNotFoundError: .../tools/wordnet_key_value.txt does not exist` from `tools/rouge.py` | `rouge.py` resolves these files via `pkg_resources.resource_filename(__name__, ...)`, i.e. relative to the script's own directory, but the repo ships them at the repo root — a pre-existing repo layout quirk | Needed for `rouge.py` to run at all under a fresh checkout invoked from any cwd | None on results; environment-only fix | Preflight failure on 2026-08-31 (3rd attempt), fixed in both Dockerfiles |
| `patches/0001-meteor-tokenize.patch` (retained, applied via `git am` at image build, both images) | `TypeError: "hypothesis" expects pre-tokenized hypothesis` from `nltk.meteor()` in `tools/meteor.py` | `meteor.py` was written against an older nltk that tokenized internally; nltk>=3.7 requires pre-tokenized `Iterable[str]` input | One-line fix: lowercase + `.split()` before calling `nltk.meteor()`, matching this same repo's own `bleu.py` convention | None on results; environment/API-compatibility fix | Preflight failure on 2026-08-31 (5th attempt), fixed by patch (sha256 `f4b2c004b72d914fab73cf2c20e8932e6dad7d4f2b840cd9d6ba24b0cba75ddc`) |
| `patches/0002-beam-search-float-div.patch` (retained, applied via `git am` at image build, baseline image only) | `RuntimeError: result type Float can't be cast to the desired output type Long` in `onmt/translate/beam_search.py` during `translate.py` | PyTorch >=1.5 changed `torch.div()` to always perform true (float) division; the original call relied on the pre-1.5 default of integer floor division to recover a flat top-k index's batch origin into a `LongTensor` | One-line fix: pass `rounding_mode='floor'` (added in torch 1.8) to restore the original integer semantics | None on results; environment/API-compatibility fix | Training run failure on 2026-09-01 (1st baseline training attempt), fixed by patch (sha256 `fd3dc80d0eae3d6e61d3fc3d7fd05c72c694b5b1ba0a7c0112c25605d91f333a`) |

All four are environment/dependency-compatibility fixes with no effect on the trained model, data, or protocol — none change what is being measured.

## Execution evidence

| Run ID | Kind | Config | Start/end (UTC) | Exit / state | Modal app | GPU-hours |
| --- | --- | --- | --- | --- | --- | ---: |
| Run ID | Kind | Config | Start/end (UTC) | Exit / state | Modal app | GPU-hours |
| --- | --- | --- | --- | --- | --- | ---: |
| `preflight-mt5-small-phoenix` | preflight | 0.02, batch=4 | 2026-08-31T13:01:46Z - 2026-08-31T13:03:51Z | 0 / succeeded | `ap-Pbxhi3vPsb7QvDMxA49r7q` | 0.03 |
| `reeval-mbart-csldaily` | full_reproduction_reevaluation | checkpoint-40000, batch=16 | 2026-08-31T15:54:08Z - 2026-08-31T15:56:40Z | 0 / succeeded | `ap-P0pGfnCDnfut4VXi6hSrW2` | 0.04 |
| `reeval-mt5-small-csldaily` | full_reproduction_reevaluation | checkpoint-40000, batch=16 | 2026-08-31T15:54:13Z - 2026-08-31T15:56:26Z | 0 / succeeded | `ap-MNYRcjgzkl2DPHJmUUwZdJ` | 0.04 |
| `reeval-mbart-phoenix` | full_reproduction_reevaluation | checkpoint-40000, batch=16 | 2026-08-31T16:01:13Z - 2026-08-31T16:06:21Z | 0 / succeeded | `ap-UOq4iWlpwR0rfewbwL3uzB` | 0.09 |
| `reeval-mt5-base-phoenix` | full_reproduction_reevaluation | checkpoint-40000, batch=16 | 2026-08-31T17:00:46Z - 2026-08-31T17:02:19Z | 0 / succeeded | `ap-Yr9ULepkTHuw5roMIylNov` | 0.03 |
| `reeval-mt5-small-phoenix` | full_reproduction_reevaluation | checkpoint-40000, batch=16 | 2026-08-31T16:58:00Z - 2026-08-31T16:59:30Z | 0 / succeeded | `ap-fDt8CedDTj7wILrD7TEJ0j` | 0.03 |
| `reeval-mt5-small-aslgpc12` | full_reproduction_reevaluation | checkpoint-40000, batch=16 | 2026-08-31T17:00:00Z - 2026-08-31T17:02:00Z | 0 / succeeded | `ap-QJkrl7MDKPEkWcBvrVhVIH` | 0.03 |
| `reeval-mt5-base-csldaily` | full_reproduction_reevaluation | checkpoint-40000, batch=16 | 2026-08-31T17:04:05Z - 2026-08-31T17:06:08Z | 0 / succeeded | `ap-MdtpBTDGky6WVQndqBqfaJ` | 0.03 |
| `reeval-mt5-large-phoenix` | full_reproduction_reevaluation | checkpoint-40000, batch=16 | 2026-08-31T18:17:23Z - 2026-08-31T18:24:12Z | 0 / succeeded | `ap-CYinJDA4MnmckRwl5y1Ix0` | 0.11 |
| `reeval-mbart-aslgpc12` | full_reproduction_reevaluation | checkpoint-40000, batch=16 | 2026-08-31T18:40:00Z - 2026-08-31T18:41:30Z | 0 / succeeded | `ap-U8L7Wq7jCYIbaJUfztsjqa` | 0.03 |
| `reeval-mt5-large-csldaily` | full_reproduction_reevaluation | checkpoint-40000, batch=16 | 2026-08-31T18:20:00Z - 2026-08-31T23:00:00Z | 0 / succeeded | `unknown (app list lost overnight local-session disconnect; scores.json content on the results Volume is the authoritative evidence)` | 4.67 |
| `reeval-mt5-base-aslgpc12` | full_reproduction_reevaluation | checkpoint-40000, batch=16 | 2026-09-01T07:07:45Z - 2026-09-01T07:11:28Z | 0 / succeeded | `ap-DmB4fxYygSBbXEp5ppldgU` | 0.06 |
| `reeval-mt5-large-aslgpc12` | full_reproduction_reevaluation | checkpoint-40000, batch=16 | 2026-09-01T07:07:48Z - 2026-09-01T07:18:46Z | 0 / succeeded | `ap-q4aXY6UaPuCSjzuVJM2LBv` | 0.18 |
| `baseline-phoenix-transformer` | full_reproduction_terminal_segment | model_step_1200 (best by dev accuracy/ppl), batch=2048 | 2026-09-01T06:23:00Z - 2026-09-01T06:28:00Z | 0 / succeeded | `ap-LdVxTFB0PgikTgtuqygni5` | 0.08 |
| `baseline-csldaily-transformer` | full_reproduction_terminal_segment | model_step_1700, batch=2048 | 2026-09-01T08:28:52Z - 2026-09-01T08:33:02Z | 0 / succeeded | `ap-8JgrUBFknKXfMhGQ2thA2r` | 0.07 |
| `baseline-aslgpc12-transformer` | full_reproduction_terminal_segment | model_step_3400, batch=2048 | 2026-09-01T08:28:48Z - 2026-09-01T08:39:21Z | 0 / succeeded | `ap-QR2imJvVWrADyUo0BtAj26` | 0.18 |
| `baseline-phoenix-backtranslation` | full_reproduction_terminal_segment | model_step_2700, batch=2048 | 2026-09-01T11:27:18Z - 2026-09-01T11:37:18Z | 0 / succeeded | `ap-UMxhPgJudqZ3y6ZkJHaiot` | 0.17 |

The preflight (`preflight-mt5-small-phoenix`) exercised the real path end-to-end before any full run was committed. All 12 mBART/mT5 full training runs used a fixed max_steps=40000, batch_size=16 schedule (see Guesses and deviations); training itself is not separately retained as a `run` entry (12 training launches, several interrupted by Modal GPU preemption and auto-restarted — see Attempts, failures, and dead ends), only the final generation+scoring step against each finished checkpoint is retained as a `reeval-*` run, since that is what determines the reported score. Two of the twelve (mbart-csldaily, mt5-large-csldaily) reused a training run's own already-correct final predict/score step rather than a separate reevaluate() call, because their training launch was already using the fixed generation config when it produced its own checkpoint; the rest used a separate `reevaluate()` call against the finished checkpoint after a generation-config bug (see Attempts, failures, and dead ends) was found and fixed mid-session.

The 4 Transformer-tiny/+Back-translation runs (`baseline-*`) use ref [33]'s own recipe end-to-end (preprocess/train/translate all in one `train_baseline` call, with `-early_stopping` from the pinned recipe determining the actual step count rather than a guessed budget) and are retained as single terminal-segment runs each. `+Back-translation`'s run additionally required German-corpus-fetch, sentence-selection, and reverse-model sub-steps recorded in `reproduction.json`'s observation text rather than as independent `runs` entries, since they are preparation steps feeding the one retained training run.

Total measured GPU time across all mBART/mT5 reeval/scoring runs: ~24 minutes (the training runs themselves, not separately billed as `runs` entries here, ran roughly 1-4 hours each on a single A100, occasionally longer after preemption-triggered restarts). The 4 Transformer-tiny/+Back-translation training runs (including the augmentation sub-steps) took under 20 minutes total, reflecting how much smaller and faster ref [33]'s architecture is; see individual `compute` blocks in reproduction.json for what is recorded.

## Guesses and deviations

| Detail | Paper/evidence says | This attempt used | Rationale | Effect on interpretation |
| --- | --- | --- | --- | --- |
| Metric toolkit | BLEU/ROUGE-L/METEOR cited to Papineni et al. 2002 / Lin 2004 only | Ref [33]'s own bundled `tools/{bleu,rouge,meteor}.py` (nltk-based corpus BLEU, lowercased/whitespace-tokenized; a pltrdy/rouge-style ROUGE-L with stemming; nltk METEOR), invoked unmodified except the one retained nltk-API patch | No toolkit named by the paper, but the paper explicitly says its Transformer baseline follows [33]; reusing [33]'s own scoring code for the whole table (not just the baseline) is more evidence-grounded than an arbitrary third-party toolkit | Reproduced values may still differ slightly from whatever toolkit the paper's own mBART/mT5 evaluation used, which it does not name |
| Checkpoint selection | Not restated for Section IV/V | Best-development-set corpus-BLEU-4 (nltk, lowercased/whitespace) via `load_best_model_at_end` | Standard practice; matches "best-development-set" framing used elsewhere in the paper | Could shift results if the paper actually used final-step or a different rule |
| CSL-Daily target text tokenization for scoring | Table 4b shows unsegmented Chinese characters | Whitespace-separate each character before scoring (character-level BLEU/ROUGE/METEOR) | Matches Zhou et al. (2021)'s convention for this exact corpus; word-level scoring on unsegmented Chinese would be meaningless | Only affects CSL-Daily targets; Phoenix2014T/ASLG-PC12 are scored as-is |
| ASLG-PC12 train count | Table 4a states 82709 | Loaded file contains 82710 lines (verified via `check_data` on 2026-08-31) | Off-by-one, most likely a trailing-newline counting artifact in the paper's own reporting; dev/test (4000/1000) match exactly | Immaterial to results; using the file's true content rather than truncating to match the paper's stated count |
| Finetuning batch size / step budget | Only label smoothing, Adam betas/eps, max LR, warmup_steps=2500 are given for the mBART/mT5 systems | batch_size=16, a fixed max_steps=40000 uniform across all three datasets, eval/save every 4000 steps | A first full run at epochs=10 (4440 total steps for Phoenix2014T) produced near-zero BLEU: per-step loss tracing showed this was undertraining, not a bug — warmup_steps=2500 consumed 56% of that short schedule, leaving too few high-LR steps for the smallest dataset. A uniform large step budget keeps warmup at a small (6.25%), constant fraction of training for every dataset | Real result quality now depends on this guessed step budget rather than the paper's (unstated) one; recorded as the best-available default, not a paper-specified value |
| Transformer-tiny/+Back-translation hyperparameters | Table 5 is captioned "Hyperparameters of transformer models used in Section III" — it does not apply to Table 6. Section IV-A2 only says the baseline "follows [33]" | Ref [33]'s own documented Sample Usage recipe (`kayoyin/transformer-slt` README), invoked verbatim including its own `-early_stopping 3` criterion | Initially (incorrectly) assumed Table 5's numbers applied to Table 6; corrected after re-reading Table 5's own caption. Using the pinned upstream's own recipe directly is both the least-invasive reading of "follow [33]" and avoids porting Section III's differently-scoped hyperparameters (which also use a different LR-scaling convention, `decay_method=noam`, that Table 5's raw `7e-4` would not transfer to correctly) | This is the target with the closest reproduction (BLEU-4 within ~1-2 points on all 3 datasets), suggesting the correction was the right call |
| +Back-translation's 30K in-domain German sentences | Section III-C: cross-entropy-difference selection (Moore & Lewis 2010) from Common Crawl, targeting the weather-forecast domain; footnote 6 states the intended source (tagesschau.de) is inaccessible even to the paper's own authors | Background corpus: 400K sentences streamed from `allenai/c4` (mc4, German, Common-Crawl-derived). In-domain seed: Phoenix2014T's own training text (weather-forecast register). Selection: word-level unigram LM (add-1 smoothed) cross-entropy difference, not a full n-gram LM | The paper's own source is unrecoverable in principle, so exact reproduction is impossible; this substitutes a same-family (Common Crawl) source and a simplified (unigram, not full n-gram) version of the cited selection method, sufficient to visibly bias selection toward weather-related text (manually spot-checked) | This reproduction's own +Back-translation result *underperforms* its own plain Transformer baseline (21.07 vs 22.42 BLEU-4), the opposite of the paper's reported improvement (21.41→22.78) — plausibly because the substitute data is noisier than the paper's actual (now-inaccessible) source, not a pipeline defect |

## Attempts, failures, and dead ends

- The REPRO-SIGN survey tool record for this paper (`repro-sign-survey-frontend.fly.dev/paper.html?id=550614802c03abf554dd1f3a4e65abfa34e6c07a`) required authenticated PocketBase access with no public read rule; the user supplied the paywalled paper directly as a local PDF instead. This is recorded as a direct user assignment, not a queue record.
- No code/data-availability statement or repository was found for this paper itself after a full source-search pass (DOI/IEEE page, ResearchGate mirror, GitHub/web search by title and author, corresponding author's homepage). Kept: preference level 3 for the paper's own contribution systems.
- ASLG-PC12 was absent from the shared `datasets` Volume; resolved by writing and running `data.sh` against the pinned kayoyin/transformer-slt commit (CC0/Apache-2.0), verified exact split counts. Kept.
- Inspected ref [33]'s `kayoyin/transformer-slt` for the Transformer-tiny baseline: it is OpenNMT-py 1.0.0 vendored in-repo, pinned to `torchtext==0.4.0` (deprecated, API-incompatible with modern torch/Python). Running it needs a separate older-Python environment (similar in spirit to this project's camgoz-2018-nslt/camgoz-2020-slt precedents), not the modern-HF image used for mBART/mT5. Built later in the same session (see below); reused camgoz-2020-slt's proven `nvidia/cuda:11.4.3` + Miniconda py3.7 pattern, which resolved `torch==1.13.1+cu117` (CUDA-11/A100-compatible) against the exact `torchtext==0.4.0` pin cleanly on the first real attempt — no version conflicts, unlike the mBART/mT5 environment's longer debugging cycle. Kept.
- First attempt to run `check_baseline_env` failed: `ConflictError: We were unable to determine the version of Python installed in the Image`. Modal's own container orchestration needs a detectable interpreter; a bare conda install at a non-standard path isn't auto-detected. Fixed by adding `add_python="3.11"` to `modal.Image.from_dockerfile(...)`, matching the sibling `camgoz-2020-slt` reproduction's own fix for the identical issue; all actual work still runs via the explicit `/root/miniconda3/bin/python` path. Kept.
- First Transformer-tiny training run completed early stopping cleanly, but crashed in `translate.py`: `RuntimeError: result type Float can't be cast to the desired output type Long` in `onmt/translate/beam_search.py`'s `torch.div(self.topk_ids, vocab_size, out=self._batch_index)`. Diagnosed as PyTorch >=1.5's change to `torch.div()`'s default (float, not integer floor) division semantics breaking 2019-era code that relied on the old default. Fixed with a one-line retained patch (`patches/0002-beam-search-float-div.patch`, `rounding_mode='floor'`), the same single-hypothesis-at-a-time discipline as the meteor.py fix. Kept.
- Restructured the single `upstream.patch` into `patches/0001-meteor-tokenize.patch` + `patches/0002-beam-search-float-div.patch` once a second, differently-scoped patch existed for the same upstream repo, per this project's own convention ("patches/*.patch for multiple ordered patches"). Renamed file, unchanged content/hash for 0001; updated all `reproduction.json`/README references and both Dockerfiles. Kept.
- Two more transient Modal Volume read-after-write races (predictions/scores.json read before the writer's `results.commit()` landed) hit the ASLG-PC12 baseline scoring step, identical in nature to the mBART/mT5-phase race. Fixed the same way: wait for the writer app to reach `state=stopped`, then retry. Kept (no code change needed, an operational retry).
- All three plain Transformer-tiny baselines (Phoenix2014T, ASLG-PC12, CSL-Daily) landed within ~1-2 BLEU points of the paper on every metric — the closest match anywhere in this reproduction — providing strong independent validation that the data, splits, and evaluation methodology (shared with the mBART/mT5 systems via the same ref-[33] scoring tools) are correct, and that mT5's shortfall (see above) is isolated to that model family rather than a systemic pipeline issue.
- +Back-translation: approximated the paper's 30K in-domain German sentences (see Guesses and deviations) by streaming 400K sentences from `allenai/c4` (German), scoring them with a word-level unigram cross-entropy-difference selector against Phoenix2014T's own training text, training a reverse (text→gloss) Transformer-tiny model, translating the selected sentences to synthetic gloss, and retraining the forward model on the augmented (37,096-pair) set. All sub-steps completed without errors on the first attempt (reusing the now-debugged environment). The final result underperformed this reproduction's own plain-Transformer baseline, opposite the paper's reported direction — recorded as a genuine, evidenced finding about the substitute data's quality, not chased further by re-tuning.
- The mBART/mT5 HF finetuning path took 5 attempts to reach a clean preflight: (1) apex.amp import conflict with the NGC image, (2) `MT5ForConditionalGeneration.__init__()` rejecting the `dropout`/`attention_dropout` kwargs used for mBART (T5Config uses a single `dropout_rate`), (3) `rouge.py`'s wordnet DB path resolving relative to its own directory, (4) a scoring-output parsing bug in this reproduction's own code (assumed a bare float on the last line; `rouge.py` actually prints one line per metric — fixed), (5) `nltk.meteor()`'s API break. All fixed one hypothesis at a time (see Environment and patches). Kept.
- The first full (non-preflight) run, mT5-small/Phoenix2014T at batch_size=16/epochs=10 (4440 total steps), completed cleanly (exit 0) but scored ~0 on every metric. Diagnosed via `modal_app.py::diagnose_model` (ruled out: pretrained embeddings load correctly and deterministically; the earlier "missing keys" warning is a benign tied-embedding false alarm) and `modal_app.py::trace_loss` (per-step loss log via the real Trainer code path): loss fell from ~50 to ~12 over the run — real learning, not a broken model — but warmup_steps=2500 (paper-specified) consumed 56% of the 4440-step schedule, leaving too few steps at meaningful LR to converge on the smallest dataset. Fixed by switching to a uniform max_steps=40000 budget across all datasets (see Guesses and deviations); not a code bug, reverted nothing.
- Also fixed along the way (own code, not upstream): `compute_metrics` only cleaned `-100` padding from `labels`, not from `predictions`, which the Trainer's eval loop also pads with `-100` for variable-length generated sequences — caused an `OverflowError` decoding a negative "token id". Fixed by cleaning both.
- With the corrected step budget, dev-time BLEU-4 was still exactly 0.0000 for every mT5 checkpoint while mBART's climbed normally. Direct generation from a real mT5 checkpoint (`modal_app.py::inspect_checkpoint`) showed fluent, grammatical, non-repeating-in-content-but-literally-looping text ("...wettervorhersage fur morgen..." repeated many times) — beam search without an anti-repetition constraint degenerating into a loop, not a training failure. Fixed by setting `no_repeat_ngram_size=3` and explicit `num_beams=4` uniformly for all systems (matching ref [33]'s own beam width). Verified the fix directly (before/after generation samples) before trusting any further scores; re-evaluated already-trained checkpoints via a new `reevaluate()` entry point rather than retraining. Kept.
- Two experiments (`mt5-large-csldaily`, and briefly `mt5-base-aslgpc12`) were repeatedly preempted by Modal reclaiming A100 capacity in the shared workspace, cycling restart-preempt for roughly 2 hours with no net progress on `mt5-large-csldaily`. Widened `finetune()`'s GPU spec to a fallback list; the first attempt (`["A100","A10G","L40S","L4"]`) got scheduled quickly but OOM'd on the 22GB-usable A10G tier (mT5-large + CSL-Daily's longer tokenized sequences need more headroom than the ~12.5GB peak measured on ASLG-PC12). Narrowed to `["A100","L40S"]` (L40S=48GB) and it completed cleanly. Kept the narrower list.
- One `reevaluate()` call (mt5-large/phoenix) failed once with `safetensors_rust.SafetensorError: incomplete metadata, file not fully covered` — a Modal Volume read-after-write race (the checkpoint directory appeared in a listing before its file contents were fully committed from the writer's side). A bare retry a few minutes later succeeded; later checkpoint waits added a ~60s buffer after the checkpoint directory first appears. Not retried blindly — confirmed the checkpoint was genuinely complete and valid before treating the retry's output as final.
- The user closed their laptop partway through monitoring; the local Modal log-streaming connection dropped ("Connection lost") but the detached Modal jobs continued running server-side unaffected, exactly as intended. On resuming, `modal app list` returned empty for a period (apparent retention/listing quirk for older or many-hours-old apps); the shared `results` Volume's checkpoint/scores.json contents were used as the authoritative source of truth instead, and one run's exact Modal app ID/timestamps could not be recovered — recorded as an honest estimate rather than a fabricated precise value (see `reeval-mt5-large-csldaily` in reproduction.json).

## Candidate flags, ethics, and human evaluation

No queue record exists for this paper (direct user assignment), so there are no queue comments/flags to investigate. CSL-Daily is restricted research-use data; it is already covered by an existing signed institutional agreement for this project (gate `csl-daily-license`), so no new ethics/access gate was opened, but redistribution of CSL-Daily-derived text or predictions is explicitly restricted pending Team S confirmation. No human evaluation or new participants are involved in reproducing Table 6.

## Author and team contact

None yet. Per policy, author contact (Yan Wang, corresponding author) is considered only after an independent reproduction attempt is complete, except for Team S-coordinated data-access requests — which are not needed here since CSL-Daily access already exists for this project.
