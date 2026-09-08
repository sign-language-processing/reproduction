# Partaourides 2021 — Variational Bayesian Seq2Seq for Memory-Efficient SLT — reproduction

**Paper ID:** `dbf0205f1d29b109ad5e29d6824ed965b511598a`

**Citation:** Partaourides, H., Voskou, A., Kosmopoulos, D., Chatzis, S., Metaxas, D.N.
*Variational Bayesian Sequence-to-Sequence Networks for Memory-Efficient Sign Language
Translation.* In: Pattern Recognition. ICPR International Workshops and Challenges 2020,
LNCS 12536, pp. 251–262. Springer (2021). arXiv:2102.06143v1 [stat.ML] 11 Feb 2021.

**Paper:** https://arxiv.org/abs/2102.06143 · **Code/artifacts:** none found after search (see Source provenance)

**Preference level:** 3

**Pipeline status:** `insufficient_information`

**Numerical agreement:** `not_assessed` — no in-scope target produced a comparable value; the reimplemented method rows carry unresolved, behaviour-changing hyperparameters and remain conditional evidence.

**Attempt date:** 2026-09-08

> This README is a work-in-progress skeleton. Results, run IDs, and the final status
> are filled in once the Modal runs complete.

## Scope and target contract

`what_to_reproduce` = "Table 1 on page 9". Table 1 ("Performance Metrics") reports
Dev and Test **BLEU-4** and **ROUGE** for six systems on the RWTH-PHOENIX-Weather 2014T
**Gloss2Text** task:

| Model | Dev BLEU-4 | Dev ROUGE | Test BLEU-4 | Test ROUGE |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 16.3 | 40.3 | 16.3 | 40.7 |
| GRU_repar | 16.7 | 41.1 | 17.0 | 41.5 |
| GRU_repar,wc | 16.2 | 40.6 | 16.7 | 40.7 |
| GRU_bp | 18.4 | 43.9 | 17.0 | 43.1 |
| SB-GRU | 17.9 | 43.0 | 18.1 | 43.5 |
| SB-GRU,wc | 17.7 | 43.0 | 17.8 | 42.8 |

24 numbers → 24 targets in `reproduction.json.targets` (`table1-<system>-<split>-<metric>`).

**System semantics** (paper §4.2): `Baseline` = the Camgoz 2018 [10] Gloss2Text GRU
*without attention*; `GRU_repar` = Gaussian weight-posterior reparameterization on the
GRU non-gate weights only; `GRU_bp` = IBP stick-breaking prior on the non-gate weight
utility indicators only; `SB-GRU` = both (the proposed model); the `,wc` rows are the
*same* trained model re-evaluated after post-hoc weight compression (bit-precision
reduction), so only four models are trained.

**Architecture** (paper §4): model from [10] — 4 encoder + 4 decoder layers, 1000 units
per layer, GRU, no attention; the last encoder layer is replaced by the proposed
recurrent variant. Adam, lr 1e-5, batch 128, dropout 0.2, "until convergence".
Implemented in TensorFlow by the authors; **reimplemented here in PyTorch** (framework
deviation; the variational math — Eqs. 1–15 — is framework-independent).

**Metrics:** paper says only "BLUE and ROUGE" with no implementation citation. Scored
here with `sacrebleu` (BLEU-4, `tok:13a`, `lc`) and `rouge-score` (ROUGE-L F1); exact
upstream implementations are unknown (guess — see below).

**Ambiguity / resolution:** target *identity* is unambiguous (Table 1, six named
systems, two splits, two metrics). What is unresolved is the *protocol* for the five
non-baseline rows — see "Guesses and deviations". Per the reproduction contract these
rows are treated as conditional evidence, not produced targets, independent of how
close the numbers land.

## Source provenance

| Artifact | Canonical source | Pinned revision / SHA-256 | Role |
| --- | --- | --- | --- |
| Paper PDF | https://arxiv.org/pdf/2102.06143 | `5fe99055e474d9109e3d2b75e1b00c56380f82234cf859e6ad38c2a649436b5b` | Target table and protocol |
| Published code | — | — | **None exists** (see below) |
| Reference [10] code | https://github.com/neccam/nslt | commit `06951580b58f04b9cd64efcf61aeca36011031d3` | Architectural reference for the Baseline row; consulted, not run |

**Source search performed (2026-09-08), nothing found:**

- arXiv abstract + PDF: no code/data link, no footnote URL; paper body only says
  "We implement our model in TensorFlow [1]".
- Springer chapter landing page (10.1007/978-3-030-64559-5_19): paywalled; ICPR 2020
  workshop paper, no code-availability statement or supplementary material indicated.
- First author GitHub `github.com/Partaourides`: repos `SERN`, `CUT_SDGs_Keyword-Mapping`
  — unrelated.
- Co-author Andreas Voskou: no public repo for this work found.
- Web / GitHub searches for "Stick-Breaking Recurrent", "SB-GRU", the exact title, and
  "Variational Bayesian sequence-to-sequence sign language" — no implementation.

Conclusion: preference level 3 (reimplement from the paper).

## Results

Filled in after the runs. Every non-baseline row is expected to be `not_produced`
(`protocol_ambiguous`) and reported as conditional evidence with raw numbers.

| Target ID | System | Split | Metric | Original | Reproduced | Difference | Terminal reason / evidence |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| _pending_ | | | | | | | |

## How to repeat this

```bash
# 0. Modal auth (once): modal setup  -> select workspace repro-sign
S=.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh

# 1. Environment / data check
$S run papers/partaourides-2021-variational-bayesian-slt/modal_app.py::check_env
$S run papers/partaourides-2021-variational-bayesian-slt/modal_app.py::preflight

# 2. Train the four models (each writes /outputs/<mode>/results.json on the
#    partaourides-2021-results volume)
for m in plain repar bp sb; do
  $S run --detach papers/partaourides-2021-variational-bayesian-slt/modal_app.py::train --mode $m
done

# 3. Weight-compression evals for GRU_repar and SB-GRU
$S run papers/partaourides-2021-variational-bayesian-slt/modal_app.py::evaluate --mode repar --quantize-bits 2
$S run papers/partaourides-2021-variational-bayesian-slt/modal_app.py::evaluate --mode sb --quantize-bits 2
```

Local smoke test (no GPU): `python sbgru_slt.py preflight --data-dir <dir with the 3 corpus CSVs>`.

## Data provenance and permissions

| Dataset | Version/subset/splits | Source and access date | License / cloud-use basis | Path in Volume `datasets` | Counts | Deviations |
| --- | --- | --- | --- | --- | --- | --- |
| RWTH-PHOENIX-Weather 2014T | Gloss2Text: `orth` → `translation`, text only (no video/features) | https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/ (2026-09-08) | Queue record: CC BY-SA 3.0; version page links CC BY-NC-SA 3.0. Text-only, non-commercial research on project cloud — permitted under either. | `rwth-phoenix-2014-t/annotations/PHOENIX-2014-T.{train,dev,test}.corpus.csv` | train 7096 / dev 519 / test 642 | none for data identity |

Verification: split counts exactly match the paper (§4.1, total 8257). German training
vocabulary = 2887 types / 1077 singletons — **exact** match to the paper. Gloss training
vocabulary = 1085 types / 355 singletons on raw whitespace split vs the paper's
1066 / 337; the ~19-type gap is attributed to an unspecified gloss-cleaning or
frequency-cutoff step and recorded as a guess.

File SHA-256:
- train `cc3dc2461f0a222b92f3927c24ac21c1467f3e5428b406ee7fe40bca1b0b8d44`
- dev `1085141d0ed6f28c6de6196a271b72c07366ed3fe5470c9717bd44640737f00b`
- test `632b19c9a87fb9c98b0821e04861750565348bce20a069c6dea1bba5bda27879`

## Environment and patches

Base image `nvcr.io/nvidia/pytorch:26.04-py3` (repo `Dockerfile`) + `sacrebleu==2.4.3`,
`rouge-score==0.1.2`. Single A10G GPU. HF cache mounted at `/cache/huggingface`
(`HF_HOME`, `HF_HUB_CACHE` set) although this reproduction downloads nothing from the Hub.

No upstream code is patched — there is no upstream code. The reimplementation lives in
`sbgru_slt.py`.

## Execution evidence

Filled in after the runs (Modal profile `repro-sign`, app
`partaourides-2021-variational-bayesian-slt`, results volume `partaourides-2021-results`).

## Guesses and deviations

| Detail | Paper says | This attempt used | Rationale |
| --- | --- | --- | --- |
| Framework | TensorFlow | PyTorch reimplementation | No published code; math is framework-independent |
| Embedding dim | — | 1000 | Match "1000 units per layer" |
| Encoder directionality | — | unidirectional 4-layer GRU | Simplest reading of "4 layers in the encoder" |
| Convergence criterion | "until convergence" | early stop on dev BLEU-4, patience 20 evals, ≤400 epochs | Standard; consistent with [10] |
| Checkpoint selection | — | best dev BLEU-4 | Standard |
| Beam width / length penalty | — | beam 3, lp 0 | Camgoz-style |
| BLEU / ROUGE implementation | "BLUE and ROUGE" | sacrebleu 13a/lc; rouge-score ROUGE-L F1 | Reproducible; [10] uses tensor2tensor-style scripts |
| Gloss preprocessing | — | raw whitespace split | vocab 1085 vs paper 1066 |
| IBP innovation α | — | 1.0 | "controls sparsity"; no value given |
| Cutoff threshold τ | mentioned, "robust to small fluctuations" | 0.5 | Median; no value given |
| Gumbel-Softmax temperature λ schedule | "annealed similar to [19]" | 1.0 → 0.5 linear over 20k steps | [19] = Jang et al.; no schedule given |
| MC samples in ELBO | "draw MC samples" | 1 | Typical SGVB; count not given |
| KL warm-up | — | linear 0→1 over 2000 steps | Standard VAE practice; not mentioned |
| Kumaraswamy a,b init | — | a=b=1 | Neutral |
| KL(q(W)‖p(W)) estimator | MC | MC (closed-form available via flag) | Faithful to paper |
| Weight-compression rule | "unit round off", "factor of 16", "bit precision 1" | uniform per-matrix 2-bit quantisation of the last encoder layer's effective W_y | Genuinely under-specified; ≈16× vs float32 |
| Seed policy | — | single seed = 1 | Paper reports one value per cell, no seeds/std |

## Attempts, failures, and dead ends

Filled in as the work proceeds.

## Candidate flags, ethics, and human evaluation

Queue flags: `includes_human_evaluation = no`, `potential_ethical_concerns = no`,
`copied_scores = no`. Gloss2Text uses only released text annotations — no participant
interaction, no sensitive data. Queue `comments` note the dataset was recorded as
"PHOENIX-2014-T ... subset gloss2text"; confirmed this is the standard 2014T release and
the paper's own dataset ("RWTH-PHOENIX-Weather 2014T", §4.1).

## Author and team contact

None.
