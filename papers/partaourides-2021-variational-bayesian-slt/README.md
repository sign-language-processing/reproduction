# Partaourides 2021 — Variational Bayesian Seq2Seq for Memory-Efficient SLT — reproduction

**Paper ID:** `dbf0205f1d29b109ad5e29d6824ed965b511598a`

**Citation:** Partaourides, H., Voskou, A., Kosmopoulos, D., Chatzis, S., Metaxas, D.N.
*Variational Bayesian Sequence-to-Sequence Networks for Memory-Efficient Sign Language
Translation.* In: Pattern Recognition. ICPR International Workshops and Challenges 2020,
LNCS 12536, pp. 251–262. Springer (2021). arXiv:2102.06143v1 [stat.ML] 11 Feb 2021.

**Paper:** https://arxiv.org/abs/2102.06143 · **Code/artifacts:** none found after search (see Source provenance)

**Preference level:** 3

**Pipeline status:** `partial` — the Baseline row produced comparable values; the five variational / weight-compression rows were not produced (protocol under-specified) and are retained as conditional evidence.

**Numerical agreement:** `does_not_agree` — the four produced Baseline values are ~4× below the paper (dev BLEU-4 4.66 vs 16.3; test 3.82 vs 16.3; dev ROUGE 19.35 vs 40.3; test 16.75 vs 40.7).

**Attempt date:** 2026-09-08

> **Attempt paused for re-evaluation.** Per the assignment instruction ("try the full
> reproduction, but if it requires a lot of debugging, stop and re-evaluate given the
> results"), the attempt was stopped after one round of fixes. All four models trained to
> a dev-BLEU-4 plateau and all 24 numbers were computed, but every row — including the
> well-specified Baseline — is ~4× below Table 1, and the Baseline plateaued rather than
> merely undertrained. Closing the gap needs iterative optimisation/architecture
> debugging (LR schedule, training length, the "no-attention" bridging, decoding) plus
> resolution of the unspecified variational hyperparameters.

## Scope and target contract

`what_to_reproduce` = "Table 1 on page 9". Table 1 ("Performance Metrics") reports Dev and
Test **BLEU-4** and **ROUGE** for six systems on the RWTH-PHOENIX-Weather 2014T
**Gloss2Text** task. 24 numbers → 24 targets (`table1-{system}-{split}-{metric}`).

**System semantics** (paper §4.2): `Baseline` = the Camgoz 2018 [10] Gloss2Text GRU
*without attention*; `GRU_repar` = Gaussian weight-posterior reparameterization on the GRU
non-gate weights only; `GRU_bp` = IBP stick-breaking prior on the non-gate weight utility
indicators only; `SB-GRU` = both (the proposed model); the `,wc` rows re-evaluate the same
trained model after post-hoc weight compression, so only four models are trained.

**Architecture** (paper §4): from [10] — 4 encoder + 4 decoder layers, 1000 units per
layer, GRU, no attention; the last encoder layer is replaced by the proposed recurrent
variant. Adam, lr 1e-5, batch 128, dropout 0.2, "until convergence". Implemented by the
authors in TensorFlow; **reimplemented here in PyTorch** (framework deviation; Eqs. 1–15
are framework-independent).

**Metrics:** the paper says only "BLUE and ROUGE" with no implementation citation. Scored
here with `sacrebleu` (BLEU-4, `tok:13a`, `lc`; signature
`nrefs:1|case:lc|eff:no|tok:13a|smooth:exp|version:2.4.3`) and `rouge-score` ROUGE-L F1;
exact upstream implementations unknown (guess).

**Ambiguity / resolution:** target *identity* is unambiguous. The *protocol* for the five
non-baseline rows is not — see the open gate `sbgru-protocol` and "Guesses and deviations".
Per the reproduction contract those rows are conditional evidence, not produced targets,
independent of numerical closeness.

## Source provenance

| Artifact | Canonical source | Pinned revision / SHA-256 | Role |
| --- | --- | --- | --- |
| Paper PDF | https://arxiv.org/pdf/2102.06143 | `5fe99055e474d9109e3d2b75e1b00c56380f82234cf859e6ad38c2a649436b5b` | Target table and protocol |
| Published code | — | — | **None exists** (see below) |
| Reference [10] code | https://github.com/neccam/nslt | commit `06951580b58f04b9cd64efcf61aeca36011031d3` | Architectural reference for the Baseline row; consulted, not run |

**Source search performed (2026-09-08), nothing found:** arXiv abstract + PDF (no code/data
link, body only says "We implement our model in TensorFlow [1]"); Springer chapter landing
page (paywalled, no code-availability statement); first author GitHub
`github.com/Partaourides` (`SERN`, `CUT_SDGs_Keyword-Mapping` — unrelated); co-author
A. Voskou (no public repo for this work); web / GitHub searches for "Stick-Breaking
Recurrent", "SB-GRU", the exact title. → preference level 3 (reimplement from the paper).

## Results

All numbers on a 0–100 scale. Differences are `reproduced − original`.

| Target | System | Split | Metric | Original | Reproduced | Diff | Status / reason |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| table1-baseline-dev-bleu4 | Baseline | dev | BLEU-4 | 16.3 | 4.66 | −11.64 | **produced** · does_not_agree |
| table1-baseline-dev-rouge | Baseline | dev | ROUGE-L | 40.3 | 19.35 | −20.95 | **produced** · does_not_agree |
| table1-baseline-test-bleu4 | Baseline | test | BLEU-4 | 16.3 | 3.82 | −12.48 | **produced** · does_not_agree |
| table1-baseline-test-rouge | Baseline | test | ROUGE-L | 40.7 | 16.75 | −23.95 | **produced** · does_not_agree |
| table1-gru-repar-dev-bleu4 | GRU_repar | dev | BLEU-4 | 16.7 | 3.68 | −13.02 | not_produced · protocol_ambiguous (conditional) |
| table1-gru-repar-dev-rouge | GRU_repar | dev | ROUGE-L | 41.1 | 11.66 | −29.44 | not_produced · protocol_ambiguous (conditional) |
| table1-gru-repar-test-bleu4 | GRU_repar | test | BLEU-4 | 17.0 | 4.06 | −12.94 | not_produced · protocol_ambiguous (conditional) |
| table1-gru-repar-test-rouge | GRU_repar | test | ROUGE-L | 41.5 | 11.29 | −30.21 | not_produced · protocol_ambiguous (conditional) |
| table1-gru-repar-wc-dev-bleu4 | GRU_repar,wc | dev | BLEU-4 | 16.2 | 3.68 | −12.52 | not_produced · protocol_ambiguous (conditional) |
| table1-gru-repar-wc-dev-rouge | GRU_repar,wc | dev | ROUGE-L | 40.6 | 11.66 | −28.94 | not_produced · protocol_ambiguous (conditional) |
| table1-gru-repar-wc-test-bleu4 | GRU_repar,wc | test | BLEU-4 | 16.7 | 4.06 | −12.64 | not_produced · protocol_ambiguous (conditional) |
| table1-gru-repar-wc-test-rouge | GRU_repar,wc | test | ROUGE-L | 40.7 | 11.29 | −29.41 | not_produced · protocol_ambiguous (conditional) |
| table1-gru-bp-dev-bleu4 | GRU_bp | dev | BLEU-4 | 18.4 | 3.35 | −15.05 | not_produced · protocol_ambiguous (conditional) |
| table1-gru-bp-dev-rouge | GRU_bp | dev | ROUGE-L | 43.9 | 12.51 | −31.39 | not_produced · protocol_ambiguous (conditional) |
| table1-gru-bp-test-bleu4 | GRU_bp | test | BLEU-4 | 17.0 | 3.66 | −13.34 | not_produced · protocol_ambiguous (conditional) |
| table1-gru-bp-test-rouge | GRU_bp | test | ROUGE-L | 43.1 | 11.92 | −31.18 | not_produced · protocol_ambiguous (conditional) |
| table1-sb-gru-dev-bleu4 | SB-GRU | dev | BLEU-4 | 17.9 | 3.76 | −14.14 | not_produced · protocol_ambiguous (conditional) |
| table1-sb-gru-dev-rouge | SB-GRU | dev | ROUGE-L | 43.0 | 12.09 | −30.91 | not_produced · protocol_ambiguous (conditional) |
| table1-sb-gru-test-bleu4 | SB-GRU | test | BLEU-4 | 18.1 | 3.99 | −14.11 | not_produced · protocol_ambiguous (conditional) |
| table1-sb-gru-test-rouge | SB-GRU | test | ROUGE-L | 43.5 | 11.46 | −32.04 | not_produced · protocol_ambiguous (conditional) |
| table1-sb-gru-wc-dev-bleu4 | SB-GRU,wc | dev | BLEU-4 | 17.7 | 3.76 | −13.94 | not_produced · protocol_ambiguous (conditional) |
| table1-sb-gru-wc-dev-rouge | SB-GRU,wc | dev | ROUGE-L | 43.0 | 12.09 | −30.91 | not_produced · protocol_ambiguous (conditional) |
| table1-sb-gru-wc-test-bleu4 | SB-GRU,wc | test | BLEU-4 | 17.8 | 3.99 | −13.81 | not_produced · protocol_ambiguous (conditional) |
| table1-sb-gru-wc-test-rouge | SB-GRU,wc | test | ROUGE-L | 42.8 | 11.46 | −31.34 | not_produced · protocol_ambiguous (conditional) |

No original scores were copied from earlier work (queue `copied_scores = no`; the paper
re-evaluates the [10] no-attention baseline itself). The Baseline reproduction does **not**
reproduce [10]'s reported ~16 BLEU-4 either.

**Diagnosis.**
- *Baseline / `plain`:* dev BLEU-4 rose from ~0 to ~4.3 by epoch 70 then stayed flat
  (4.0–4.7) through epoch 134; ROUGE ~19. Decoded output is fluent weather-domain German
  only loosely tied to the gloss (e.g. gloss `MITTWOCH REGEN KOENNEN NORDWEST … WIND` →
  "am samstag regnet es im süden und süden teilweise freundlich"). This is a functioning
  but content-weak seq2seq that has *converged to a plateau* far below the paper — not an
  interrupted run. Likely causes: the no-attention single-vector bottleneck as implemented,
  absence of an LR schedule, or a much longer training horizon than "until convergence"
  reached here. Unresolved.
- *`repar`:* Gaussian-posterior KL is well-behaved (~0.09 after the σ-init fix); still
  plateaus at ~3.7 BLEU-4, tracking the baseline problem.
- *`bp` / `sb`:* with α=1 the IBP stick-breaking KL term is ~3170 after dividing by N
  (vs NLL ≈ 5.5), so it dominates the loss ~500×; dev BLEU-4 freezes (identical values for
  60+ epochs) — the variational last encoder layer stops learning. This is the
  `sbgru-protocol` gate.
- *`,wc` rows:* 2-bit post-hoc quantisation of one weight matrix moved the scores by
  ≤0.01 — unsurprising for an already-degenerate layer.

## How to repeat this

```bash
# 0. Modal auth (once): modal setup  -> workspace repro-sign
S=.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh
APP=papers/partaourides-2021-variational-bayesian-slt/modal_app.py

$S run $APP::check_env
$S run $APP::preflight                                   # tiny 4-mode end-to-end check on real data

for m in plain repar bp sb; do
  $S run --detach $APP::train --mode "$m" --max-epochs 250   # writes /outputs/$m/{results.json,best.pt}
done

$S run --detach $APP::evaluate --mode repar --quantize-bits 2   # -> /outputs/repar/eval_q2/eval.json
$S run --detach $APP::evaluate --mode sb    --quantize-bits 2
```

Results land on the Modal Volume `partaourides-2021-results` (`{mode}/results.json`,
`{mode}/best.pt`, `{mode}/run_meta.json`). Local smoke test (no GPU):
`python sbgru_slt.py preflight --data-dir DIR` where `DIR` holds the three
`PHOENIX-2014-T.{train,dev,test}.corpus.csv` files. Training resumes from scratch (no
checkpoint-resume implemented); each run is independent and idempotent per output dir.

## Data provenance and permissions

| Dataset | Version/subset/splits | Source and access date | License / cloud-use basis | Path in Volume `datasets` | Counts | Deviations |
| --- | --- | --- | --- | --- | --- | --- |
| RWTH-PHOENIX-Weather 2014T | Gloss2Text: `orth` → `translation`, text only (no video/features) | https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/ (2026-09-08) | Queue record: CC BY-SA 3.0; version page links CC BY-NC-SA 3.0. Text-only non-commercial research on project cloud — permitted under either. | `rwth-phoenix-2014-t/annotations/PHOENIX-2014-T.{train,dev,test}.corpus.csv` | train 7096 / dev 519 / test 642 | none for data identity |

Split counts exactly match the paper (§4.1, total 8257). German training vocabulary = 2887
types / 1077 singletons — **exact** match to the paper. Gloss training vocabulary =
1085 types / 355 singletons on raw whitespace split vs the paper's 1066 / 337; the ~19-type
gap is attributed to an unspecified gloss-cleaning / frequency-cutoff step (guess).

File SHA-256: train `cc3dc2461f0a222b92f3927c24ac21c1467f3e5428b406ee7fe40bca1b0b8d44`,
dev `1085141d0ed6f28c6de6196a271b72c07366ed3fe5470c9717bd44640737f00b`,
test `632b19c9a87fb9c98b0821e04861750565348bce20a069c6dea1bba5bda27879`.

## Environment and patches

Base image `nvcr.io/nvidia/pytorch:26.04-py3` (repo `Dockerfile`) + `sacrebleu==2.4.3`,
`rouge-score==0.1.2`. Single **NVIDIA A10G**, driver 580.95.05, CUDA minor-version
compatibility mode (image CUDA 13.2 on driver CUDA 13.0 — benign). HF cache mounted at
`/cache/huggingface` with `HF_HOME` / `HF_HUB_CACHE` set (nothing is fetched from the Hub).
No upstream code is patched — there is no upstream code; the reimplementation is
`sbgru_slt.py`.

## Execution evidence

Modal profile `repro-sign`, app `partaourides-2021-variational-bayesian-slt`, results
Volume `partaourides-2021-results`. All six runs exited 0.

| Run | App ID | function-call | Start → End (UTC) | GPU | Terminal |
| --- | --- | --- | --- | --- | --- |
| train-plain | `ap-0TID6xKjXXNhbwYMP1njUJ` | `fc-01M20QAAJ3CRB2QTCZB4270W6V` | 14:39:28 → 14:55:35 | A10G | succeeded / completed (early-stop plateau, best epoch 94/134) |
| train-repar | `ap-ZpAxljsKew8rsKOAT5Abhv` | `fc-01M20QAD1JBS07PTC7KNM8KN37` | 14:39:10 → 14:57:21 | A10G | succeeded / completed (best epoch 116/156) |
| train-bp | `ap-SRVwv8HXPaM0xtwKlrEaFN` | `fc-01M20QAG46B32E7BBZBEF44W4P` | 14:39:11 → 14:49:19 | A10G | succeeded / completed (best epoch 42/82; KL-frozen) |
| train-sb | `ap-KcpysRyUJ9PAbuxskpyxNY` | `fc-01M20QAK0XJTPWCXEXWABMM124` | 14:39:13 → 14:55:04 | A10G | succeeded / completed (best epoch 94/134; KL-frozen) |
| eval-repar-wc | `ap-DTPMdzded597fvvZWMOCu9` | `fc-01M20S2E08Y08DAZZMR358QH7P` | 15:09:52 → 15:10:07 | A10G | succeeded / completed (2-bit quant) |
| eval-sb-wc | `ap-0uFwmdSWXHnYOJx5K8s32Y` | `fc-01M20S2DYA246DF60KMF34FWDR` | 15:09:45 → 15:10:00 | A10G | succeeded / completed (2-bit quant) |

Ceilings declared before launch: 24 h wall / 12 GPU-h / 15 CHF per training group, 3
attempts; none approached (longest run 18 min, ~0.3 GPU-h). Raw metrics: artifacts
`plain-results`, `repar-results`, `bp-results`, `sb-results`, `repar-wc-eval`, `sb-wc-eval`
(`modal://partaourides-2021-results/...`, SHA-256 in `reproduction.json`).

## Guesses and deviations

| Detail | Paper says | This attempt used | Effect |
| --- | --- | --- | --- |
| Framework | TensorFlow | PyTorch reimplementation | Should be behaviour-neutral; unverifiable without author code |
| Embedding dim | — | 1000 | Model capacity |
| Encoder directionality | — | unidirectional 4-layer GRU | Content transfer capacity |
| "no attention" bridging | "without attention" | decoder layers initialised from final encoder-layer states, no other connection | **Likely material** to the baseline gap |
| Convergence criterion | "until convergence" | early stop on dev BLEU-4, patience 20 evals (≤250 epochs) | Baseline plateaued ~epoch 70, so longer training alone is unlikely to close the gap, but not ruled out |
| LR schedule | "ADAM, lr 0.00001" | constant 1e-5, no warmup/decay | Possible cause of the plateau |
| Checkpoint selection | — | best dev BLEU-4 | Standard |
| Beam width / length penalty | — | beam 3, lp 0 | Minor; greedy gives near-identical output |
| BLEU / ROUGE implementation | "BLUE and ROUGE" | sacrebleu 13a/lc; rouge-score ROUGE-L F1 | Scale/offset vs the paper's unknown scripts |
| Gloss preprocessing | — | raw whitespace split | vocab 1085 vs 1066 |
| IBP innovation α | — | 1.0 | **Material** — drives the KL that freezes `bp`/`sb` |
| Cutoff threshold τ | mentioned | 0.5 | Inference sparsity |
| Gumbel-Softmax λ schedule | "annealed similar to [19]" | 1.0 → 0.5 linear over 20k steps | Relaxation tightness |
| MC samples in ELBO | "draw MC samples" | 1 | Gradient variance |
| KL warm-up | — | linear 0→1 over 2000 steps | Did not prevent KL domination |
| Kumaraswamy a,b init | — | a=10, b=1 (sticks near 1 at init) | Changed from a=b=1 after preflight underflow of π_k |
| Gaussian σ init | — | σ≈1 (`wy_sigma_init=1.0`) | **Fix** — the initial −5 rho gave KL_w≈140k and crushed `repar`; corrected to the Bayes-by-Backprop convention |
| KL(q(z)‖p(z)) estimator | MC (Eq. 14) | analytic Bernoulli KL | Deviation for numerical stability; same quantity in expectation |
| KL(q(W)‖p(W)) estimator | MC (Eq. 15) | MC (closed-form available via flag) | Faithful |
| Weight-compression rule | "unit round off", "×16", "bit precision 1" | uniform per-matrix 2-bit quantisation of the last encoder layer's effective W_y | **Material & under-specified**; `,wc` rows conditional |
| Seed policy | — | single seed = 1 | Paper reports one value per cell |

## Attempts, failures, and dead ends

1. **Local preflight (CPU).** `plain` end-to-end OK. `repar`/`bp`/`sb`: KL ≈ 140k–480k;
   `repar` traced to `W_y_rho` init −5 (σ≈0.007) giving a huge `−2 log σ` KL. *Fix kept:*
   `wy_sigma_init = 1.0`. `bp`/`sb` KL still large → traced to π_k underflow (Kumaraswamy
   a=b=1 over 1000 sticks). *Fixes kept:* `kuma_a_init = 10`, analytic Bernoulli KL_z.
2. **Modal preflight (A10G, real PHOENIX data).** All four modes: data checksum verified,
   train steps + checkpoint save/reload + real eval + metrics + 2-bit quant eval all pass.
3. **Full runs (A10G, ≤250 epochs).** All four exit 0 and reach evaluation. `plain` dev
   BLEU-4 plateaus at ~4.4 from epoch ~70; `repar` ~3.7; `bp`/`sb` frozen at ~3.4–3.7 with
   KL/N ≈ 3170 ≫ NLL. Decoded samples are fluent but low-content.
4. **Weight-compression evals.** 2-bit quant of the last encoder candidate weight changes
   scores by ≤0.01.
5. **Stopped here** per the assignment's re-evaluation instruction. Not attempted: LR
   schedule / much longer training / alternative no-attention bridging for the baseline;
   an α / τ / λ / KL-weight sweep for the variational layer (would be score-seeking against
   an unspecified protocol without author input).

## Candidate flags, ethics, and human evaluation

Queue flags: `includes_human_evaluation = no`, `potential_ethical_concerns = no`,
`copied_scores = no`. Gloss2Text uses only released text annotations — no participant
interaction, no sensitive data. Queue `comments` note "PHOENIX-2014-T … subset gloss2text";
confirmed this is the standard 2014T release and the paper's own dataset (§4.1). No ethics
gate applies.

## Author and team contact

None. (Contacting the authors for the missing hyperparameters is a possible next step but is
gated on an independent attempt first — this attempt — and on a decision to continue.)
