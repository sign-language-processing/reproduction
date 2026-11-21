# OLMD reproduction

**Paper ID:** `fa37313bf1a4558af1462085e70b884b1418fd50`

**Citation:** Yiheng Yu, Sheng Liu, Yuan Feng, Min Xu, Zhelun Jin, and Xuhua Yang. 2025.
OLMD: Orientation-aware Long-term Motion Decoupling for Continuous Sign Language
Recognition. In *Proceedings of the Thirty-Ninth AAAI Conference on Artificial
Intelligence (AAAI-25)*, 39(9), 9707–9715. https://doi.org/10.1609/aaai.v39i9.33052

**Paper:** https://arxiv.org/abs/2503.08205 · **Code/artifacts:** none found after a
documented search (see [Source provenance](#source-provenance))

**Preference level:** 3

Level 3 means reimplementation from the paper: no published code exists.

**Pipeline status:** `complete` — all four in-scope Table 1 targets are produced from
completed 70-epoch full runs on Modal: PHOENIX14 on A100-40GB, PHOENIX14-T on H100.

**Numerical agreement:** `does_not_agree` — official sclite chain, beam-10:
PHOENIX14 **35.0 dev / 34.2 test** against published **17.1 / 17.2**, PHOENIX14-T
**34.5 dev / 32.6 test** against published **17.1 / 18.4**.

**Attempt dates:** 2026-09-08 → 2026-09-10 (PHOENIX14); 2026-09-22 → 2026-09-24 (PHOENIX14-T)

> **Status: both datasets reproduced end to end; the numbers disagree with the paper.**
> The full stated protocol ran to completion on each (70 epochs, batch 2, best-dev
> checkpoint selection) and both were evaluated with the official PHOENIX sclite chain.
> Every reproduced WER is roughly twice its published counterpart. On PHOENIX14, of the
> raw 34-point gap first observed, ~16.5 points were traced to this reproduction's own
> evaluation methodology and corrected.
>
> **The two datasets do not share one model variant.** After the PHOENIX14 run, a
> figure audit found two wiring deviations. PHOENIX14-T was trained with the corrected
> wiring and the lineage padding. The reported PHOENIX14 values still carry both
> deviations, because no corrected PHOENIX14 run is part of this report.
>
> **PHOENIX14-T shows that the residual gap is not evaluation error.** Its annotations
> carry no PHOENIX markers, so the asymmetric-stripping bug that cost ~13 WER on PHOENIX14
> cannot arise. Its internal metric also agrees with the official chain to within 0.13
> WER, yet it lands at a similar distance from its published value.
>
> Both datasets also score worse than the paper's own no-modules ablation baseline (18.9
> dev), so the shortfall is not about the marginal value of OLMD's modules. The residual
> **~14-18 points remain unexplained**, with the unpublished MP-Block the leading
> candidate. A reviewer decides the scientific conclusion; every variant, guess, and dead
> end is recorded.
>
> Only Modal experiments are reported. Runs on other infrastructure are excluded from
> this report.

## Agent attribution

| Agent | Model | Application | Contribution |
| --- | --- | --- | --- |
| `claude-opus-5-claude-code` | Claude Opus 5 (`claude-opus-5`) | Claude Code 2.1.247, desktop app | Implementation, data gate, preflights, cache, first PHOENIX14 run and CTC fix (2026-09-08 → 09-09); all PHOENIX14-T work (2026-09-22 → 09-24) |
| `claude-fable-5-claude-code` | Claude Fable 5 (`claude-fable-5`) | Claude Code 2.1.247, desktop app | PHOENIX14 attempt 3, rescoring and sclite, figure audit, smoke test of the corrected wiring (2026-09-09 → 09-10) |
| `claude-opus-5-5-report-maintenance` | Claude Opus 5.5 (`claude-opus-5-5`) | Claude Code 2.1.309 | Report maintenance only (2026-11-04): this attribution, and restricting the report to Modal experiments; ran nothing |

Each run's `agent_ids` names the model that issued its launch command. The evidence is
the `model` field on the launching assistant message in the local Claude Code session
transcripts (session IDs in `reproduction.json.agents`). The transcripts were not
published. The `Co-Authored-By` trailers on this paper's commits agree with that
evidence: 38 Opus 5 commits and 14 Fable 5 commits. One commit, `97ddc1d`, has no
trailer. One conversation switched between Fable 5 and Opus 5, so attribution is per
launch, not per session.

## Scope and target contract

The queue export's `what_to_reproduce` is **"Table 1"**, and on 2026-09-08 the assignee
confirmed **Table 1 only**. Table 1 covers **PHOENIX14 and PHOENIX14-T**; CSL-Daily
appears only in **Table 2** and is **out of scope**, despite the export listing it among
the paper's datasets. (An earlier instruction the same day had briefly widened scope to
Table 2; that is superseded.)

Table 2's rows were removed from the target ledger rather than carried as not-produced
targets. Every controlled not-produced reason code asserts a *blocker*, and none would be
truthful here — Table 2 is simply not requested, and CSL-Daily presented no data or
compute obstacle. CSL-Daily's verified data-gate evidence is retained under
`reproduction.json.datasets`, because it was genuinely obtained and the corpus defect
found there is worth preserving.

### In-scope targets — OLMD's own rows, extracted verbatim

| Target ID | Table | Dataset | Split | Published WER |
| --- | --- | --- | --- | ---: |
| `table1-p14-dev-wer` | 1 | PHOENIX14 | dev | 17.1 |
| `table1-p14-test-wer` | 1 | PHOENIX14 | test | 17.2 |
| `table1-p14t-dev-wer` | 1 | PHOENIX14-T | dev | 17.1 |
| `table1-p14t-test-wer` | 1 | PHOENIX14-T | test | 18.4 |

Table 1 also reports del/ins for PHOENIX14 (4.8/2.4 dev, 4.7/2.3 test) but not for
PHOENIX14-T. The assignment's metric is WER, so del/ins is recorded as context, not as a
target.

### Out-of-scope targets — copied comparison rows

The other 60 cells (15 prior methods × 4 dataset/split columns) are in the ledger with
`in_scope: false`, `copied_baseline: true`, and a terminal `not_produced` result with
reason code `copied_baseline`. They account for Table 1's comparison without re-executing
previously published systems.

**Basis for the copied determination:** the paper carries no "results are taken from the
original papers" disclaimer, but the del/ins column is empty exactly for those methods
whose own papers did not report del/ins (CoSign, C2SLR\*, TwoStream-SLR\*,
CNN+LSTM+HMM\*). Had the OLMD authors re-evaluated those systems they could have computed
del/ins themselves. This indicates transcription. The export's `copied_scores: "yes"`
flag is consistent with that reading.

### Metric

WER = (#sub + #ins + #del) / #ref, the paper's Equation 15, lower is better, reported as a
corpus-level percentage, one value per cell. The paper does **not** name its scoring
implementation; pinning it to each dataset's official evaluation script is an open guess.

## Source provenance

| Artifact | Canonical source | Pinned revision / SHA-256 | Role |
| --- | --- | --- | --- |
| Paper PDF | https://arxiv.org/pdf/2503.08205v1 | `8063662fcaae5d7374451721b94e1e03c706573ebd90378759469587a8963d70` | Target values and protocol |
| Paper HTML | https://arxiv.org/html/2503.08205v1 | arXiv:2503.08205v1 | Verbatim Table 1 / Table 2 extraction |
| Version of record | https://ojs.aaai.org/index.php/AAAI/article/view/33052 | DOI 10.1609/aaai.v39i9.33052 | AAAI camera-ready; checked for supplements |
| Published code | — | — | **None found** |

### Source-search checklist (completed 2026-09-08)

| Location checked | Result |
| --- | --- |
| arXiv abstract page | No code-availability statement, no repository link |
| arXiv HTML full text (downloaded, grepped) | Only `github.com` strings are arXiv's own LaTeXML/`html_feedback` boilerplate. No "code is available", no footnote URL, no gitee link |
| AAAI-25 proceedings entry | Downloads are the PDF and a poster only; no supplementary code archive |
| GitHub code search: "OLMD sign language", "OLMD CSLR" | Zero matching repositories |
| GitHub search: bare "OLMD" | 22 repositories, all unrelated (LMDB wrapper, markdown editor, profile configs) |
| GitHub author accounts (first author: `yuyiheng-eu`) | Account exists and publishes code for a *different* paper (FGDM), but hosts no OLMD repository. These authors do release code for other work, so the absence appears deliberate |
| Papers With Code | 302-redirects to `huggingface.co/papers/2503.08205`, which lists no associated repository |
| Author / lab homepages (Zhejiang University of Technology) | None located via search |
| Semantic Scholar `fa37313bf…` | **NOT VERIFIED** — Graph API returned HTTP 429 on five attempts; web page returned HTTP 202 with an empty body (bot protection). External code links could not be read. Low residual risk because PWC/HuggingFace, its usual code-link source, lists none |

**Conclusion:** no public code implementation of OLMD exists, official or unofficial.
Preference levels 1 and 2 require an executable published artifact, so this reproduction
is **level 3 — reimplementation from the paper**.

## Results

| Target ID | Paper location | Dataset/split | Original | Reproduced | Difference | Status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `table1-p14-dev-wer` | Table 1, OLMD row | PHOENIX14 / dev | 17.1 | **35.0** | +17.9 | produced (`sclite-phoenix14-002`) |
| `table1-p14-test-wer` | Table 1, OLMD row | PHOENIX14 / test | 17.2 | **34.2** | +17.0 | produced (`sclite-phoenix14-002`) |
| `table1-p14t-dev-wer` | Table 1, OLMD row | PHOENIX14-T / dev | 17.1 | **34.5** | +17.4 | produced (`sclite-phoenix14t-wiring-fix-001`) |
| `table1-p14t-test-wer` | Table 1, OLMD row | PHOENIX14-T / test | 18.4 | **32.6** | +14.2 | produced (`sclite-phoenix14t-wiring-fix-001`) |
| 60 copied rows | Table 1, prior methods | both datasets | see `reproduction.json` | n/a | n/a | `copied_baseline` |

Difference is reproduced minus original, in WER points. WER is lower-is-better, so a
positive difference means a higher error rate than published.

The reproduced values are the **official PHOENIX evaluation chain** (RWTH `preprocess.sh`
+ `mergectmstm.py` + SCTK v2.4.12 `sclite`, all sha256-pinned) applied to **beam-10**
(no LM) decoding of the best-dev checkpoint of `train-phoenix14-seed0-attempt3` (epoch 57
of 70, Modal A100-40GB). This is the evaluation the field's published numbers use, so it
is the comparable value. Error breakdown at that setting: dev 18.3 sub / 12.3 del / 4.4
ins; test 18.1 / 11.9 / 4.2 (paper reports 4.8 del / 2.4 ins dev, 4.7 / 2.3 test).

### Wiring differs between the two datasets

| Dataset | Training run | Temporal head and LMA routing | Collate |
| --- | --- | --- | --- |
| PHOENIX14 | `train-phoenix14-seed0-attempt3` | as implemented (deviations `head-wiring-vs-figure2`, `lma-in-trunk-vs-figures`) | uniform batch lengths |
| PHOENIX14-T | `train-phoenix14t-wiring-fix-002` | corrected to the paper's figures (commit `a33b3e3`) | lineage padding |

The figure audit happened after the PHOENIX14 run. No PHOENIX14 run with the corrected
wiring is part of this report, so the two datasets' values are not from one model
variant. Both runs keep the unshared temporal branches (deviation
`unshared-branch-weights`).

### How the reported number depends on scoring — measured, same checkpoint

| Scoring | Decoding | Dev WER | Test WER |
| --- | --- | ---: | ---: |
| Asymmetric marker stripping (this repo's original scorer — **a bug**) | greedy | 51.47 | 51.98 |
| Symmetric Levenshtein, markers kept (as official STM refs) | greedy | 40.45 | 40.13 |
| Symmetric Levenshtein, markers stripped both sides | beam-10 | 38.19 | 37.43 |
| **Official sclite chain** | **beam-10** | **35.00** | **34.20** |
| Published (Table 1) | unstated | 17.1 | 17.2 |

Two findings from the diagnosis runs (`rescore-phoenix14-001`, `sclite-phoenix14-002`):

- **~16.5 of the raw 34-point gap was this reproduction's own evaluation methodology.**
  The training-time scorer stripped `__ON__`/`__OFF__`-class markers from references but
  not hypotheses (~13 points — every emitted marker counted as an insertion), and did not
  apply the official RWTH normalizations (letter merging, compound joins, `-PLUSPLUS`
  removal, repeat dedup; ~3.3 points). Both corrected; the scorer bug is fixed for future
  runs in `modal_app.py`.
- **Decoding is immaterial (~0.3 points).** Beam width 10, the lineage's setting, improves
  on greedy by 0.25 dev / 0.23 test. The paper's unstated decoding cannot explain the gap.

### The remaining ~16-point gap is unexplained

The PHOENIX14 training run was healthy: loss fell monotonically, both LR cuts (epochs
25/40) produced visible improvements, and errors are dominated by substitutions and
deletions. Context that sharpens the diagnosis: the paper's own no-modules ablation
baseline (Table 5, "None") reaches 18.9 dev — our 35.0 is far worse than even that, so the
shortfall points at the **inferred parts of the recipe**, not at the marginal value of
OLMD's novel modules.

**Post-run figure audit (2026-09-10) found two wiring deviations.** A full-resolution
inspection of the paper's figure assets (the prose never states either detail, and both
were adopted under then-open guesses before training) found:

1. **Temporal head** (deviation `head-wiring-vs-figure2`): Figure 2 feeds the trunk
   features to one 1D-CNN and the cross-stage output X_CS to its *own* 1D-CNN, sums the
   two outputs, and runs the BiLSTM on the sum — so one local CTC loss directly
   supervises the coupling path. The run instead computed `head(trunk + X_CS)` with two
   identical parallel branches over the same input.
2. **LMA routing** (deviation `lma-in-trunk-vs-figures`): Figures 2 and 3 draw LMA on a
   side branch — the trunk continues from the *raw* stage output plus the α/β-scaled
   coupling, so with α=β=0 at init the ImageNet-pretrained trunk starts undisturbed. The
   run instead routed LMA's residual output (whose expand convolution is not
   zero-initialized) into the trunk.

Both are training-side candidate contributors to the residual gap. PHOENIX14-T was
trained with both corrected; PHOENIX14 was not retrained with them in this report. The
same audit *positively verified* the rest of the
architecture against the figures: FFN residual, cascade order and content (DS of the
previous post-FFN output, fused before the MP-Block), α/β mapping, HMP/VMP orientation
assignment, and the LMA difference/concat structure. It also confirmed Figure 2's "×N"
annotations mark the three stages, not repeated 1D-CNNs. Pipeline-side, the audit found
only minor items: dead lineage-padding helpers in `data.py` (`padded_length`, unused —
consistent with the recorded `uniform-batch-lengths` deviation), and
`zero_infinity=True` on the CTC loss where the lineage uses `False` (deviation
`ctc-zero-infinity`, negligible). The image path is clean: PIL/RGB end to end, cache
JPEG roundtrip verified, per-clip crop/flip, lineage [-1, 1] scaling (recorded guess).

**The MP-Block fidelity caveat is central.** The MP-Block, the core of the paper's
headline HMP/VMP contribution, is inferred: the paper defers it to supplementary material
that does not exist publicly (arXiv has no ancillary files; the AAAI entry ships PDF and
poster only). These numbers are therefore conditional evidence about an inferred
architecture, and a reviewer should weigh that before treating the disagreement as a
statement about the paper. Other open training-side guesses (cross-stage injection point,
uniform batch lengths, unstated hyperparameters) are recorded in
`reproduction.json.guesses` with their expected leverage.

## Preflight

`preflight-phoenix14-002` (Modal app `ap-29IaSd3vio60Lv8PSxDRha`, A100-SXM4-40GB, exit 0,
124 s) met every AGENTS.md preflight requirement on PHOENIX14 with TLP active:

| Requirement | Result |
| --- | --- |
| Real data through the real loader | 24 train + 8 dev utterances, real PNG frames, real preprocessing |
| Intended weights | ImageNet-pretrained ResNet34 loaded |
| Real optimizer steps | 12 steps, Adam lr 0.001 wd 0.001, batch size 2 |
| Checkpoint save **and reload** | 1.34 GB saved, reloaded, **every tensor verified equal** |
| Real evaluation on a subset | 8 dev utterances decoded and scored |
| Parsed metric + terminal status | corpus WER emitted, exit code 0 |

CTC loss fell **57.72 → 20.89** and total loss **71.42 → 25.90** across 12 steps.

### The preflight WER is not a result

Both preflights scored **WER 100.0** with **94 deletions, 0 insertions, 0 substitutions** —
the model emitted *empty* hypotheses. That is the expected CTC early-training regime after
12 steps from initialization: the blank class dominates before alignments sharpen, which is
consistent with the falling loss. It demonstrates that the metric path executes end to end
and **must never be compared with the paper's 17.1/17.2**.

### The first preflight's estimate was wrong; the second fixed it

`preflight-phoenix14-001` reported `hours_70_epochs: 21.4`. That figure is void: it counted
training only, and its timer wrapped just the optimizer step, so frame loading fell outside
it. `preflight-phoenix14-002` times loading and GPU work separately and includes the
per-epoch dev pass.

### TLP's auxiliary regularizer behaves correctly

`preflight-phoenix14-002` records per-step loss components. CTC falls monotonically
57.72 → 20.89, while TLP's lifting regularizer drifts down from ~1650 to ~1300 with noise —
the expected behaviour as the predictor learns to predict odd samples from even ones. It
contributes ~1.8 to the total at TLP's own 0.001 weight, so it is active without dominating.

## Compute requirement (measured, with TLP active)

Measured on A100-40GB with the full 111.8 M-parameter model, on real ~180-frame clips:

| Cost | Measured |
| --- | ---: |
| GPU step (batch 2, fwd+bwd+update) | **0.422 s** |
| Serial batch load (2 clips of PNGs) | **5.326 s** — 12.6× the GPU work |
| Cached batch load (JPEG `.npz`) | 0.534 s |
| Peak memory | 24.47 GiB |

**TLP costs only ~9% more GPU time, not 2×.** The A100 step went 0.388 s → 0.422 s. An
earlier provisional range of 70–79 GPU-hours scaled a doubling seen on an A10; that
reflected the A10's weaker throughput, not the model, and is superseded.

### Prefetching hides 95.7% of the loading latency (measured)

`loader-benchmark-001` measured the JPEG cache behind a DataLoader with 8 workers,
`prefetch_factor=4` and pinned memory:

| | Measured |
| --- | ---: |
| Median wall time per step | **0.4073 s** |
| Median GPU step | 0.390 s |
| **Loading still exposed** | **17.3 ms (4.3%)** |

Before the cache and DataLoader, batch loading cost **5.326 s fully exposed**. The first
step takes 4.98 s (worker startup, cuDNN autotuning) and steady state arrives by the third.

For 70 epochs, batch size 2, with best-dev checkpoint selection (a dev pass each epoch):

| Stage of the estimate | PHOENIX14 | PHOENIX14-T |
| --- | ---: | ---: |
| Serial PNG loading (as first written) | ~317 GPU-h | — |
| With the JPEG cache alone | ~57.8 GPU-h | ~72.3 GPU-h |
| **Cache + prefetching (measured)** | **24.5 GPU-h** ($52, 1.0 d) | **30.1 GPU-h** ($63, 1.3 d) |

Plus a one-time, CPU-only, idempotent, resumable cache build measured at **8.5 h / $6.42 /
9.7 GiB** for PHOENIX14 (10.3 h / $7.75 / 11.8 GiB for PHOENIX14-T).

**The estimate fell 13× through data plumbing alone** — no change to the model, protocol,
epochs or augmentation.

Two caveats on the measurement: the benchmark landed on an A100-**80**GB while
`preflight-phoenix14-002` measured 0.422 s on an A100-**40**GB (close, and the 0.390 s GPU
step is consistent, but not identical hardware); and with only 20 batches cached the 40-step
benchmark cycled one epoch, so later steps re-read page-cached clips. Real full-dataset
throughput could be slightly worse. The full run should pin the GPU type.

### GPU choice, measured on both cards

| | A100-40GB | **H100** |
| --- | ---: | ---: |
| GPU step | 0.390 s | **0.2347 s** |
| Wall step | 0.4073 s | **0.2384 s** |
| Loading exposed | 17.4 ms (4.3%) | **3.7 ms (1.5%)** |
| PHOENIX14 | 24.5 GPU-h, $52, 1.0 d | **14.4 GPU-h, $57, 0.6 d** |
| PHOENIX14-T | 30.1 GPU-h, $63 | 17.7 GPU-h, $70 |
| Clears 24 h gate? | **no** (by ~2%) | **yes** |

The H100's measured speedup is **1.66×** on GPU step time. Both my own 1.2–1.5× estimate
and the nominal 1.9× were wrong; measuring settled it. Requested as `gpu="H100!"` so Modal
could not substitute an H200 — the run reported `NVIDIA H100 80GB HBM3`.

Cost is nearly indifferent between the cards (~$5) because Modal's hourly rates scale with
throughput. The H100's real advantage is wall-clock and **gate compliance**.

Memory rules out the cheap cards outright: measured peak is 24.47 GiB at the paper's batch
size 2, so T4 (16 GiB), L4 (24 GiB) and A10 (24 GiB) cannot run this configuration without
a batch-size or precision change — a protocol deviation, not an optimization.

### Each dataset clears the gate; the pair does not

PHOENIX14 at 14.4 and PHOENIX14-T at 17.7 GPU-hours each fall inside the 24-hour threshold,
but together they are **32.1 GPU-hours**. They must therefore run as **separate staged
jobs**, which is what the assignee's PHOENIX14-first instruction already implies.

### Status of the gate — closed by explicit authorization

The assignee authorized the PHOENIX14 launch on 2026-09-08 (gate
`compute-full-run-table1`, commit `56f0051`) and re-authorized the post-fix launch
directly on 2026-09-09. The run executed on **A100-40GB** (not H100): the training
function checkpoints every epoch and resumes across invocations, so the 24-hour function
timeout was not a constraint, and the attempt-3 invocation (66 epochs, ~22.1 GPU-hours)
fit inside it. Total training spend across all attempts (collapsed attempt 1, fix probe,
completed attempt 3, diagnostics) is roughly 33 GPU-hours / ~USD 75 — about 0.35% of the
study budget.

### The full run as executed

1. **`train-phoenix14-seed0-attempt1`** (2026-09-08) — stopped after 29 epochs, dev WER
   pinned at 100.0: an agent-introduced loss bug (`nn.CTCLoss(reduction="mean")`
   length-normalizes per sample, letting the γ₂=25 distillation term dominate; all three
   branches collapsed to blank). Terminal `stopped/invalid_run`; retained as evidence.
2. **`train-phoenix14-fixprobe-002`** (2026-09-09) — 4-epoch probe of the corrected
   reduction (`none` + batch `.mean()`, verified against the lineage). CTC fell 182.6 →
   141.3, dev WER left the collapse point. Terminal `succeeded`, `retry_decision: resume`.
3. **`train-phoenix14-seed0-attempt3`** (2026-09-09 → 09-10) — resumed the probe's
   epoch-3 checkpoint (identical seed, config, optimizer/scheduler state; the probe was
   the first 4 epochs of the full protocol) and trained through epoch 69. Best dev WER
   51.47 at epoch 57 under the then-current (asymmetric) scorer; terminal `succeeded`.
4. **`rescore-phoenix14-001` / `sclite-phoenix14-001..002`** (2026-09-10) — inference-side
   diagnosis; see [Results](#results). sclite attempt 1 failed on a missing
   `make install` (deterministic, one scoped fix, retried).

## How to repeat this

```bash
git clone https://github.com/sign-language-processing/reproduction.git
cd reproduction && git checkout repro/yu-2025-olmd
```

Modal authentication (interactive, human-run — the wrapper refuses to do this itself):

```bash
python3.9 -m pip install --user modal
```

```bash
modal setup
```

Select workspace **`repro-sign`**, then verify through the wrapper only:

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh
```

Validate the record:

```bash
python3.9 .agents/skills/reproduce-paper/scripts/validate_reproduction.py papers/yu-2025-olmd
```

All entry points run through the wrapper and are idempotent:

```bash
# One-time JPEG frame cache (CPU fan-out; resumable)
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/yu-2025-olmd/modal_app.py --action build-cache-sharded
```

```bash
# Representative preflight (AGENTS.md): real data, steps, checkpoint reload, tiny eval
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/yu-2025-olmd/modal_app.py --action preflight
```

```bash
# Full 70-epoch training (resumes from latest.pt if present; pass --limit N for a probe)
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/yu-2025-olmd/modal_app.py --action train
```

```bash
# Re-decode the best-dev checkpoint (greedy + beam-10) under all scoring variants
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/yu-2025-olmd/modal_app.py --action rescore
```

```bash
# Official sclite chain (pinned RWTH preprocessing + SCTK v2.4.12) on the rescore CTMs
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/yu-2025-olmd/modal_app.py --action sclite
```

Training state, per-epoch history, final results, hypotheses, and sclite reports live on
the Modal Volume `olmd-yu-2025-results` under `/train/phoenix14/` and
`/rescore/phoenix14/`, hashed in `reproduction.json.artifacts`.

## PHOENIX14-T experiment (stage two)

Branch `repro/yu-2025-olmd-phoenix14t` runs **the same experiment as PHOENIX14, on
PHOENIX14-T, on the Modal path**, with the **figure-conformant wiring** (`olmd.py`
restored from `a33b3e3`). This is not the wiring behind the reported PHOENIX14 values;
see [Wiring differs between the two datasets](#wiring-differs-between-the-two-datasets).

Five changes were needed to reach a dataset the pipeline had never executed:

1. **`_load_video` was broken.** It called `simple_video_utils.read_video`, which does
   not exist. The function was written alongside the PNG path and its video branch had
   never run, so the error surfaced only when a video dataset was first used. The real
   API is `read_frames_batched`.
2. **`simple-video-utils` is not in the base image**; pinned at `0.9.1`.
3. **The sclite chain hardcoded PHOENIX14's ground truth.** PHOENIX14-T is re-segmented
   and has its own references, so `GROUNDTRUTH_STM` maps dataset → STM. The two
   `phoenix2014-T-groundtruth-{dev,test}.stm` files are sha256-pinned at the same
   `slr_eval` commit (`a812814`); their 519 and 642 lines match the official counts.
4. **`published` in `result.json`** only ever held PHOENIX14's pair; `PUBLISHED_TABLE1`
   now carries both rows (PHOENIX14-T is 17.1 / 18.4, verified against Table 1).
5. **The collate crashed on short clips.** The first attempt died in epoch 4 with
   `Expected more than 1 spatial element when training, got input size [2, 1024, 1]`
   from `InstanceNorm1d` inside TLP's `LocalWeighting`. Our collate padded only to the
   batch maximum, so the padding-free `{K5,P2,K5,P2}` convolutions consumed **real**
   frames and a clip of length L emitted `(L-8)/4` steps — meaning anything under 20
   frames collapsed to a single temporal element. PHOENIX14-T's shortest clips are **16
   frames**; PHOENIX14's minimum is ~147, roughly 9× the threshold, which is why the
   defect stayed latent through every PHOENIX14 run. The fix adopts the lineage's own
   scheme (CorrNet `collate_fn` @ `a812814`): replicate 6 frames each side, round lengths
   up to a multiple of 4, and report per-clip lengths. The head then emits `ceil(L/4)`,
   so a 16-frame clip yields 4 steps. This also retires the `uniform-batch-lengths`
   deviation.

   **Comparability note:** this padding change also affects PHOENIX14, so the PHOENIX14-T
   result carries one correction the reported PHOENIX14 35.0/34.2 does not. PHOENIX14 was
   not re-run; the difference is recorded rather than silently absorbed.

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/yu-2025-olmd/modal_app.py --action build-cache-sharded --dataset phoenix14t --limit 40
```

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/yu-2025-olmd/modal_app.py --action preflight --dataset phoenix14t
```

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/yu-2025-olmd/modal_app.py --action train --dataset phoenix14t
```

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/yu-2025-olmd/modal_app.py --action rescore --dataset phoenix14t
```

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/yu-2025-olmd/modal_app.py --action sclite --dataset phoenix14t
```

Measured, against the estimates the compute gate carried:

| Stage | Projected (from PHOENIX14) | Measured on PHOENIX14-T |
| --- | --- | --- |
| Cache build | 10.3 h CPU, $7.75, 11.8 GiB | **2.8 min**, 8.83 GiB, 8257/8257 entries |
| Per-clip load | 6.45 s warm PNG | **0.39 s** mp4 decode |
| Training | 30.1 GPU-h (A100) | **17.7 GPU-h** (H100), inside the 24 h gate |

The cache build collapsed because PHOENIX14 stores ~179 separate PNG files per
utterance while PHOENIX14-T stores one mp4, and the cost was always per-request latency
rather than bandwidth.

### Result

`train-phoenix14t-wiring-fix-002` completed all 70 epochs on one H100 80GB in **15.90
GPU-hours** (788.7 s/epoch, peak 51.50 GiB), against the 17.7 predicted. Official chain,
beam-10, epoch-69 best-dev checkpoint:

| Split | Published | Reproduced | Sub | Del | Ins | Greedy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 17.1 | **34.5** | 17.9 | 15.0 | 1.6 | 34.7 |
| test | 18.4 | **32.6** | 16.0 | 14.6 | 2.0 | 32.8 |

Three observations, none of which close the gap:

- **The evaluation path is clean here.** The internal symmetric metric and the official
  chain agree to within 0.13 WER. On PHOENIX14 the same comparison differed by ~16.5 WER
  before correction — the difference is that PHOENIX14-T carries no PHOENIX markers for
  stripping to mishandle, so that entire class of error is absent by construction.
- **Decoding is not material.** Beam-10 beats greedy by 0.2 WER on both splits, matching
  the ~0.3 seen on PHOENIX14.
- **Deletions dominate** (15.0 dev / 14.6 test against 1.6 / 2.0 insertions), the same
  imbalance PHOENIX14 showed: the model under-emits glosses rather than hallucinating.

**Convergence caveat, recorded rather than smoothed over.** The best dev epoch is the
*last* one (69), and the final 30 epochs oscillate in a 34.63–38.26 dev band with 6
epochs within 1.0 WER of the best, while training loss sits flat at 34.4. The epoch-69
minimum is better read as the low end of a noise band than as a converged optimum, so the
reported value sits at the optimistic edge of roughly a 3.6 WER spread. The protocol is
unchanged — 70 epochs and best-dev selection were both fixed in advance — but a reviewer
comparing against a single published figure should know the run had not settled.

### A crash worth recording

The first PHOENIX14-T attempt died at epoch 4 on a 9-frame clip. PHOENIX14's shortest
utterance is 16 frames, so the pipeline had never met a clip short enough for the
temporal stack to reduce to zero length. The fix adopts the lineage's padding convention
in `collate`; it is a behaviour-changing change, recorded as a deviation, and it cost
~50 s/epoch because it pads ~12 extra frames per clip. The crashed run's checkpoints were
deleted before relaunching, so no weights trained under the broken padding carried over.

## Data provenance and permissions

All datasets verified on the shared `datasets` Volume before training (2026-09-08).

| Dataset | Version/subset/splits | Source and access date | License / cloud-use basis | Path in Volume `datasets` | Counts / checksum | Deviations |
| --- | --- | --- | --- | --- | --- | --- |
| PHOENIX14 | `phoenix-2014.v3`, multisigner, `features/fullFrame-210x260px` | RWTH i6 FTP `phoenix-2014.v3.tar.gz`; 2026-09-08 | CC BY-SA 3.0 per export (not independently re-verified); already in the project Volume | `rwth-phoenix-2014/` | **5672 / 540 / 629** train/dev/test — canonical; annotation CSVs SHA-256 recorded in `reproduction.json` | none |
| PHOENIX14-T | `PHOENIX-2014-T`, manual gloss (`orth`), per-split mp4 | RWTH i6; 2026-09-08 | CC BY-SA 3.0 per export (not independently re-verified); already in the project Volume | `rwth-phoenix-2014-t/` | **7096 / 519 / 642** — canonical; annotation CSVs SHA-256 recorded | none |
| CSL-Daily *(out of scope)* | `csl2020ct_v2`, `split_1`, mp4 videos | USTC-SLR project page; 2026-09-08 | Custom **research-only**; academic reproduction only, **no redistribution assumed** | `csl-daily/` | **18400 / 1077 / 1176** usable — see corpus defect below | not used by any in-scope target |

**CSL-Daily corpus defect:** `split_1.txt` lists 18401 unique training names, but
`S000005_P0004_T00` has no entry in `csl2020ct_v2.pkl` at all, so only 18400 training
utterances are usable. This is a defect in the published corpus, not in this
reproduction's loader; it was diagnosed by run `diagnose-csl-001` rather than accepted as
a silent off-by-one. CSL-Daily is now out of scope, but the finding is preserved.

All three passed `check_modal_dataset.sh` on 2026-09-08 and are mounted **read-only** at
their `/datasets/` paths. Split sizes were counted from the shipped annotation files and match
the canonical figures in the CSLR literature, which is the dataset-identity evidence.

**Data formats differ per corpus**, which the reimplementation must handle: PHOENIX14 ships
extracted per-utterance **PNG frame directories**; PHOENIX14-T ships per-split **mp4**
(plus PAMI feature archives that OLMD does not use, since it trains a ResNet34 on images);
CSL-Daily ships per-utterance **mp4**. Per AGENTS.md, `simple-video-utils` is used wherever
video files are decoded.

The export marked PHOENIX-2014-T and CSL-Daily `on_modal: yes` but left **PHOENIX-2014
blank**. Treating that as a lead rather than a fact was correct in both directions:
PHOENIX14 **is** present and complete. No data acquisition, `data.sh`, or Team S
coordination was needed, and no data gate was opened.

## Environment and patches

All runs use base image `ghcr.io/sign-language-processing/reproduction:latest` on Modal
(profile `repro-sign`), with the shared `huggingface-cache` Volume mounted at
`/cache/huggingface` (`HF_HOME`, `HF_HUB_CACHE`, `TORCH_HOME`). At level 3 there is no
upstream code to patch. External pins used by the reimplementation and evaluation:

| Pinned artifact | Revision | Role |
| --- | --- | --- |
| TLP released code (`hulianyuyy/Temporal-Lift-Pooling`) | structure verified against `modules/tconv.py` | TLP pooling reimplemented to match exactly (split, branches, losses, LocalWeighting, zero-inits) |
| CorrNet `evaluation/slr_eval` (`hulianyuyy/CorrNet`) | `a812814f…2263`, files sha256-pinned | Official PHOENIX preprocessing + ground-truth STMs (downloaded at runtime, hash-verified, **not committed** — corpus-derived data) |
| NIST SCTK | `62aabfa9…ce52` (tag v2.4.12) | `sclite` scorer, built in-container |
| torchvision ResNet34 | `IMAGENET1K_V1` weights | Paper-stated ImageNet-pretrained backbone |

Host used for setup: Linux 4.18 login node, no GPU, no Docker, system `python3` 3.6.8 with
`python3.9` available. Modal CLI 1.2.6 installed as a user-level package via
`python3.9 -m pip install --user modal`.

**`setup.sh` does not complete in this environment.** It blocks indefinitely on its first
step, `claude plugin marketplace add DietrichGebert/ponytail`, which needs an interactive
terminal. That step is the optional simplicity plugin, so the script never reaches its
Modal-install or `docker pull` steps. Modal was therefore installed directly. `docker` is
absent from this login node, so the base image cannot be pulled locally; image builds must
happen on Modal. Recorded as an environment limitation, not patched.

## Execution evidence

| Run ID | Kind | Platform | Terminal state | Key outcome |
| --- | --- | --- | --- | --- |
| `data-inspect-001` | data check | Modal CPU | succeeded / completed | Nine splits match; vocabularies 1295 / 1115 / 2000 confirm the paper |
| `diagnose-csl-001` | data check | Modal CPU | succeeded / completed | CSL-Daily corpus defect isolated to `S000005_P0004_T00` |
| `gpu-smoke-003` | smoke test | Modal A10 | succeeded / completed | Model constructs; finite loss; gradients flow |
| `preflight-phoenix14-001` | preflight | Modal A100-40GB | succeeded / completed | Preflight passed (pre-TLP); its compute estimate later found void |
| `throughput-phoenix14-001` | measurement | Modal A100-40GB | succeeded / completed | I/O is 165–295× GPU cost; bottleneck is per-request latency |
| `cache-probe-phoenix14-001` | measurement | Modal CPU | succeeded / completed | JPEG cache gives 24.1× with verified fidelity |
| `preflight-phoenix14-002` | preflight | Modal A100-40GB | succeeded / completed | Preflight passed with TLP; measured 0.422 s GPU step |
| `loader-benchmark-001` (+H100) | measurement | Modal A100/H100 | succeeded / completed | Prefetching hides 95.7% of load latency; H100 1.66× |
| `cache-build-sharded-*` | data prep | Modal CPU ×40 | succeeded / completed | Full JPEG cache, 6841 clips, all three splits |
| `train-phoenix14-seed0-attempt1` | full train | Modal A100-40GB | stopped / invalid_run | Blank collapse from `reduction="mean"` CTC bug; diagnosed, fixed |
| `train-phoenix14-fixprobe-002` | train probe | Modal A100-40GB | succeeded / completed | Corrected loss escapes collapse; resume chain to attempt 3 |
| `train-phoenix14-seed0-attempt3` | full train | Modal A100-40GB | succeeded / completed | 70 epochs complete; best dev epoch 57; targets produced |
| `rescore-phoenix14-001` | evaluate | Modal A10G | succeeded / completed | Beam-10 ≈ greedy (+0.3); scorer asymmetry worth ~13 WER |
| `sclite-phoenix14-001` | evaluate | Modal CPU | failed / command_failed | SCTK build missing `make install`; one scoped fix |
| `sclite-phoenix14-002` | evaluate | Modal CPU | succeeded / completed | Official-chain WER 35.0 dev / 34.2 test (beam-10) |

Full provenance for each — commands, Modal app IDs, dashboards, mounts, cache environment,
declared stop ceilings, and terminal classifications — is in `reproduction.json.runs`.
Modal profile `repro-sign` throughout.

## Guesses and deviations

The full ledger is `reproduction.json.guesses` (26 entries, each with evidence). The
material ones for interpreting the result:

| Detail | Paper says | This attempt used | Status |
| --- | --- | --- | --- |
| **MP-Block internals** | "detailed in the supplementary materials" — **which do not exist publicly** | MetaFormer macro-block, depthwise (9,3) mixer + BN/ReLU + pointwise | **blocking inference**; leading candidate for the residual gap |
| Loss weights γ₁, γ₂ | stated after Eq. 14 | 1 and 25 | resolved (stated, not a guess) |
| LMA context length | ablated in Table 5 | 9 (the table's optimum, matches Table 1) | resolved |
| LMA reduction ratio r | "compressed to C/r"; r unstated | r=2, recovered from Table 5's +0.22M/context parameter deltas | constrained by measurement |
| Cross-stage injection point | Eq. 6 output destination unstated | added to frame features feeding the temporal head | open, material (Table 8: 1.2 WER) |
| Decoding | unstated | beam-10 vs greedy both measured: Δ ≈ 0.3 WER | **resolved by measurement** |
| Metric implementation | Eq. 15 formula only | official sclite chain (pinned) as the comparable value; Levenshtein variants preserved | **resolved by measurement** |
| Gloss post-processing | unstated | official RWTH normalizations worth ~3.3 WER; an asymmetric-stripping bug in this repo's scorer worth ~13 WER found and fixed | **resolved by measurement** |
| Checkpoint rule | unstated | best dev WER epoch (57), lineage convention | adopted, recorded |
| Seed policy | none stated | single run, seed 0 | no variance estimate |
| Batching/lengths | unstated | batch-max replicate padding, uniform lengths (diverges from lineage per-sample lengths) | deviation `uniform-batch-lengths` |

One deviation recorded so far (`reproduction.json.deviations[uniform-batch-lengths]`):
batch collation pads clips to the batch maximum with replicated last frames and reports
that maximum as every sample's length, instead of the lineage's per-sample lengths with
6 replicated frames of explicit padding per side. The shorter clip in each batch-of-2
therefore has replicated tail frames counted as real CTC/BiLSTM input, and the two
padding=0 K5 convolutions consume 8 real frames of the batch-longest clip. The paper
never specifies batching, padding or length masking, so this diverges from the lineage
convention, not from the paper. Adopted before the full run launched.

### Paper-internal inconsistency (recorded, not resolved)

The abstract claims an absolute **1.6%** WER improvement on PHOENIX14. The body text says
**1.0% (Dev) and 1.7% (Test)** against TCNet's 18.1/18.9. Neither equals 1.6%. This is the
paper's own inconsistency; this reproduction records it rather than choosing a reading.

## Attempts, failures, and dead ends

1. **Queue ingestion — failed, by design.** *Hypothesis:* the export can be ingested with
   `ingest_candidate.py`. *Test:* inspected the file against the script's contract.
   *Result:* the export is a single `reproductions` object, not the required top-level
   array; it has no top-level `paper_id`, **no `confirmation` field anywhere**, and its
   top-level `status` is `in_progress` (only the nested `expand.paper.status` is `final`).
   *Kept:* recorded the assignment as `direct_user_request` instead. Synthesizing a queue
   record would have meant fabricating a `confirmation` value the export never contained,
   which `references/candidate-record.md` forbids. The export is preserved as evidence by
   path and SHA-256. **Worth reporting to the REPRO-SIGN maintainers as a queue-exporter
   observation.**
2. **`setup.sh` — hangs.** *Hypothesis:* the script runs unattended. *Test:* ran it;
   timed out at 2 minutes. *Result:* blocks on the interactive `claude plugin marketplace
   add`. *Kept:* installed Modal directly; recorded the limitation rather than patching a
   repository-level script.
3. **Source search — no code.** Recorded above; drives preference level 3.
4. **First full run — blank collapse (agent-introduced bug).** *Hypothesis:* torch's
   default CTC reduction is equivalent to the lineage's. *Result:* false —
   `reduction="mean"` divides by target length, shrinking CTC gradients ~11.5× against
   the γ₂=25 distillation term; all three branches collapsed to mutual blank agreement
   for 29 epochs (~8.7 GPU-hours). *Kept:* corrected to the lineage's `none` + batch
   mean (verified in CorrNet source), verified by a 4-epoch probe, resumed. Two process
   lessons recorded: a 12-step preflight structurally cannot catch slow collapse, and
   per-component loss logging (added after) turns a 29-epoch diagnosis into a 3-epoch one.
5. **Asymmetric WER scoring (agent-introduced bug).** The training scorer stripped
   PHOENIX markers from references but not hypotheses, inflating every reported WER by
   ~13 points and understating the model for the entire first analysis. Found during the
   metric-mismatch diagnosis, quantified by `rescore-phoenix14-001`, fixed for future
   runs. The paper is not implicated; the official chain keeps `__ON__`/`__OFF__` on
   both sides.
6. **SCTK build — missing `make install`.** `sclite-phoenix14-001` failed
   deterministically; one scoped change, retried, succeeded.

## Open validation items

The 2026-11-04 `validate_reproduction.py` from `main` passes with **zero issues**. The
validator had since added agent attribution checks, and the report was brought up to them
on that date. What remains open is scope, not validity:

- Several runs' launch-time timestamps appear to be CEST local time labelled as UTC; for
  example, launch commits and session logs place some starts about two hours earlier than
  recorded. They are kept as originally recorded.

- All four in-scope targets are produced; the compute gate is resolved. What remains
  open is diagnosis, not scope.
- The residual ~14-18 point gap, seen on **both** datasets, is undiagnosed beyond the
  recorded guesses. PHOENIX14-T narrows the search usefully: it rules out evaluation
  methodology as the explanation, since that dataset's clean metric path produces a
  similar shortfall.
  The cheapest next probes, if pursued: per-sample batch lengths (deviation
  `uniform-batch-lengths`), alternative cross-stage injection points, and MP-Block
  variants — each a behavior-changing protocol experiment requiring its own decision,
  never silent tuning toward the published score.

## Candidate flags, ethics, and human evaluation

The export reports `includes_human_evaluation: "no"` and `potential_ethical_concerns:
"no"`. The reproduction adds no participants and no human raters: it trains on
already-collected, published sign language corpora and scores WER automatically. No
ethics gate is opened. CSL-Daily's research-only terms are a **licence/access** question
for the data gate, not an ethics question. All three corpora contain identifiable signers,
so any redistribution of data or predictions must be checked against each licence before
publication.

## Author and team contact

- **Who and when:** on 2026-09-16 the assigned operator emailed Sheng Liu, the paper's
  second author, on behalf of the REPRO-SIGN collaboration. This came after the
  independent PHOENIX14 attempt had completed (2026-09-10), as AGENTS.md requires, and
  before the PHOENIX14-T run.
- **Why:** to ask for code, implementation details and the supplementary material for
  the OLMD row of Table 1. The email said anything shared would be published with the
  results, with full attribution to the authors.
- **Response:** none as of 2026-11-05.
- **What it changed:** nothing. No author material was received or used, so every
  inferred detail, including the MP-Block internals, remains a recorded guess.
- **Team S:** not involved. This was a request for author help, not a data-access request.

Details are in `reproduction.json.contacts`.
