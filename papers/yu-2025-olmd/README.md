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

**Pipeline status:** `blocked_on_compute` — the pipeline is implemented and verified end to
end with TLP active, but the measured full-run requirement (~25.5 GPU-hours for PHOENIX14
GPU-bound, ~57.8 with the cache alone) exceeds the 24-hour single-GPU autonomous gate.
Awaiting authorization; see gate `compute-full-run-table1`.

**Numerical agreement:** `not_assessed` — no in-scope target produced a comparable value.

**Attempt date:** 2026-09-08 (in progress)

> **Status: blocked on compute, awaiting a human budget decision.** Everything up to the
> full run is complete and verified: assignment record, target ledger, Modal access, data
> gate, the level-3 reimplementation, the metric, and the AGENTS.md preflight. The four
> requested numbers were not produced because the reviewed requirement exceeds the
> autonomous compute gate — not because of any data, code, or specification obstacle.

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

No in-scope target produced a comparable value. The pipeline is complete and verified;
the four requested numbers await the compute-gate decision.

| Target ID | Paper location | Dataset/split | Original | Reproduced | Terminal reason |
| --- | --- | --- | ---: | ---: | --- |
| `table1-p14-dev-wer` | Table 1, OLMD row | PHOENIX14 / dev | 17.1 | — | `compute_budget_blocked` |
| `table1-p14-test-wer` | Table 1, OLMD row | PHOENIX14 / test | 17.2 | — | `compute_budget_blocked` |
| `table1-p14t-dev-wer` | Table 1, OLMD row | PHOENIX14-T / dev | 17.1 | — | `compute_budget_blocked` |
| `table1-p14t-test-wer` | Table 1, OLMD row | PHOENIX14-T / test | 18.4 | — | `compute_budget_blocked` |
| 60 copied rows | Table 1, prior methods | both datasets | see `reproduction.json` | n/a | `copied_baseline` |

Each in-scope target cites the three measurement runs, four artifacts, and gate
`compute-full-run-table1` as evidence.

**`compute_budget_blocked` is the earliest blocker that independently prevents the
requested pipeline.** The assignment, targets, data, environment, implementation, metric,
and preflight are all resolved and verified — nothing else stands in the way of launching.

**One fidelity caveat on any future produced value.** The MP-Block, the core of the
paper's headline HMP/VMP contribution, is inferred: the paper defers it to supplementary
material that does not exist publicly. A number from this implementation is therefore
conditional evidence about an inferred architecture, and a reviewer should weigh that
before treating agreement or disagreement as a statement about the paper.

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

For 70 epochs, batch size 2, with best-dev checkpoint selection (a dev pass each epoch):

| Scenario | PHOENIX14 | PHOENIX14-T | Both |
| --- | ---: | ---: | ---: |
| As measured (serial PNG loading) | ~317 GPU-h | — | — |
| **With the JPEG cache** | **~57.8 GPU-h** ($121, 2.4 d) | ~72.3 GPU-h | ~130 GPU-h |
| **Cache + prefetching (GPU-bound)** | **~25.5 GPU-h** ($54, 1.1 d) | ~31.9 GPU-h | ~57 GPU-h |

Plus a one-time, CPU-only, reusable cache build: ~12.6 h and ~$4.75 for PHOENIX14.

`cache-probe-phoenix14-001` verified the cache: per-clip reads drop from 6.45 s to
**0.267 s (24.1×)**, frame counts match exactly, and JPEG q90 costs **1.028 mean absolute
pixel levels out of 255**. The cache is 1.73 MB/clip (~11 GiB per dataset, against 224 GiB
raw `uint8`), built at **256×256 pre-crop** so the paper's random 224×224 crop and 50%
horizontal flip still vary per epoch — caching post-crop would have silently removed the
stated augmentation.

The 317-hour figure describes **this reproduction's serial PNG loading, not the paper's
protocol** — which is why no compute gate was opened on it. The cache and a prefetching
loader are ordinary engineering, not protocol changes.

### The gate still applies

Even the GPU-bound best case of **~25.5 GPU-hours for PHOENIX14 marginally exceeds** the
AGENTS.md 24-hour single-GPU threshold, and the cache-only figure of ~57.8 clearly does.
Gate `compute-full-run-table1` remains open for authorization.

Cost is not the constraint: ~$54–121 for PHOENIX14 is well under 1% of the study's
CHF 20,000. Wall-clock is: 1.1–2.4 days on one GPU.

**Reducing epochs, evaluation scope, or checkpoint-selection fidelity to fit the gate is
not on the table** — that would be a protocol change requiring its own decision, and
AGENTS.md forbids shrinking the experiment silently.

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

Data, build, train, and evaluation commands are **not yet written** — they follow the data
gate and the reimplementation.

## Data provenance and permissions

Nothing verified yet; the data gate requires Modal access.

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

Planned base image `ghcr.io/sign-language-processing/reproduction:latest`; no container
built yet and no patches applied. At level 3 there is no upstream code to patch — any
prior-work module reused (TLP pooling, VAC distillation) will be pinned to an upstream
commit and recorded here.

Host used for setup: Linux 4.18 login node, no GPU, no Docker, system `python3` 3.6.8 with
`python3.9` available. Modal CLI 1.2.6 installed to `~/.local/bin` via
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

Full provenance for each — commands, Modal app IDs, dashboards, mounts, cache environment,
declared stop ceilings, and terminal classifications — is in `reproduction.json.runs`.
Modal profile `repro-sign` throughout.

## Guesses and deviations

| Detail | Paper/evidence says | This attempt will use | Rationale | Effect on interpretation |
| --- | --- | --- | --- | --- |
| Loss weights γ₁, γ₂ | Formula given; **no numeric values** | open | Search the full PDF/appendix; else adopt the VAC convention the loss derives from | Directly affects the CTC/distillation balance; a material unknown |
| LMA context length | Ablated in Table 5; **not stated** for the headline results | open | Read the optimum from Table 5 and record the selection | Central to the paper's own contribution |
| Seed policy | **None stated**; one value per cell | single run per target | No repeat count is defined by the paper | No variance estimate; limits comparability |
| Metric implementation | Formula given; **tool not named** | official per-dataset evaluation script | Standard for these benchmarks | Small WER differences can come from scoring details |
| Checkpoint rule | **Not stated** | best dev WER epoch | The VAC/SMKD/CorrNet lineage convention this work follows | Affects which epoch's test WER is reported |

No deviations yet — nothing has been run.

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

## Open validation items

`validate_reproduction.py` reports 8 issues, down from 171 at the start. Every remaining
one requires an executed run:

- 4 × `target needs a terminal result` — the four in-scope OLMD targets.
- `invalid pipeline status` / `invalid numerical_agreement` / `zero-produced status needs
  blocker` — both status axes stay unset until a run produces a value or a blocker is
  actually established. Setting a blocker now would assert a terminal outcome that has not
  been demonstrated.

The dataset and experiment errors are resolved.

## Candidate flags, ethics, and human evaluation

The export reports `includes_human_evaluation: "no"` and `potential_ethical_concerns:
"no"`. The reproduction adds no participants and no human raters: it trains on
already-collected, published sign language corpora and scores WER automatically. No
ethics gate is opened. CSL-Daily's research-only terms are a **licence/access** question
for the data gate, not an ethics question. All three corpora contain identifiable signers,
so any redistribution of data or predictions must be checked against each licence before
publication.

## Author and team contact

None. AGENTS.md permits author contact only after an independent attempt, and no attempt
has been made yet.
