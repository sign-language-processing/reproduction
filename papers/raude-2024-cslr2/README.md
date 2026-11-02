# Raude et al. (2024) — CSLR²: reproduction

**Paper ID:** `13195df56368306dc1c1f9733cebb8dd694c94af`

**Citation:** Charles Raude, K R Prajwal, Liliane Momeni, Hannah Bull, Samuel Albanie, Andrew Zisserman, Gül Varol. *A Tale of Two Languages: Large-Vocabulary Continuous Sign Language Recognition from Spoken Language Supervision.* arXiv:2405.10266v1, 2024.

**Paper:** https://arxiv.org/abs/2405.10266v1 · **Code/artifacts:** https://github.com/gulvarol/cslr2 (MIT, pinned `5f3bd697e1d2e184a5882cbbd0da64ce9455fd28`), plus a released CSLR² checkpoint and data bundle on Google Drive (see Source provenance)

**Preference level:** 1

The preference level is provisional; see `status.preference_level_basis`. Official code and a released checkpoint cover the in-scope CSLR² and InfoNCE cells. Swin-8K† is evaluated only from released artifacts.

**Pipeline status:** not yet set. Target scope is resolved: 22 cells in scope, 19 out of scope. The attempt is paused at two open gates for the team, `data-bobsl-access` and `compute-training-runs`, before any download, container build or run. No target has a terminal result, and AGENTS.md forbids `blocked_on_compute` before the team discussion.

**Numerical agreement:** `not_assessed` — no target has been produced.

**Attempt date:** 2026-11-02

## Reproduction agents

| Agent ID | Model and version | Agent application | Contribution | Attribution evidence / unknowns |
| --- | --- | --- | --- | --- |
| `claude-opus-5-5-claude-code` | Claude Opus 5.5 (`claude-opus-5-5`) | Claude Code 2.1.316 | Assignment intake, paper verification, target ledger, source search and pinning, read-only Modal Volume inspection, target and data gates | Session system context names the model and ID. `claude --version` returned 2.1.316. No runs executed. |

## Scope and target contract

### Assignment intake

The assignment is a single `reproductions`-collection export plus the paper PDF. The export's `what_to_reproduce` is "Table 4".

The export cannot go through `ingest_candidate.py`:

- The paper record is nested under `expand.paper`. Its status is `final`; the top-level `in_progress` status belongs to the reproduction task.
- There is no `confirmation` field.
- Its `id` values are internal queue IDs, not the paper ID.

It is therefore recorded as a `direct_user_request`, following `papers/pu-2024-siformer`. The record is kept unchanged except that every email address is replaced by `[redacted]`, including `assignees`, `finalized_by` and `status_history`. Its SHA-256 is recorded in `reproduction.json`.

### What Table 4 contains

The supplied PDF is byte-identical to arXiv v1, the only arXiv version. No venue version was found. Table 4 (p. 8) holds **41 numbers in 9 rows**.

**CSLR on CSLR-TEST.** Columns are WER ↓, mIoU ↑ and F1@{0.1, 0.25, 0.5} ↑.

| Row | Values | Origin of the evaluated system |
| --- | --- | --- |
| Subtitle-based spotting [4] | 93.8, 7.1 | Released BOBSL spottings, thresholds [M.8 D.8] |
| Subtitle-based spotting [43] | 85.5, 18.1 | Released dense annotations, [M\*.8 D\*.8 A.0 P.5] |
| ISLR I3D-2K [4] | 453.5, 8.6, 10.6, 8.5, 5.4 | Released I3D weights, raw (repetitions collapsed) |
| ISLR I3D-2K [4] † | 82.5, 18.1, 25.7, 23.8, 16.0 | Same weights, paper post-processing |
| ISLR I3D-8K [43] † | 74.7, 27.0, 36.7, 35.4, 26.7 | I3D weights from [43] |
| ISLR Swin-8K [44] † | 71.7, 30.1, 40.0, 37.9, 27.2 | Authors' own Video-Swin-T, trained "in a similar fashion as [44]" |
| CSLR² (Ours) † | 65.5, 35.5, 47.1, 46.0, 37.1 | This paper |

**Retrieval.** Columns are SENT-VAL (2K) T2V R@1 and V2T R@1, then SENT-TEST (20K) T2V R@1, T2V R@5, V2T R@1 and V2T R@5.

| Row | Values | Origin |
| --- | --- | --- |
| InfoNCE | 38.9, 39.2, 19.5, 35.1, 18.9, 33.8 | Baseline trained by the authors |
| CSLR² (Ours) | 51.7, 50.2, 29.4, 45.2, 28.1, 44.9 | This paper |

**No number is copied from earlier work.** CSLR-TEST is introduced by this paper, so every CSLR value was computed by the authors on it. The bracketed citations name where the evaluated model or annotations come from, not where the number comes from. The InfoNCE row equals the authors' own ablations (Table 3 row 1, Table A.8 row 2). Every target therefore has `copied_baseline: false`.

### Metric definitions

Metrics are taken from Sec. 5.1 and the pinned `frame_level_evaluation.py`:

- **WER:** corpus-level Levenshtein over lexical glosses, with synonyms counted correct.
- **mIoU:** word-set IoU per subtitle, averaged over subtitles.
- **F1@τ:** pooled segment F1 at temporal IoU threshold τ.
- **R@k:** standard retrieval recall at rank k (`loops/retrieval.py`, ties averaged).

### Scope decision (gate `target-table4-rows`, resolved 2026-11-02)

The ledger in `reproduction.json` has one target per number. Each target carries `scope_alternatives`, listing which options include it, and a `scope_note`. Three options were put to the operator:

- **A — CSLR² rows only:** 11 numbers.
- **B — A plus the baselines the authors trained themselves:** 22 numbers.
- **C — the full table:** 41 numbers. This was the skill's default for a whole-table assignment.

The operator chose **B**, with a condition on Swin-8K†:

| Cells | In scope | How they are handled |
| --- | --- | --- |
| 11 CSLR² cells: CSLR row and retrieval row | yes | Official cslr2 code |
| 6 InfoNCE retrieval cells | yes | Official cslr2 code; a second training run via config overrides |
| 5 ISLR Swin-8K† cells | yes | Evaluated only if released artifacts give the authors' Video-Swin-T per-frame predictions on CSLR-TEST without retraining. That means either released per-frame predictions for test episodes, or the released features plus the trained classifier head. Otherwise they close as `not_produced` with reason `compute_budget_blocked`, a compute gate and evidence. Video-Swin-T is not retrained. |
| 19 cells: spotting [4]/[43], I3D-2K raw/†, I3D-8K† | no | Kept with published values. They evaluate prior-work models and annotations on CSLR-TEST and are not needed to test the paper's claimed comparison. |

**Contract gap to settle before handoff.** The controlled target reason codes have nothing for a cell that is *not copied* but was excluded from scope by the operator. `copied_baseline` requires `copied_baseline: true`, which would be false here. The 19 out-of-scope targets therefore carry no `result` yet. This is recorded in the gate resolution rather than worked around by misusing an existing code.

## Source provenance

| Artifact | Canonical source | Pinned revision / SHA-256 | Role |
| --- | --- | --- | --- |
| Paper PDF | https://arxiv.org/pdf/2405.10266v1 (accessed 2026-11-02) | `246fb354a2c41d24dd0b7b93062a851fd2ed56a47490aac92cbf89f8b1e026f4` | Targets and protocol |
| Published code | https://github.com/gulvarol/cslr2 | `5f3bd697e1d2e184a5882cbbd0da64ce9455fd28` | Training, retrieval test, CSLR extraction and evaluation |
| CSLR² checkpoint | Google Drive `1qyFHSFnxmy1rRGjlKEBfsjC8yt2kdalx` | Unpinned (mutable; re-uploaded 2024-07-02 per issue #1). Hash at acquisition | Evaluation cross-check |
| `bobsl.zip` (annotations, metadata, CSLR CSVs) | Google Drive `13pp83GCoy1SVScvZRmNsoFxtNm8ogI7h` | Unpinned. Hash at acquisition | Required by `config/paths/public.yaml` |
| `t5_checkpoint.zip` | Google Drive `1hxkb8KAC0sgSYKefOLue1wyO2fJmmqxT` | Unpinned. Hash at acquisition | Frozen t5-large text encoder |
| BOBSL Video-Swin-T features (262G) | `thor.robots.ox.ac.uk/~vgg/data/bobsl/features/lmdb/feats_vswin_t-bs256_float16/` | Credentialed; not accessed | Model input |
| BOBSL RGB LMDB (1.5T) | `thor.robots.ox.ac.uk/~vgg/data/bobsl/videos/lmdb/rgb_anon-public_1962/` | Credentialed; not accessed | Visualisation; I3D rows |
| BOBSL paper | https://arxiv.org/pdf/2111.03635v1 | `0e75ee96bd548c0946a3a0737040d58ad7e44c0582e3b1e85b82a95789e29006` | SENT-VAL/SENT-TEST definitions, licence basis |
| BBC R&D BOBSL access terms | https://www.bbc.co.uk/rd/projects/extol-dataset (accessed 2026-11-02) | HTML `d4b006f6…` | First-party access and use restrictions |
| CSLR² data page and licence | https://gulvarol.github.io/cslr2/data.html | data.html `3cea110c…`, license.txt `65b0727d…` | BOBSL-CSLR CC BY 4.0 terms |
| transpeller (considered) | https://github.com/prajwalkr/transpeller | `dca7c8e3d463ca3c5909d9ee47c0ae8aeba4c0ff` | Not selected: Video-Swin-**S** checkpoint of [44]; Table 4 uses the authors' Video-Swin-**T** |

### Searches performed (2026-11-02)

- **arXiv and project pages.** arXiv abs/PDF; the paper's project URL, which redirects to gulvarol.github.io/cslr2, and its data page.
- **The official repo.** Its history (single branch, no tags or releases) and both issues.
- **GitHub repository searches** for "cslr2" and "BOBSL". No baseline code was found.
- **transpeller**, the repo pointed to from cslr2 issue #2.
- **The BOBSL page.** It refused TLS connections from this machine (curl error 35, HTTP 503) and WebFetch also failed; the Wayback availability API was rate-limited.
- **Venue version.** OpenReview/DBLP and web search found only CoRR 2024.

### What the released code covers

- **CSLR² and InfoNCE.** The repository covers CSLR² end to end. The InfoNCE baseline can be expressed as config overrides: `pooling=cls`, HN-NCE with β=0, `lda_sign_ret=0`.
- **Partial hooks for baselines.** `extract_for_eval.py swin=True` evaluates a Swin classifier head, and `frame_level_evaluation.py automatic_annotations=True` accepts spotting predictions.
- **What is missing:**
  - I3D inference code.
  - A converter from the BOBSL spotting releases to the pickle format.
  - A documented Swin-T head checkpoint.
  - An InfoNCE checkpoint.

## Results

No target has been run. All 41 targets are listed in `reproduction.json.targets` with paper location, split, metric, published value, copied-baseline determination and scope. None has a terminal result yet.

| Group | Cells | In scope | Published values |
| --- | ---: | --- | --- |
| CSLR² (Ours) † — CSLR-TEST | 5 | yes | WER 65.5, mIoU 35.5, F1 47.1 / 46.0 / 37.1 |
| CSLR² (Ours) — retrieval | 6 | yes | SENT-VAL R@1 51.7 / 50.2; SENT-TEST T2V 29.4 / 45.2, V2T 28.1 / 44.9 |
| InfoNCE — retrieval | 6 | yes | SENT-VAL R@1 38.9 / 39.2; SENT-TEST T2V 19.5 / 35.1, V2T 18.9 / 33.8 |
| ISLR Swin-8K [44] † | 5 | yes, conditional on released artifacts | WER 71.7, mIoU 30.1, F1 40.0 / 37.9 / 27.2 |
| Spotting [4]/[43], I3D-2K raw/†, I3D-8K † | 19 | no | See Scope and target contract |

## How to repeat this

Nothing executable exists yet. The work stopped at the gates before writing a Dockerfile, `data.sh` or Modal entry point. The Modal checks made so far were:

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh            # verified profile and workspace repro-sign
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh volume ls datasets /
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh volume ls huggingface-cache /
.agents/skills/reproduce-paper/scripts/check_modal_dataset.sh bobsl         # fails: path /bobsl/ does not exist
.agents/skills/reproduce-paper/scripts/check_modal_dataset.sh bobsl-cslr    # fails: path /bobsl-cslr/ does not exist
```

## Data provenance and permissions

| Dataset | Version/subset/splits | Source and access date | License/permission and cloud-use basis | Path in Volume `datasets` | Counts / manifest / checksum | Deviations |
| --- | --- | --- | --- | --- | --- | --- |
| BOBSL | Video-Swin-T feature LMDB, pseudo-labels, automatic and manual subtitles. SENT-VAL_H, SENT-TEST | BBC R&D terms page, 2026-11-02. Oxford page unreachable | BBC-owned, non-commercial academic research only. Per-person signed Permission to Use with an individual password; "for your use only"; no redistribution; held securely; 12 months. Shared-Volume storage and Modal processing **not established** | `bobsl/` — **absent** | SENT-VAL_H 1,973, SENT-TEST 20,870 (BOBSL paper Table 4). No files yet | None |
| BOBSL-CSLR | Processed lexical-gloss CSVs `cslr/manual_glosses_extended_boundaries_fix_alignment/0/{train,val,test}` inside `bobsl.zip` | gulvarol.github.io/cslr2/data.html, 2026-11-02 | CC BY 4.0 for the text annotations; videos and features stay under the BOBSL licence | `bobsl/cslr/` — **absent** | Test 4,518, val 1,153, train 1,463 sentences (Table A.2). Sec. 5.1 says 4,950 test subtitles | None |

**Data gate (`data-bobsl-access`, open).** `check_modal_dataset.sh bobsl` and `check_modal_dataset.sh bobsl-cslr` both fail because the paths do not exist (2026-11-02). The BBC terms are per person and forbid redistribution, so Team S needs to decide three things:

1. Whether an eligible study member holds, or will apply for, a current BBC BOBSL agreement.
2. Whether those terms allow a copy in the shared `repro-sign` Volume `datasets/bobsl`, which other workspace members can read, and processing on Modal. Another permitted location may be required, and the BBC may need to confirm.
3. How the access path is supplied: through an approved channel, never as a credential in chat or in committed files.

No BOBSL file, `bobsl.zip`, CSLR² checkpoint or T5 bundle has been downloaded.

## Compute plan (gate `compute-training-runs`, open)

The in-scope cells need two full training runs of the official code: CSLR² for 11 cells and the InfoNCE baseline for 6 cells. Swin-8K† involves no training. The gate was opened before any measurement, at the operator's request, so it reaches the team together with the data gate.

**Estimate.** These figures are paper-reported, not measured:

| Item | Figure | Basis |
| --- | --- | --- |
| CSLR² training, 20 epochs | ≤ 20 h on 4 × V100-32GB, about 80 V100 GPU-hours | Paper App. A and cslr2 README |
| InfoNCE training | about 80 V100 GPU-hours | Assumed equal to CSLR²; the paper does not state it |
| Total | up to about 160 V100 GPU-hours | Above the 24 GPU-hour autonomous ceiling per run |
| Modal GPU cost, both runs | about USD 400 (A100 80GB) to USD 632 (H100) | Upper bracket: no speed-up over V100 assumed; modal.com/pricing read 2026-11-02 |
| CPU and memory | about USD 60 per run | 40 physical cores, about 128 GiB, 20 h |
| Storage | about 280 GiB in `datasets` | Features 262G, `bobsl.zip` 15G decompressed, T5 1.4G |

**Configuration.** Exact replay needs 4-process DDP on one node, for example one Modal container with 4 GPUs at batch 128 per process. The training loop all-gathers sentence embeddings, so SentRet negatives span the global 512 batch. SignRet, however, is computed per process over 128 subtitles. A single GPU at batch 512 would therefore change the sign-level negatives, which is a protocol deviation and is not proposed.

**Decision needed in `#repro-sign-team-r`:** whether to fund the two runs, plus per-run wall-time, GPU-hour and cost ceilings and a restart allowance. The estimate will be replaced by a measured one from the representative preflight. That preflight needs BOBSL, so it can only happen after the data gate clears and before any full run.

## Environment and patches

None yet. The plan is the root `reproduction` base image with the cslr2 dependencies.

The upstream environment is Python 3.9.16, PyTorch 1.12.1 with CUDA 11.6, hydra-core 1.3.2, sentence-transformers 2.2.2, lmdb, and nltk wordnet.

Expected adaptations are container paths only. Upstream `main.py test=True` also needs the RGB LMDB unless `vis` handling is adapted. Any change will be one patch per demonstrated failure.

## Execution evidence

No runs.

## Guesses and deviations

None made yet. The following discrepancies between the paper and the code were found and must be settled before or at the protocol stage. They are also listed in `reproduction.json.protocol_notes`.

| Detail | Paper/evidence says | Code says | Status |
| --- | --- | --- | --- |
| SENT-VAL/SENT-TEST sizes | Sec. 5.1: 20,870 / 1,973 | Table 4, App. D.2 and BOBSL paper: 1,973 / 20,870 | Targets follow Table 4 and BOBSL |
| CSLR-TEST size | Sec. 5.1: 4,950 subtitles | Tables 1/A.2: 4,518 sentences | Check against CSVs after acquisition |
| Retrieval test set | 20K SENT-TEST | README: "25K manually aligned test set" | Check against data |
| θ, m tuning set | Sec. 5.2: validation CSLR sequences | `frame_level_evaluation.py` searches `0/train/*.csv` | Protocol item |
| Checkpoint selection | App. A: best T2V R@1 on manually aligned validation subtitles | `main.py` overwrites `model_best.pth` every epoch (`best_t2v` not carried); validation retrieval uses a random 2,000 subset of the automatic subtitles | Protocol item before retraining |
| SENT-VAL evaluation of a fixed checkpoint | Reported in Table 4 | No entry point (`test=True` runs only SENT-TEST) | Needs a minimal entry point |
| Training compute | Under 20 h on 4 × V100 (about 80 V100 GPU-hours) | Same in README | Gate `compute-training-runs` open; a measured estimate replaces this before launch |

## Attempts, failures, and dead ends

- **BOBSL page.** It could not be fetched (TLS reset, HTTP 503; WebFetch failed; Wayback rate-limited). The BOBSL arXiv paper was used for the licence basis and split counts instead.
- **Project licence link.** The README's `imagine.enpc.fr` licence URL returns 404. The redirected GitHub Pages copy was used.

## Candidate flags, ethics, and human evaluation

- **Queue flags.** The queue reports no human evaluation, no ethical concerns and no copied scores. The paper confirms the absence of human evaluation in Table 4, and the copied-score reading is above.
- **Signer data.** BOBSL contains identifiable BSL interpreters from BBC broadcasts. The in-scope work only processes released features and annotations. Its secure-holding and storage terms are part of the open data gate; no new participant interaction is involved.

## Author and team contact

- **Authors:** none contacted.
- **Operator:** decided the Table 4 scope at gate `target-table4-rows` on 2026-11-02.
- **Team S:** coordination needed for gate `data-bobsl-access`; not yet contacted by this agent.
- **`#repro-sign-team-r`:** discussion needed for gate `compute-training-runs`; not yet posted by this agent.
