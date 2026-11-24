# Raude et al. (2024) — CSLR²: reproduction

**Paper ID:** `13195df56368306dc1c1f9733cebb8dd694c94af`

**Citation:** Charles Raude, K R Prajwal, Liliane Momeni, Hannah Bull, Samuel Albanie, Andrew Zisserman, Gül Varol. *A Tale of Two Languages: Large-Vocabulary Continuous Sign Language Recognition from Spoken Language Supervision.* arXiv:2405.10266v1, 2024.

**Paper:** https://arxiv.org/abs/2405.10266v1 · **Code/artifacts:** https://github.com/gulvarol/cslr2 (MIT, pinned `5f3bd697e1d2e184a5882cbbd0da64ce9455fd28`), plus a released CSLR² checkpoint and data bundle on Google Drive (see Source provenance)

**Preference level:** 1

Level 1 was planned and never exercised: official code and a released checkpoint cover the in-scope CSLR² and InfoNCE cells, but the attempt closed at the data gate before any container build or run.

**Pipeline status:** `blocked_on_data` — the BBC refused BOBSL access to REPRO-SIGN on 2026-11-24, and every Table 4 cell needs BOBSL.

**Numerical agreement:** `not_assessed` — no in-scope target produced a value, so nothing can be compared with the published numbers.

**Attempt date:** 2026-11-02 to 2026-11-24

## Reproduction agents

| Agent ID | Model and version | Agent application | Contribution | Attribution evidence / unknowns |
| --- | --- | --- | --- | --- |
| `claude-opus-5-5-claude-code` | Claude Opus 5.5 (`claude-opus-5-5`) | Claude Code 2.1.316 | 2026-11-02: assignment intake, paper verification, target ledger, source search and pinning, read-only Modal Volume inspection, target and data gates | Session system context names the model and ID. `claude --version` returned 2.1.316, and the session transcript confirms that version for the running client, with effort `xhigh`. No runs executed. |
| `claude-opus-5-5-claude-code-closure` | Claude Opus 5.5 (`claude-opus-5-5`) | Claude Code 2.1.331 | 2026-11-24: report maintenance and closure. Rebase onto the current contract, queue-export re-ingestion, recording the BBC refusal as reported by the operator, gate resolutions, terminal results, status, guesses and deviations | Session system context names the model and ID; the session transcript records client version 2.1.331 and effort `xhigh`. Did not contact the BBC and ran no experiment. |

## Scope and target contract

### Assignment intake

The assignment is the Reproduction Tracker export `reproductions-13195df56368306dc1c1f9733cebb8dd694c94af-2026-10-30.json` (SHA-256 `6e64ca02825163fc0cec6d3f1093bff374c939181d49e1f00a76171830cdc487`). Its paper record (`expand.paper`) has status `final`, and `what_to_reproduce` is "Table 4".

On 2026-11-24 the export was ingested with `ingest_candidate.py` as a `queue_record`. `reproduction.json.assignment` keeps only the export's filename and SHA-256, and every email address in the stored record is replaced by `[redacted]`.

On 2026-11-02 it had been recorded as a `direct_user_request`, because the ingestion script then expected the retired candidate-list format. The stored record is unchanged by the re-ingestion.

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

**Retrieval on SENT-VAL and SENT-TEST.** Columns are SENT-VAL (2K) T2V R@1 and V2T R@1, then SENT-TEST (20K) T2V R@1, T2V R@5, V2T R@1 and V2T R@5.

| Row | Values | Origin |
| --- | --- | --- |
| InfoNCE | 38.9, 39.2, 19.5, 35.1, 18.9, 33.8 | Baseline trained by the authors |
| CSLR² (Ours) | 51.7, 50.2, 29.4, 45.2, 28.1, 44.9 | This paper |

**No number is copied from earlier work.** CSLR-TEST is introduced by this paper, so every CSLR value was computed by the authors on it. The bracketed citations name where the evaluated model or annotations come from, not where the number comes from. The InfoNCE row equals the authors' own ablations (Table 3 row 1, Table A.8 row 2), and CSLR² is the paper's own model. Every target therefore has `copied_baseline: false`.

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

| Cells | In scope | How they were to be handled |
| --- | --- | --- |
| 11 CSLR² cells: CSLR row and retrieval row | yes | Official cslr2 code |
| 6 InfoNCE retrieval cells | yes | Official cslr2 code; a second training run via config overrides |
| 5 ISLR Swin-8K† cells | yes | Evaluated only if released artifacts give the authors' Video-Swin-T per-frame predictions on CSLR-TEST without retraining; otherwise `compute_budget_blocked`. Video-Swin-T is not retrained. |
| 19 cells: spotting [4]/[43], I3D-2K raw/†, I3D-8K† | no | Kept with published values. They evaluate prior-work models and annotations on CSLR-TEST and are not needed to test the paper's claimed comparison. |

**Contract gap.** The controlled reason codes have nothing for a cell that is *not copied* but was excluded by a scope decision. It did not arise at closure: the 19 out-of-scope cells are independently blocked by the BOBSL refusal, so they close as `data_permission_blocked`, and each detail states the scope exclusion. No controlled code covers that case in general.

## Source provenance

| Artifact | Canonical source | Pinned revision / SHA-256 | Role |
| --- | --- | --- | --- |
| Paper PDF | https://arxiv.org/pdf/2405.10266v1 (accessed 2026-11-02) | `246fb354a2c41d24dd0b7b93062a851fd2ed56a47490aac92cbf89f8b1e026f4` | Targets and protocol |
| Published code | https://github.com/gulvarol/cslr2 | `5f3bd697e1d2e184a5882cbbd0da64ce9455fd28` | Training, retrieval test, CSLR extraction and evaluation |
| CSLR² checkpoint | Google Drive `1qyFHSFnxmy1rRGjlKEBfsjC8yt2kdalx` | Unpinned (mutable; re-uploaded 2024-07-02 per issue #1). Never acquired | Evaluation cross-check |
| `bobsl.zip` (annotations, metadata, CSLR CSVs) | Google Drive `13pp83GCoy1SVScvZRmNsoFxtNm8ogI7h` | Unpinned. Never acquired | Required by `config/paths/public.yaml` |
| `t5_checkpoint.zip` | Google Drive `1hxkb8KAC0sgSYKefOLue1wyO2fJmmqxT` | Unpinned. Never acquired | Frozen t5-large text encoder |
| BOBSL Video-Swin-T features (262G) | `thor.robots.ox.ac.uk/~vgg/data/bobsl/features/lmdb/feats_vswin_t-bs256_float16/` | Credentialed; never accessed | Model input |
| BOBSL RGB LMDB (1.5T) | `thor.robots.ox.ac.uk/~vgg/data/bobsl/videos/lmdb/rgb_anon-public_1962/` | Credentialed; never accessed | Visualisation; I3D rows |
| BOBSL paper | https://arxiv.org/pdf/2111.03635v1 | `0e75ee96bd548c0946a3a0737040d58ad7e44c0582e3b1e85b82a95789e29006` | SENT-VAL/SENT-TEST definitions, licence basis |
| BBC R&D BOBSL access terms | https://www.bbc.co.uk/rd/projects/extol-dataset (accessed 2026-11-02) | HTML `d4b006f6…` | First-party access and use restrictions |
| CSLR² data page and licence | https://gulvarol.github.io/cslr2/data.html | data.html `3cea110c…`, license.txt `65b0727d…` | BOBSL-CSLR CC BY 4.0 terms |
| transpeller (considered) | https://github.com/prajwalkr/transpeller | `dca7c8e3d463ca3c5909d9ee47c0ae8aeba4c0ff` | Not selected: Video-Swin-**S** checkpoint of [44]; Table 4 uses the authors' Video-Swin-**T** |

"Never acquired" means not downloaded before the data gate and not downloaded after the refusal (gate `data-bobsl-access`).

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

No target was produced. All 41 close as `not_produced` with reason `data_permission_blocked`, referencing gate `data-bobsl-access`. The 19 out-of-scope targets also reference gate `target-table4-rows`: their detail says they were excluded by the recorded scope decision and are independently blocked by data.

Every cell needs BOBSL. The CSLR cells are evaluated on CSLR-TEST, a BOBSL subset, and the retrieval cells on BOBSL SENT-VAL and SENT-TEST, all through the BOBSL Video-Swin-T features. **The released CSLR² checkpoint cannot be evaluated either, for the same reason:** it needs the same BOBSL features and the BOBSL-derived files in `bobsl.zip`.

| Target ID | Paper location | System | Dataset/split | Metric + version | Original | Reproduced | Difference | Terminal reason / evidence |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `t4-spot4-wer` | Table 4, CSLR | Subtitle-based spotting [4] | CSLR-TEST | WER ↓ | 93.8 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; out of scope (gate `target-table4-rows`) |
| `t4-spot4-miou` | Table 4, CSLR | Subtitle-based spotting [4] | CSLR-TEST | mIoU ↑ | 7.1 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; out of scope (gate `target-table4-rows`) |
| `t4-spot43-wer` | Table 4, CSLR | Subtitle-based spotting [43] | CSLR-TEST | WER ↓ | 85.5 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; out of scope (gate `target-table4-rows`) |
| `t4-spot43-miou` | Table 4, CSLR | Subtitle-based spotting [43] | CSLR-TEST | mIoU ↑ | 18.1 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; out of scope (gate `target-table4-rows`) |
| `t4-i3d2k-raw-wer` | Table 4, CSLR | ISLR I3D-2K [4] | CSLR-TEST | WER ↓ | 453.5 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; out of scope (gate `target-table4-rows`) |
| `t4-i3d2k-raw-miou` | Table 4, CSLR | ISLR I3D-2K [4] | CSLR-TEST | mIoU ↑ | 8.6 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; out of scope (gate `target-table4-rows`) |
| `t4-i3d2k-raw-f1-iou0.1` | Table 4, CSLR | ISLR I3D-2K [4] | CSLR-TEST | F1@0.1 ↑ | 10.6 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; out of scope (gate `target-table4-rows`) |
| `t4-i3d2k-raw-f1-iou0.25` | Table 4, CSLR | ISLR I3D-2K [4] | CSLR-TEST | F1@0.25 ↑ | 8.5 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; out of scope (gate `target-table4-rows`) |
| `t4-i3d2k-raw-f1-iou0.5` | Table 4, CSLR | ISLR I3D-2K [4] | CSLR-TEST | F1@0.5 ↑ | 5.4 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; out of scope (gate `target-table4-rows`) |
| `t4-i3d2k-pp-wer` | Table 4, CSLR | ISLR I3D-2K [4] † | CSLR-TEST | WER ↓ | 82.5 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; out of scope (gate `target-table4-rows`) |
| `t4-i3d2k-pp-miou` | Table 4, CSLR | ISLR I3D-2K [4] † | CSLR-TEST | mIoU ↑ | 18.1 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; out of scope (gate `target-table4-rows`) |
| `t4-i3d2k-pp-f1-iou0.1` | Table 4, CSLR | ISLR I3D-2K [4] † | CSLR-TEST | F1@0.1 ↑ | 25.7 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; out of scope (gate `target-table4-rows`) |
| `t4-i3d2k-pp-f1-iou0.25` | Table 4, CSLR | ISLR I3D-2K [4] † | CSLR-TEST | F1@0.25 ↑ | 23.8 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; out of scope (gate `target-table4-rows`) |
| `t4-i3d2k-pp-f1-iou0.5` | Table 4, CSLR | ISLR I3D-2K [4] † | CSLR-TEST | F1@0.5 ↑ | 16.0 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; out of scope (gate `target-table4-rows`) |
| `t4-i3d8k-pp-wer` | Table 4, CSLR | ISLR I3D-8K [43] † | CSLR-TEST | WER ↓ | 74.7 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; out of scope (gate `target-table4-rows`) |
| `t4-i3d8k-pp-miou` | Table 4, CSLR | ISLR I3D-8K [43] † | CSLR-TEST | mIoU ↑ | 27.0 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; out of scope (gate `target-table4-rows`) |
| `t4-i3d8k-pp-f1-iou0.1` | Table 4, CSLR | ISLR I3D-8K [43] † | CSLR-TEST | F1@0.1 ↑ | 36.7 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; out of scope (gate `target-table4-rows`) |
| `t4-i3d8k-pp-f1-iou0.25` | Table 4, CSLR | ISLR I3D-8K [43] † | CSLR-TEST | F1@0.25 ↑ | 35.4 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; out of scope (gate `target-table4-rows`) |
| `t4-i3d8k-pp-f1-iou0.5` | Table 4, CSLR | ISLR I3D-8K [43] † | CSLR-TEST | F1@0.5 ↑ | 26.7 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; out of scope (gate `target-table4-rows`) |
| `t4-swin8k-pp-wer` | Table 4, CSLR | ISLR Swin-8K [44] † | CSLR-TEST | WER ↓ | 71.7 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; in scope |
| `t4-swin8k-pp-miou` | Table 4, CSLR | ISLR Swin-8K [44] † | CSLR-TEST | mIoU ↑ | 30.1 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; in scope |
| `t4-swin8k-pp-f1-iou0.1` | Table 4, CSLR | ISLR Swin-8K [44] † | CSLR-TEST | F1@0.1 ↑ | 40.0 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; in scope |
| `t4-swin8k-pp-f1-iou0.25` | Table 4, CSLR | ISLR Swin-8K [44] † | CSLR-TEST | F1@0.25 ↑ | 37.9 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; in scope |
| `t4-swin8k-pp-f1-iou0.5` | Table 4, CSLR | ISLR Swin-8K [44] † | CSLR-TEST | F1@0.5 ↑ | 27.2 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; in scope |
| `t4-cslr2-wer` | Table 4, CSLR | CSLR² (Ours) † | CSLR-TEST | WER ↓ | 65.5 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; in scope |
| `t4-cslr2-miou` | Table 4, CSLR | CSLR² (Ours) † | CSLR-TEST | mIoU ↑ | 35.5 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; in scope |
| `t4-cslr2-f1-iou0.1` | Table 4, CSLR | CSLR² (Ours) † | CSLR-TEST | F1@0.1 ↑ | 47.1 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; in scope |
| `t4-cslr2-f1-iou0.25` | Table 4, CSLR | CSLR² (Ours) † | CSLR-TEST | F1@0.25 ↑ | 46.0 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; in scope |
| `t4-cslr2-f1-iou0.5` | Table 4, CSLR | CSLR² (Ours) † | CSLR-TEST | F1@0.5 ↑ | 37.1 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; in scope |
| `t4-infonce-sentval-t2v-r1` | Table 4, retrieval | InfoNCE | SENT-VAL (2K) | T2V R@1 ↑ | 38.9 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; in scope |
| `t4-infonce-sentval-v2t-r1` | Table 4, retrieval | InfoNCE | SENT-VAL (2K) | V2T R@1 ↑ | 39.2 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; in scope |
| `t4-infonce-senttest-t2v-r1` | Table 4, retrieval | InfoNCE | SENT-TEST (20K) | T2V R@1 ↑ | 19.5 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; in scope |
| `t4-infonce-senttest-t2v-r5` | Table 4, retrieval | InfoNCE | SENT-TEST (20K) | T2V R@5 ↑ | 35.1 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; in scope |
| `t4-infonce-senttest-v2t-r1` | Table 4, retrieval | InfoNCE | SENT-TEST (20K) | V2T R@1 ↑ | 18.9 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; in scope |
| `t4-infonce-senttest-v2t-r5` | Table 4, retrieval | InfoNCE | SENT-TEST (20K) | V2T R@5 ↑ | 33.8 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; in scope |
| `t4-cslr2-sentval-t2v-r1` | Table 4, retrieval | CSLR² (Ours) | SENT-VAL (2K) | T2V R@1 ↑ | 51.7 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; in scope |
| `t4-cslr2-sentval-v2t-r1` | Table 4, retrieval | CSLR² (Ours) | SENT-VAL (2K) | V2T R@1 ↑ | 50.2 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; in scope |
| `t4-cslr2-senttest-t2v-r1` | Table 4, retrieval | CSLR² (Ours) | SENT-TEST (20K) | T2V R@1 ↑ | 29.4 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; in scope |
| `t4-cslr2-senttest-t2v-r5` | Table 4, retrieval | CSLR² (Ours) | SENT-TEST (20K) | T2V R@5 ↑ | 45.2 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; in scope |
| `t4-cslr2-senttest-v2t-r1` | Table 4, retrieval | CSLR² (Ours) | SENT-TEST (20K) | V2T R@1 ↑ | 28.1 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; in scope |
| `t4-cslr2-senttest-v2t-r5` | Table 4, retrieval | CSLR² (Ours) | SENT-TEST (20K) | V2T R@5 ↑ | 44.9 | not produced | — | `data_permission_blocked`, gate `data-bobsl-access`; in scope |

Metric versions are those of gulvarol/cslr2 at `5f3bd697` (see Metric definitions). No original score is copied from earlier work, so no baseline was reproduced or skipped on that ground.

**Selected blocker: `data_permission_blocked`.** It lists the 22 in-scope targets. It is the earliest blocker and independently prevents every in-scope target: CSLR², InfoNCE and Swin-8K† all need BOBSL data, both for training and for evaluating any released checkpoint.

**Other blockers, reported separately:**

- **Compute (gate `compute-training-runs`):** resolved as moot. On 2026-11-02 the team lead approved funding both training runs in `#repro-sign-team-r`, without per-run limits on hours, GPU-hours, cost or restarts. The data refusal on 2026-11-24 made the approval moot before anything was measured or run.
- **Swin-8K† artifacts:** the scope condition (released artifacts sufficient without retraining) was never checked, because the check itself needs `bobsl.zip` and the BOBSL features. The `compute_budget_blocked` fallback in the scope decision therefore did not arise.
- **The 19 out-of-scope cells:** excluded by the scope decision and also blocked by the same refusal.

## How to repeat this

Nothing executable exists. The attempt closed at the data gate before any Dockerfile, `data.sh` or Modal entry point was written. The commands that were run:

```bash
# Assignment (2026-11-24); the export is the Reproduction Tracker download named above
python3 .agents/skills/reproduce-paper/scripts/ingest_candidate.py \
  reproductions-13195df56368306dc1c1f9733cebb8dd694c94af-2026-10-30.json \
  13195df56368306dc1c1f9733cebb8dd694c94af papers/raude-2024-cslr2/reproduction.json

# Read-only Modal checks (2026-11-02)
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh            # verified profile and workspace repro-sign
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh volume ls datasets /
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh volume ls huggingface-cache /
.agents/skills/reproduce-paper/scripts/check_modal_dataset.sh bobsl         # fails: path /bobsl/ does not exist
.agents/skills/reproduce-paper/scripts/check_modal_dataset.sh bobsl-cslr    # fails: path /bobsl-cslr/ does not exist

# Record check
python3 .agents/skills/reproduce-paper/scripts/validate_reproduction.py papers/raude-2024-cslr2
```

A team whose members are eligible for BOBSL under the BBC terms could resume from the target ledger, the source pins and the compute estimate recorded here.

## Data provenance and permissions

| Dataset | Version/subset/splits | Source and access date | License/permission and cloud-use basis | Path in Volume `datasets` | Counts / manifest / checksum | Deviations |
| --- | --- | --- | --- | --- | --- | --- |
| BOBSL | Video-Swin-T feature LMDB, pseudo-labels, automatic and manual subtitles. SENT-VAL_H, SENT-TEST | BBC R&D terms page, 2026-11-02. Oxford page unreachable. **Access refused 2026-11-24; no data obtained** | BBC-owned, non-commercial academic research only, per-person signed Permission to Use. **Permission denied for REPRO-SIGN**, so `cloud_processing_allowed` is false | `bobsl/`: intended path only; **it does not exist** | SENT-VAL_H 1,973, SENT-TEST 20,870, from the BOBSL paper Table 4. **Split definition only, no data obtained:** each split's file entry is the BOBSL paper PDF (`0e75ee96…`), not a data file | None; see `sent-split-counts`, `readme-retrieval-test-size` |
| BOBSL-CSLR | Processed lexical-gloss CSVs `cslr/manual_glosses_extended_boundaries_fix_alignment/0/{train,val,test}` inside `bobsl.zip` | gulvarol.github.io/cslr2/data.html, 2026-11-02. **Not acquired; no data obtained** | CC BY 4.0 for the text annotations, but they ship only inside `bobsl.zip` with BOBSL-licensed files and are unusable without BOBSL features. **Permission denied**, so `cloud_processing_allowed` is false | `bobsl/cslr/`: intended path only; **it does not exist** | Test 4,518, val 1,153, train 1,463 sentences, from the CSLR² paper Table A.2. **Split definition only, no data obtained:** each split's file entry is the CSLR² paper PDF (`246fb354…`), not a data file | None; see `cslr-test-size` |

### Data gate (`data-bobsl-access`, resolved 2026-11-24: access denied)

On 2026-11-02, `check_modal_dataset.sh bobsl` and `check_modal_dataset.sh bobsl-cslr` both failed because the paths do not exist. The BBC terms are per person and forbid redistribution, so the gate went to the team. What followed, as reported by the operator:

- **2026-11-02:** REPRO-SIGN sent the BBC BOBSL Permission to Use form, with a cover email, to the BBC BOBSL dataset contact. A follow-up the same day named this paper as the intended use.
- **2026-11-24:** a reminder was sent.
- **2026-11-24:** the BBC R&D contact named in the form refused access.

**Reasons given, paraphrased:**

- The BBC does not allow any involvement of commercial organisations in projects that use BOBSL, citing past disputes over preferential access, and REPRO-SIGN includes a commercially affiliated collaborator.
- The BBC also does not permit publishing model weights, which the request had marked as optional.

The study accepted the refusal. No BOBSL file, `bobsl.zip`, CSLR² checkpoint or T5 bundle was downloaded at any point, and no other route to the data was attempted. The correspondence is held by the study and is not quoted here.

**Scope of the refusal.** It is a constraint from the composition of the REPRO-SIGN consortium, not evidence that the paper's data is unavailable in general. BOBSL remains offered to eligible non-commercial academic researchers under the BBC terms.

### Split entries without data

This is the study's first record with pipeline `blocked_on_data` (checked against main `597fab7`).

No data was obtained, so each split's file entry points to the split's definition in the paper:

- The path reads "split definition only, no data obtained", followed by the paper and table that define the split.
- The SHA-256 is that paper's PDF.
- `sample_count` comes from the same table.
- `modal_path` keeps the intended `bobsl` path, with a note that it does not exist.

## Compute plan (gate `compute-training-runs`, resolved 2026-11-24: moot)

The in-scope cells needed two full training runs of the official code: CSLR² for 11 cells and the InfoNCE baseline for 6 cells. The gate was opened on 2026-11-02 before any measurement, at the operator's request, so that it reached the team together with the data gate.

On 2026-11-02 the team lead approved funding both training runs in `#repro-sign-team-r`, without setting per-run limits on hours, GPU-hours, cost or restarts. The approval became moot on 2026-11-24 when the data gate closed, because the representative preflight and both runs need BOBSL. Nothing was measured and no compute was spent. The estimate is kept as recorded. It is paper-reported, not measured:

| Item | Figure | Basis |
| --- | --- | --- |
| CSLR² training, 20 epochs | ≤ 20 h on 4 × V100-32GB, about 80 V100 GPU-hours | Paper App. A and cslr2 README |
| InfoNCE training | about 80 V100 GPU-hours | Assumed equal to CSLR²; the paper does not state it |
| Total | up to about 160 V100 GPU-hours | Above the 24 GPU-hour autonomous ceiling per run |
| Modal GPU cost, both runs | about USD 400 (A100 80GB) to USD 632 (H100) | Upper bracket: no speed-up over V100 assumed; modal.com/pricing read 2026-11-02 |
| CPU and memory | about USD 60 per run | 40 physical cores, about 128 GiB, 20 h |
| Storage | about 280 GiB in `datasets` | Features 262G, `bobsl.zip` 15G decompressed, T5 1.4G |

**Configuration.** Exact replay needs 4-process DDP on one node, for example one Modal container with 4 GPUs at batch 128 per process. The training loop all-gathers sentence embeddings, so SentRet negatives span the global 512 batch. SignRet, however, is computed per process over 128 subtitles, so a single GPU at batch 512 would change the sign-level negatives.

## Environment and patches

None. No container was built and no patch was written.

The planned environment was the root `reproduction` base image with the cslr2 dependencies. The upstream environment is Python 3.9.16, PyTorch 1.12.1 with CUDA 11.6, hydra-core 1.3.2, sentence-transformers 2.2.2, lmdb, and nltk wordnet.

Implementation notes recorded in `reproduction.json.protocol_notes`, which no run reached:

- **No SENT-VAL entry point.** Upstream exposes no command for SENT-VAL retrieval of a fixed checkpoint; `test=True` evaluates only SENT-TEST.
- **RGB LMDB dependency.** `main.py test=True` references `rgb_lmdb_env`, which is only opened when `vis=True`, and `vis=True` opens the 1.5T RGB LMDB.
- **Batch size.** The paper's batch of 512 corresponds to 128 per process in `config/dataloader` across 4 processes.
- **Episode count.** Sec. 4 says the CSLR-TEST annotations cover all 35 SENT-TEST episodes; the BOBSL paper lists 36.

## Execution evidence

No runs. No Modal app was launched, and no GPU or CPU compute was used.

## Guesses and deviations

| ID | Kind | Description | Paper says | This attempt used | Basis | Effect | Scope | Status | Evidence | Related IDs | Affects | Dataset IDs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `sent-split-counts` | `paper_internal_conflict` | The paper gives the SENT-VAL and SENT-TEST sizes in opposite order in two places. | Sec. 5.1: SENT-VAL 20,870 and SENT-TEST 1,973 sentences. Table 4 column headers (SENT-VAL (2K), SENT-TEST (20K)) and App. D.2: the opposite. | Target splits follow Table 4 and App. D.2: SENT-VAL 1,973 and SENT-TEST 20,870 sentences. | Table 4 is the reported table, and the BOBSL paper (Table 4: SENT-VAL_H 1,973; SENT-TEST 20,870) agrees with it. | Labels the retrieval targets only; no value depends on it because none was produced. |  | Adopted in the target ledger on 2026-11-02. No run reached it: the attempt closed at the data gate before any download, build or run. | arXiv 2405.10266v1 Sec. 5.1, Table 4 (p. 8), App. D.2; BOBSL paper arXiv 2111.03635v1 Table 4 |  | `evaluate` | `bobsl` |
| `cslr-test-size` | `paper_internal_conflict` | The paper gives two sizes for CSLR-TEST. | Sec. 5.1: CSLR-TEST corresponds to 4,950 test subtitles. Tables 1 and A.2: 4,518 sentences. | Target splits cite both counts; the dataset record uses Table A.2 (4,518). The count was to be checked against the released test CSVs after acquisition. | Only the released CSVs can show which count the reported numbers used. | None on any value: no CSLR value was produced. |  | Unresolved. No run reached it: the attempt closed at the data gate before any download, build or run. | arXiv 2405.10266v1 Sec. 5.1, Table 1, Table A.2 |  | `evaluate` | `bobsl-cslr` |
| `readme-retrieval-test-size` | `paper_code_conflict` | The released code's README and the paper give different sizes for the retrieval test set. | Paper Table 4 and App. D.2: SENT-TEST (20K), 20,870 manually aligned sentences. gulvarol/cslr2 README: '25K manually aligned test set'. | Targets are labelled with the paper's SENT-TEST (20,870 sentences). The size of the set that main.py test=True loads was to be counted after acquisition. | The paper defines the reported split, and the BOBSL paper agrees with it. | None on any value: no retrieval value was produced. |  | Unresolved. No run reached it: the attempt closed at the data gate before any download, build or run. | gulvarol/cslr2@5f3bd697e1d2e184a5882cbbd0da64ce9455fd28 README.md; arXiv 2405.10266v1 Table 4, App. D.2 |  | `evaluate` | `bobsl` |
| `checkpoint-selection` | `paper_code_conflict` | The paper keeps the best validation checkpoint; the released training loop overwrites model_best.pth every epoch. | App. A: keep the checkpoint with the best T2V R@1 on a manually aligned subset of validation subtitles. gulvarol/cslr2 main.py passes best_t2v=-1 to log_and_save every epoch, so model_best.pth is the last epoch, and validation retrieval uses a random 2,000-sentence subset of the automatic subtitles of the validation episodes. |  |  | Would decide which checkpoint every retrained CSLR² and InfoNCE value comes from. No training run happened. |  | Unresolved; it was to go to a protocol gate before retraining. No run reached it: the attempt closed at the data gate before any download, build or run. | arXiv 2405.10266v1 App. A; gulvarol/cslr2@5f3bd697e1d2e184a5882cbbd0da64ce9455fd28 main.py (log_and_save called with best_t2v=-1) |  | `train`; `evaluate` | `bobsl` |
| `postprocessing-tuning-split` | `paper_code_conflict` | The paper and the released code tune the CSLR post-processing threshold and repetition count on different splits. | Sec. 5.2: θ and m are tuned 'on unseen manually annotated CSLR sequences from the validation set'. gulvarol/cslr2 frame_level_evaluation.py searches them on `0/train/*.csv` unless test_search is set. |  |  | Would affect every † CSLR value (CSLR² and Swin-8K†). No evaluation happened. |  | Unresolved; it was to go to a protocol gate before evaluation. No run reached it: the attempt closed at the data gate before any download, build or run. | arXiv 2405.10266v1 Sec. 5.2; gulvarol/cslr2@5f3bd697e1d2e184a5882cbbd0da64ce9455fd28 frame_level_evaluation.py |  | `evaluate` | `bobsl-cslr` |

## Noncompliance

None.

## Attempts, failures, and dead ends

- **BOBSL page.** It could not be fetched (TLS reset, HTTP 503; WebFetch failed; Wayback rate-limited). The BOBSL arXiv paper was used for the licence basis and split counts instead.
- **Project licence link.** The README's `imagine.enpc.fr` licence URL returns 404. The redirected GitHub Pages copy was used.
- **BOBSL access request (2026-11-02 to 2026-11-24).** Refused by the BBC; see the data gate above. This is the terminal blocker. No workaround was attempted.

## Candidate flags, ethics, and human evaluation

- **Queue flags.** The queue reports no human evaluation, no ethical concerns and no copied scores. The paper confirms the absence of human evaluation in Table 4, and the copied-score reading is above.
- **Signer data.** BOBSL contains identifiable BSL interpreters from BBC broadcasts. No BOBSL data was obtained or processed, and no participant interaction was involved.

## Author and team contact

- **Authors:** none contacted.
- **Operator:** decided the Table 4 scope at gate `target-table4-rows` on 2026-11-02. On 2026-11-24 the operator reported the BBC's refusal recorded at gate `data-bobsl-access`.
- **BBC (dataset owner, not an author):** REPRO-SIGN requested BOBSL access through the BBC Permission to Use process on 2026-11-02 and was refused on 2026-11-24. This changed the outcome: the attempt closes as `blocked_on_data`. The agents took no part in the correspondence.
- **`#repro-sign-team-r`:** on 2026-11-02 the team lead approved funding both training runs (CSLR² and InfoNCE) at gate `compute-training-runs`, without per-run limits on hours, GPU-hours, cost or restarts. The approval became moot on 2026-11-24 with the data refusal; nothing was run.
