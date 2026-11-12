# Manjur, Biswas and Gurbuz (2025): multimodal video and radar fusion, reproduction

**Paper ID:** `36e6dc7d50d396e2f7aef5c584228fdcefde9020`

**Citation:** Sultan Mohammad Manjur, Sabyasachi Biswas, Ali C. Gurbuz. *A Multimodal Video and Radar Fusion Framework for High-Accuracy Isolated Sign Language Recognition.* In 2025 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), pp. 5061–5070, 2025. doi:10.1109/ICCVW69036.2025.00526

**Paper:** https://doi.org/10.1109/ICCVW69036.2025.00526 · **Code/artifacts:** https://github.com/impressncsu/ICCV2025_MSLR (pinned `7dd73c4e67b2959a95f828c103269723528b9271`), plus the authors' logit files and weights on Google Drive (see Source provenance)

**Preference level:** 1

The preference level is provisional; see `status.preference_level_basis`. Published code exists for every system. SL-GCN training will need the cited SAM-SLR training configs, because the paper's repository ships only test configs.

**Pipeline status:** not yet set. Training protocol A1 (published code) was chosen by the operator on 2026-11-09. The attempt-wide compute ceiling, first CHF 300 / 250 L4 GPU-hours, was raised to CHF 450 / 380 on 2026-11-10, and this paper may use at most 30 GPUs at once. Radar, video and SL-GCN training finished on 2026-11-09. The SL-GCN bone, joint-motion and bone-motion models turned out to be trained with a normalization (`is_vector: True`) that contradicts the authors' published configs (dead end, see Attempts). Following gate `slgcn-retrain-settings` (R1, operator 2026-11-10), they were retrained with `is_vector: False`. The first retrain attempt was interrupted after 120–127 of 250 epochs per fold (`slgcn-retrain-full-001`); the second resumed it and finished on 2026-11-11 (`slgcn-retrain-full-002`). SL-GCN inference, fusion and the 11 Kaggle submission files come next. No target has a terminal result yet.

**Numerical agreement:** `not_assessed` — no target has been produced.

**Modal environment:** every Modal command for this paper runs in a Restricted Environment of the `repro-sign` workspace, written `$MODAL_ENV` throughout. Its literal name embeds an operator account name and is withheld pending the operator's confirmation; the project lead holds it.

**Attempt dates:** 2026-11-04 (intake, target ledger, gates), 2026-11-06 (data, preprocessing, preflight), 2026-11-09 (protocol and compute decisions, full runs), 2026-11-10 (`is_vector` dead end, ceiling increase, retrain gate)

## Reproduction agents

| Agent ID | Model and version | Agent application | Contribution | Attribution evidence / unknowns |
| --- | --- | --- | --- | --- |
| `claude-opus-5-5-claude-code` | Claude Opus 5.5 (`claude-opus-5-5`) | Claude Code 2.1.318 | 2026-11-04: assignment intake, target ledger, source search and pinning, Kaggle metadata and leaderboard reading, read-only Modal Volume inspection, the CPU logit audit `audit-author-logits-001`, target and data gates | The session system context names the model and its ID, and `claude --version` returned 2.1.318. Executed `audit-author-logits-001`. |
| `claude-opus-5-5-claude-code-2-1-319` | Claude Opus 5.5 (`claude-opus-5-5`) | Claude Code 2.1.319 | From 2026-11-06, same conversation: recorded the gate decisions, created the `$MODAL_ENV` Volumes, wrote `scripts/data.sh` and `scripts/modal_app.py`, and executed every Modal run from `data-populate-001` on. **It inferred the SL-GCN training configuration** (see below). | The session system context names the model, and `claude --version` returned 2.1.319 on 2026-11-06 and 2026-11-10 (the application was updated between 2026-11-04 and 11-06). |

**Attribution of the SL-GCN training configuration.** The reproduction agent `claude-opus-5-5-claude-code-2-1-319` inferred the SL-GCN training configuration, including `is_vector: True` for bone, joint-motion and bone-motion, from SAM-SLR's original training configs. The authors' published fold configs contain a `train_feeder_args` section with `is_vector: False` for all four streams. The agent did not surface it. Protocol option A1, as the agent presented it to the human reproducer, described the training config only as "SAM-SLR train config" and did not disclose this contradicting evidence. The human approval of A1 was given on that basis. The inference was the agent's, not the human reproducer's. Why the section was missed, and what should have happened, is under Attempts, failures, and dead ends.

## Scope and target contract

### Assignment intake

The assignment is a single `reproductions`-collection export, `reproductions-36e6dc7d50d396e2f7aef5c584228fdcefde9020-2026-11-04.json` (SHA-256 `0e2d198f…8401e4`), plus the publisher PDF and a local download of the authors' logit folder. It is recorded as `assignment.kind = direct_user_request`, following `papers/pu-2024-siformer`:

- The paper record is nested under `expand.paper` with `status: final`.
- The top-level `status: in_progress` belongs to the reproduction task.
- There is no `confirmation` field, and the `id` values are queue record IDs, not the paper ID.

The export was not edited and `ingest_candidate.py` was not run. Every email address in the stored record is replaced with `[redacted]`.

The queue's `what_to_reproduce` is "Tables 1, 2 and 3". The operator excluded the internal 80/20-split numbers in the Section 5 prose (88.18, 89.68, 92.50, 90.70, 95.83) unless they add them later.

### The 22 targets

All three tables report top-1 accuracy (%) of the authors' own systems on the official MultiMeDaLIS challenge splits, with one "Val Top-1" and one "Test Top-1" per row. None is copied from earlier work: the queue flag `copied_scores: no` was checked against the tables and text. Where the prose gives a third decimal, it is recorded as `prose_value`.

| Table | System (experiment) | Val target | Val | Test target | Test |
| --- | --- | --- | ---: | --- | ---: |
| 1 | CNN 2D on µ-D (`radar-mdcnn`) | `t1-radar-val` | 92.61 (92.607) | `t1-radar-test` | 93.03 (93.029) |
| 1 | ResNet(2+1)D on video (`video-r2plus1d`) | `t1-video-val` | 94.01 (94.013) | `t1-video-test` | 93.74 (93.737) |
| 2 | SL-GCN joint (`slgcn-joint`) | `t2-joint-val` | 99.31 | `t2-joint-test` | 99.07 |
| 2 | SL-GCN bone (`slgcn-bone`) | `t2-bone-val` | 99.11 | `t2-bone-test` | 99.04 |
| 2 | SL-GCN joint motion (`slgcn-joint-motion`) | `t2-jm-val` | 98.62 | `t2-jm-test` | 98.61 |
| 2 | SL-GCN bone motion (`slgcn-bone-motion`) | `t2-bm-val` | 98.66 | `t2-bm-test` | 98.56 |
| 2 | Skeletal ensemble 0.5/0.1/0.3/0.1 (`slgcn-ensemble`) | `t2-ensemble-val` | 99.34 | `t2-ensemble-test` | 99.29 |
| 3 | Radar + Video 0.5/0.5 (`fusion-radar-video`) | `t3-radar-video-val` | 97.19 (97.188) | `t3-radar-video-test` | 97.42 |
| 3 | Radar + Skeletal 0.25/0.75 (`fusion-radar-skeletal`) | `t3-radar-skeletal-val` | 99.37 (99.365) | `t3-radar-skeletal-test` | 99.44 |
| 3 | Skeletal + Video 0.75/0.25 (`fusion-skeletal-video`) | `t3-skeletal-video-val` | 99.50 (99.501) | `t3-skeletal-video-test` | 99.55 |
| 3 | Skeletal + Video + Radar 0.5/0.25/0.25 (`fusion-all`) | `t3-all-val` | 99.61 (99.614) | `t3-all-test` | 99.71 (99.708) |

### What "Val Top-1" and "Test Top-1" are

The paper states that the official validation and test labels were withheld from participants (Sections 4.3 and 5). The evidence shows the tabled values are Kaggle leaderboard scores of the competition `iccv-mslr-2025-track-2`:

- **Headline match.** The public Kaggle leaderboard, read on 2026-11-04, lists team IMPRESS NCSU (submission 45543069) at **public 0.99614** and **private 0.99708**. These are exactly the paper's three-modality values. Another team, CPAMI (UW), has public 0.99365 and private 0.99444, which equal the val 99.37 and test 99.44 reported in its own paper (FusionEnsemble-Net).
- **One combined submission.** The competition has one 11,970-row solution (validation 4,410 + test 7,560), `leaderboardPercentage` 37 (4,410 / 11,970 = 36.8%), and `scoreTruncationNumDecimals` 5.
- **Arithmetic fit.** Among denominators from 3,000 to 6,000, only N = 4,410 makes every published val value a 5-decimal-truncated k/N. N = 7,560 does the same for every published test value except the prose 93.737. No k/7,560 truncates to 93.737, while 7,087/7,560 = 0.93743 is consistent with Table 1's 93.74. The table value is the target.
- **Conclusion (inferred, not stated by the host).** Val is the public leaderboard over the 4,410 validation rows; Test is the private leaderboard over the 7,560 test rows.

**Split sizes in the paper text (a paper-text discrepancy, not a data mismatch).** Section 5 says the official validation and test sets have 35 and 70 samples per class. The validation figure is consistent: 126 × 35 = 4,410. The test figure is not:

- **Row counts.** The competition data, the authors' released logits and every prediction CSV in their repository (`official-code` at `7dd73c4`) have 4,410 val and 7,560 test rows. The three `*_all.csv` files have 11,970 = 4,410 + 7,560 rows. 126 × 70 would be 8,820.
- **Per class.** 7,560 / 126 = 60. The test labels are concealed, so per-class counts can't be checked directly. The authors' own final test predictions (`Final_prediction/Final_LIS_prediction_test.csv`) give 58–62 per predicted class, median 60; their val predictions give 32–37, median 35. The test set therefore has about 60 per class, not 70.
- **Organisers' summary.** The SignEval 2025 summary (Section 4.2) states only the totals: 12,600 train, 4,410 val and 7,560 test.
- **Corpus size.** Section 3.1 gives 25,830 instances for the corpus, which equals 126 × (100 + 35 + 70) = 126 × 205. The paper's 70 may be derived from that figure (205 − 100 − 35), but this is not established. The 25,830 is not treated as wrong: the same summary section describes the full corpus as 205 performances of each of the 126 items (25,830). The released Track-2 partition has 24,570 (126 × 195).

The only consequence is the test denominator: 7,560, not 8,820. The data used here match the competition release and the authors' own files.

### Can the tabled scores be obtained today?

No public release of the 2025 Track-2 validation or test labels was found. The search covered Hugging Face, Zenodo, GitHub repository and code search, web search, the challenge site, the organisers' summary paper and two later papers on the data. A 2026 participant's copy of the CVPR MSLR 2026 host Q&A quotes the host: the 2025 RGB/RDM data may not be used in the 2026 challenge, and sessions were reorganized across releases to prevent cross-release matching. That makes a label release unlikely.

A comparable value therefore needs one of these:

- Kaggle scoring of new predictions, through an account that has joined the competition and with late submissions open. Anonymous metadata shows deadline 2025-06-23 and `locked: true`, and does not say whether late submissions are accepted.
- Labels released by the organisers.

**Decision (gate `target-eval-route`, resolved 2026-11-05).** The operator chose Kaggle late submissions (alternative A). The operator reports that late submission is open, that they accepted the competition rules (date corrected by the operator to 2026-11-06), and that `kaggle competitions files -c iccv-mslr-2025-track-2` lists the data with their account. The workflow:

- The agent prepares one CSV per table row, each covering all 11,970 val+test ids (11 CSVs), and records their SHA-256.
- The operator submits at most two per day and returns each public and private score with the submission URL or screenshot and date.

**Metric caveat.** Every tabled value, and every value this attempt will report, is an external leaderboard score computed by Kaggle against a concealed solution. The labels cannot be inspected, verified or hashed by this study. Only the submitted prediction files (by SHA-256) and the returned scores (by URL or screenshot and date) are recorded. Repeating the attempt requires a Kaggle account that has joined `iccv-mslr-2025-track-2` and accepted its rules, and the competition must still accept late submissions.

**Alternative B was not requested.** B was a Team S request for the organisers' labels. Kaggle scoring uses the same hidden solution, public/private partition and 5-decimal truncation that produced the paper's numbers, so it needs no inferred val/test-to-leaderboard mapping. The organisers deliberately keep the labels concealed, so a request was unlikely to succeed and would have added an external contact without improving comparability.

## Source provenance

| Artifact | Canonical source | Pinned revision / SHA-256 | Role |
| --- | --- | --- | --- |
| Paper PDF | https://doi.org/10.1109/ICCVW69036.2025.00526 | `2fb85a35b8f35acbd7b3e67f060d0b73f29eda99fd054e0b2f2a634ae3633721` | Target and protocol. Operator-supplied publisher PDF, not committed |
| Published code | https://github.com/impressncsu/ICCV2025_MSLR | `7dd73c4e67b2959a95f828c103269723528b9271` | Selected. Linked in the paper. One commit (2025-08-15), no tags or submodules |
| Code mirror | https://github.com/sultanmanjur/ICCV_2025_MSLR_COMPETITION | `7dd73c4e67b2959a95f828c103269723528b9271` | Same commit as the published code |
| Earlier author snapshot | https://github.com/smanjur2505/ICCV2025_MSLR | `5ee6d5c2d9bea0db3635fabbfe693a8aa826492a` | Considered. Its per-system `*_all.csv` files hold 11,970 val+test rows (`id,pred`), the format of a combined Kaggle submission; its `Final_LIS_prediction.csv` is test-only (corrected 2026-11-06) |
| Authors' logits | Google Drive folder `1JpJLOxWdzL_dvkrKkHJydguEkmRt_RmB` (publicly listable 2026-11-04) | zip `2e691817086be6205ec175aa3c06420b69aa7e17d4ed110e5ba7e5b3083b3cf7`; per-file hashes in `sources` | Evidence only (audit run below), not committed |
| Radar/video final weights | Google Drive folder `1oSWKC3bShKOiI5yynSaQuwl1nnKQfRKf` | unpinned (HTTP 401 to anonymous listing) | Not public; not needed for retraining |
| SL-GCN fold weights | Google Drive file `14kS0_yFGE0217vzck71Ni_Z2_TsEFHvV` (`save_models.zip`) | unpinned, not downloaded | Not needed for retraining |
| HRNet whole-body weights | Google Drive file `1f_c3uKTDQ4DR3CrwMSI8qdsTKJvKVt7p` (`hrnet_w48_coco_wholebody_384x288-6e061c6a_20200922.pth`) | to be hashed at download | Pose extraction |
| SAM-SLR (cited SL-GCN basis) | https://github.com/jackyjsy/CVPR21Chal-SLR | `de6c53a6fbbc08a5d9a728e33614a4541836714c` | SL-GCN training configs and labelled data generator absent from the paper's repository |
| Kaggle competition | https://www.kaggle.com/competitions/iccv-mslr-2025-track-2 (id 102067) | metadata and leaderboard read 2026-11-04 | Data distribution and the only scorer of the concealed labels |
| SignEval 2025 summary | CVF open access (Luqman et al., ICCVW 2025) | `dc02c5ceaa4a28b226c30fc8b05697ddeda463d35e44f2dea4c20def24b14d63` | Split sizes and concealed labels |
| 2026 participant notes | https://github.com/SheikhGalib/CVPR-MSLR-2026-Track-2-kaggle | `207a568a1c9c9e2fa1d93ebfee205edc9b03838c` | Secondary copy of the 2026 host Q&A |
| FusionEnsemble-Net, CAST | arXiv 2508.09362, arXiv 2605.08663 | `60dfbbf2…942e44`, `38612556…d0d9a5` | Considered; corroborate the val/test = public/private mapping (FusionEnsemble-Net); no label release |

The full search log is in `reproduction.json.source_search`.

## Results

| Target ID | Paper location | System | Dataset/split | Metric + version | Original | Reproduced | Difference | Terminal reason / evidence |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| all 22 targets | Tables 1–3, p.7 | see table above | MultiMeDaLIS Track-2 val (4,410) / test (7,560) | Kaggle Accuracy Score, metric version 17210 | see table above | not yet produced | — | No terminal result yet. Radar and video training and their val/test inference are done (`predict-full-001`). The SL-GCN streams trained with `is_vector: True` (`slgcn-train-full-002`) produce no targets (dead end). Bone, joint-motion and bone-motion were retrained with `is_vector: False` (`slgcn-retrain-full-002`, gate `slgcn-retrain-settings` R1). Then come SL-GCN inference, fusion and the 11 submission files, which the operator submits to Kaggle, returning each public/private score |

No targets are copied baselines. No scores have been produced, so no numerical agreement is assessed.

## How to repeat this

All commands run from a checkout of this repository. `$MODAL_ENV` is this paper's Restricted Environment, and every Modal command goes through the wrapper. The Modal app `scripts/modal_app.py` pins the upstream commit and applies `patches/*.patch` in order. It also pins SAM-SLR `de6c53a6…` and the HRNet checkpoint (SHA-256 checked), and defines two images:

- **TF image:** Python 3.11, TensorFlow 2.15.1 (Keras 2.15.0).
- **PyTorch image:** the project base image `ghcr.io/sign-language-processing/reproduction@sha256:305b6165…`.

```bash
W=.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh
APP=papers/manjur-2025-video-radar-fusion/scripts/modal_app.py
# 1. data (idempotent; needs a Kaggle account that accepted the rules, token in Secret kaggle-credentials)
$W run --detach --env "$MODAL_ENV" $APP::data --mode populate
# 2. symlink work tree, train label CSV, val/test id lists (idempotent)
$W run --env "$MODAL_ENV" $APP::worktree
# 3. authors' preprocessing, sharded: micro-Doppler PNGs and resized .h5 videos
$W run --env "$MODAL_ENV" $APP::prep_all --task radar
$W run --env "$MODAL_ENV" $APP::prep_all --task video
```

Full runs so far (protocol A1). Each was launched detached; run IDs as in Execution evidence:

```bash
# 4. HRNet keypoints for all 24,570 videos (64 shards, 4 processes per L4)
$W run --detach --env "$MODAL_ENV" $APP::pose_all --procs 4
# 5. radar and video models, the authors' scripts as published (+ patch 0002 for the video save)
$W run --detach --env "$MODAL_ENV" $APP::tf_train --model radar --run-id train-radar-full-001
$W run --detach --env "$MODAL_ENV" $APP::tf_train --model video --run-id train-video-full-001
# 6. SL-GCN fold arrays, then 4 streams x 5 folds x 250 epochs, one L4 each
$W run --detach --env "$MODAL_ENV" $APP::slgcn_data --run-id slgcn-full-001
$W run --detach --env "$MODAL_ENV" $APP::slgcn_all --run-id slgcn-full-001 --num-epoch 250 --per-gpu 1
#    after an interruption, continue every fold from its last checkpoint:
$W run --detach --env "$MODAL_ENV" $APP::slgcn_all --run-id slgcn-full-001 --num-epoch 250 --per-gpu 1 --resume
```

Inference so far:

```bash
# 7. radar and video inference on all val and test samples (authors' prediction scripts)
for SPLIT in val test; do
  $W run --detach --env "$MODAL_ENV" $APP::tf_predict --split $SPLIT \
    --radar-h5 /outputs/runs/train-radar-full-001/relational_md_cnn_enhanced_final_valAcc_0.9099.h5 \
    --video-h5 /outputs/runs/train-video-full-001/r2p1d_final.h5 --out-id predict-$SPLIT-full-001
done
# 8. SL-GCN inference with the authors' 20 fold test configs and run_fusion.py (our selected checkpoints)
$W run --detach --env "$MODAL_ENV" $APP::slgcn_test --run-id slgcn-full-001 --split val
$W run --detach --env "$MODAL_ENV" $APP::slgcn_test --run-id slgcn-full-001 --split test
#    conditional evidence only (option V1): test-time is_vector True for bone, jm, bm
$W run --env "$MODAL_ENV" $APP::slgcn_test --run-id slgcn-full-001 --split val --vector-streams bone,jm,bm
$W run --env "$MODAL_ENV" $APP::slgcn_test --run-id slgcn-full-001 --split test --vector-streams bone,jm,bm
```

SL-GCN retrain (gate `slgcn-retrain-settings`, R1), on the same folds and keypoints:

```bash
# 9. bone, jm, bm x 5 folds, is_vector False in both feeders, 15 L4s (within the 30-GPU limit)
$W run --detach --env "$MODAL_ENV" $APP::slgcn_all --run-id slgcn-full-002 --num-epoch 250 --per-gpu 1 \
  --streams bone,jm,bm --data-run slgcn-full-001 --authors-is-vector
# enforce its ceilings: stop the retrain app (attempt 1: ap-BLTNkE5vRv7KdWGHgpCoBi) at the guard_deadline time unless it has finished
$W run --detach --env "$MODAL_ENV" $APP::guard --target-app ap-BLTNkE5vRv7KdWGHgpCoBi \
  --deadline-utc 2026-11-10T22:01:50Z --env-name "$MODAL_ENV" --poll-seconds 120
# attempt 2 after the preemption cascade: the same command with --resume (each fold continues
# from its last checkpoint with an eval record; a preempted fold's automatic retry does the same)
$W run --detach --env "$MODAL_ENV" $APP::slgcn_all --run-id slgcn-full-002 --num-epoch 250 --per-gpu 1 \
  --streams bone,jm,bm --data-run slgcn-full-001 --authors-is-vector --resume
# its guard (app ap-Or6VA4QMCT2DYpKBSlMDaI, started 19:54:25Z), also capping the running tasks at 15 L4s + the orchestrator
$W run --detach --env "$MODAL_ENV" $APP::guard --target-app ap-Or6VA4QMCT2DYpKBSlMDaI \
  --deadline-utc 2026-11-11T00:46:35Z --env-name "$MODAL_ENV" --poll-seconds 120 --max-tasks 16
```

Its inference, fusion and submission building (`fuse`, preflighted) will be added when they run.

**Stop-policy enforcement (from 2026-11-10).** Every declared ceiling is enforced mechanically, not only recorded.

- **GPU runs.** Right after a GPU run starts, the Modal function `guard` is launched as its own detached app. It stops the run's app at a deadline unless the run has finished.
- **The deadline.** `guard_deadline` computes it as start + min(wall ceiling, GPU-hour ceiling ÷ GPUs, cost ceiling ÷ (GPUs × all-in USD per container-hour)). This bounds all three ceilings only while at most that many GPU containers run.
- **Container count.** From `slgcn-retrain-full-002` on, the guard also takes `--max-tasks`, the planned number of containers (for example 15 L4s + 1 orchestrator = 16). It stops the app when two consecutive polls see more, so the deadline's premise is enforced, not assumed (test `pf-guard-tasks-001`).
- **Rates.** USD 1.15 for SL-GCN training (measured 1.123) and USD 1.50 for `slgcn_test` and `tf_predict` (measured 1.43–1.44).
- **Backup.** A local watcher through the wrapper stops the app 10 minutes after the deadline, and alerts if the guard stops first.
- **CPU runs.** These rely on their function timeout, with ceilings declared to match it.
- **Mechanism.** The guard calls the same `AppList`/`AppStop` RPCs as `modal app list` and `modal app stop` (modal 1.6.2 internals), because the Modal CLI cannot run inside containers. Tests: `pf-guard-001`, `pf-guard-002`, `pf-guard-sync-001`, `pf-guard-tasks-001`.
- **History.** Before this, only function timeouts bounded runs, which limited wall time but not GPU-hours or cost. That is how `predict-full-001` overran.

**GPU concurrency limit.** From 2026-11-10 this paper uses at most 30 GPUs at once across all its jobs, because REPRO-SIGN shares a 50-GPU Modal limit with other reproductions. `MAX_GPUS = 30` in `scripts/modal_app.py` caps every GPU function (`max_containers`), and `slgcn_all` refuses more than 30 jobs. Two GPU fan-outs are never launched together. `pose_all`'s 64 shards would now queue in at most 30 containers, roughly doubling its ~75-minute wall time if it ever had to be rerun. Earlier runs predate the limit: `pose-full-001` ran 64 single-GPU shards (peak concurrency not recorded, between about 30 and 64), and the SL-GCN trainings used 20.

**Audit of the authors' logits (CPU, local).** Get the logit zip from the Drive folder above (check SHA-256 `2e691817…b3cf7`) and the published code at the pinned commit. Then, from a scratch directory outside both:

```bash
git clone https://github.com/impressncsu/ICCV2025_MSLR.git upstream
git -C upstream checkout 7dd73c4e67b2959a95f828c103269723528b9271
unzip -q Logit_folder-20261104T104422Z-1-001.zip -d logits
python3 -P -E path/to/papers/manjur-2025-video-radar-fusion/scripts/audit_author_logits.py logits/Logit_folder upstream
```

This needs Python 3 with numpy, pandas and openpyxl. The script ran from the paper root on 2026-11-04 and was later moved unchanged to `scripts/`.

## Data provenance and permissions

| Dataset | Version/subset/splits | Source and access date | License/permission and cloud-use basis | Path in Volume `datasets` | Counts / manifest / checksum | Deviations |
| --- | --- | --- | --- | --- | --- | --- |
| MultiMeDaLIS, ICCV MSLR 2025 Track-2 release | RGB video (`.mkv`) plus three 60 GHz RDM clips (`.mp4`, "RDM" in the name) per sample; train in 126 class folders `0_A`…`125_Z`; val/test class-agnostic with concealed labels | Kaggle `iccv-mslr-2025-track-2`, archive downloaded 2026-11-06T11:55Z (21,989,719,853 bytes, SHA-256 `68699aa6…dd741d`); rules accepted by the operator 2026-11-06 | CC BY-NC-SA 4.0 plus the competition rules. Rule 2.4b limits access to people who accepted them, so the data is processed only in Restricted Environment `$MODAL_ENV` (project-lead decision 2026-11-05) | Volume `datasets` **in environment `$MODAL_ENV`**, path `multimedalis-iccv-mslr-2025-track-2/raw/` | Verified train 126 × 100 = 12,600, val 4,410, test **7,560** (60 per class on average, not the paper's 70; a paper-text discrepancy, see "Split sizes in the paper text"); 24,570 unique ids; `MANIFEST.sha256` (98,280 files) `27241621…ceee5f`; `INVENTORY.json` `4ace0512…145029`; `PROVENANCE.json` `8232e67e…d9ab3` | Restricted-environment Volume instead of the shared `datasets` Volume (below) |

The val and test ids in the data match the ids in the authors' released logits exactly (4,410 and 7,560), and no id is shared between train, val and test.

**Deviation `restricted-dataset-environment`.** AGENTS.md places every dataset in the single shared Volume `datasets` (environment `main`). Kaggle competition rule 2.4b limits data access to people who accepted the rules, and the shared Volume is readable by every workspace member. So the project lead decided on 2026-11-05 to keep this dataset in a separate VolumeFS v2 Volume, also named `datasets`, inside the Restricted Environment `$MODAL_ENV`:

- Default member role is No Access; the operator is Contributor; the two workspace Owners keep administrator access, the same policy as the project's HPC environments.
- All of this paper's Modal work runs there, including derived data, checkpoints and outputs in Volume `manjur-2025-radar-fusion-results`.
- The shared `huggingface-cache` Volume is mounted from `main` (`environment_name="main"`).

The Kaggle credential is the operator-created Modal Secret `kaggle-credentials` (variable `KAGGLE_API_TOKEN`). It is never inspected or printed.

**Data-gate history.**

- The first download (`data-populate-001`) returned 403 Forbidden. The cause, which the operator identified, was that the competition rules had not yet been accepted.
- The operator accepted them on 2026-11-06 (correcting an earlier report of 2026-11-05).
- The rerun `data-populate-rules-accepted-001` downloaded the archive, verified it, wrote the manifest and deleted the archive.

## Environment and patches

| Patch | Demonstrated failure | Hypothesis | Why necessary | Behavioral effect | Evidence |
| --- | --- | --- | --- | --- | --- |
| `patches/0001-data-paths.patch` (SHA-256 `1c97eec2…a534da`; first written as root `upstream.patch`, moved unchanged) | Every data path in 8 scripts is a literal path on the authors' Windows machine; `demo.py` has empty input/output paths; two scripts select the split by a hand-edited `DIR = "Test"` line | Reading these from environment variables makes the scripts runnable in a container | Containerization; no other way to point the scripts at the data | None: 14 assignments changed in 9 files. `fusion_pred_trainval.py` (out of scope) untouched | Applies cleanly to `7dd73c4`; applied in both images at build time |
| `patches/0002-video-save-final-model.patch` (SHA-256 `10f1219e…10c9ae`) | `resnet21D_V2.py` trains with `EarlyStopping(restore_best_weights=True)` but saves only every-5-epoch checkpoints and an xlsx; there is no final model for the prediction scripts to load as `video_final.h5` | One `model.save("r2p1d_final.h5")` after `model.fit()` persists the model the script ends with | Required by protocol A1 (operator decision 2026-11-09) | None on training: output only | Applies after 0001; both verified on a fresh checkout of `7dd73c4` |

Other environment decisions (details in `reproduction.json.environment_notes`):

- **Upstream checkout name.** Upstream is cloned to `/opt/ICCV_MSLR_2025`, because the authors' scripts find their project root by that directory name. This avoids patching the lookup.
- **TF image.** The radar and video models run in a Python 3.11 image with `tensorflow[and-cuda]==2.15.1`, numpy 1.26.4, opencv-python-headless 4.10.0.84, h5py 3.11.0, pandas 2.2.2, scikit-learn 1.5.2, matplotlib 3.9.2 and openpyxl 3.1.5. Their code uses Keras-2 APIs (`K.abs` inside `Lambda`, `.h5` `load_model` with `custom_objects`). TF 2.15 is the last release where `tf.keras` is Keras 2, and it needs Python 3.11 or older, so the Python 3.12 project base image cannot host it. An L4 check confirmed TF 2.15.1, Keras 2.15.0 and a visible GPU.
- **PyTorch image.** HRNet and SL-GCN run in the project base image (torch 2.11 nv26.02, CUDA 13.1) plus yacs 0.1.8, natsort 8.4.0, opencv-python-headless 4.11.0.86 and scikit-learn 1.5.2.
- **Pinned extras in the PyTorch image.** The HRNet COCO-WholeBody checkpoint comes from MMPose's release URL (SHA-256 `6e061c6a16ef…1e6976`, matching its file-name prefix), and SAM-SLR is pinned at `de6c53a6` for its SL-GCN training configs.
- **Video decoding.** Video decoding happens only inside the authors' own preprocessing (OpenCV), which is kept unchanged; this attempt adds no decoding of its own.
- **Read-only data.** The authors' preprocessing writes outputs beside the raw files. The dataset stays read-only: the scripts run on a mirror tree of per-sample folders with symlinks to the raw files, kept in the results Volume.

## Execution evidence

| Run ID / agent IDs | Attempt / max | Kind / targets | Platform / hardware | Seed/config | Start/end | Exit / terminal state / reason | Failure class | Stop ceilings | Logs/artifacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `audit-author-logits-001` / `claude-opus-5-5-claude-code` | 1 / 1 | Evidence audit; no targets | Local CPU (Darwin arm64); Python 3.11.5 | Deterministic; inputs pinned by SHA-256 | 2026-11-04T11:16:59Z / 11:17:09Z | 0 / succeeded / completed | none | 600 s wall time; no GPU or cost ceiling (CPU) | stdout SHA-256 `85a2a572…c96897` |
| `data-populate-001` / `claude-opus-5-5-claude-code-2-1-319` | 1 / 1 (declared 4, lowered on classification) | Data population | Modal `repro-sign` / `$MODAL_ENV`, CPU 8 cores, 16 GiB; app `ap-jHEqJOQRwyJ3qLTrWhKgDy` | kaggle 2.2.4 | 2026-11-06T09:48:30Z / 09:48:37Z | 1 / failed / `command_failed`; retry decision `stop` (gate `data-kaggle-download-forbidden`, since resolved) | `license_access` | 6 h; CHF 10 | 403 Forbidden from `DownloadDataFiles` (rules not yet accepted); the successful rerun is `data-populate-rules-accepted-001`, in its own attempt group |
| `data-populate-rules-accepted-001` / same | 1 / 4 | Data population | as above; app `ap-7rb6LQj5t7fDmGgT4xQ6U2` | kaggle 2.2.4 | 2026-11-06T11:54:58Z / 12:21:28Z | 0 / succeeded / completed | none | 6 h; CHF 10 | artifacts `dataset-manifest`, `dataset-inventory`, `dataset-provenance`, `data-image-freeze` |
| `prep-video-full-001` / same | 1 / 4 | Video `.h5` for all splits | Modal CPU, 247 shards × 2 cores; app `ap-5TUZmoXqHdqrR28m356W7C` | authors' `train/val_video_resized.py` | 12:35:20Z / 12:38:53Z | 0 / succeeded / completed | none | 3 h; CHF 20 | 24,570 `.h5` files in the results Volume |
| `prep-radar-full-001` / same | 1 / 4 | Radar PNGs | app `ap-9fCGZ5JeXsOCqraKs431Ya` | authors' `train/val_md_gen.py` | 12:35:20Z / 12:36:12Z | 1 / failed / `command_failed` | `deterministic_code` | 3 h; CHF 20 | Glue bug: fixed `/tmp` path collided when Modal reused a container; retried |
| `prep-radar-full-002` / same | 2 / 4 | Radar PNGs for all splits | app `ap-J44QCOXkIvGSqzEUgAmJXy` | as above | 12:40:01Z / 12:45:24Z | 0 / succeeded / completed | none | 3 h; CHF 20 | 73,710 PNGs (3 per sample) |
| `bench-slgcn-cpu4-001` / same | 1 / 2 | SL-GCN speed benchmark | L4, 4 cores, 16 GiB; app `ap-uvNxefzbZhLLqSq5dvMD60` | joint fold 0, 3 epochs | 2026-11-09T08:42:10Z / 08:51:57Z | 0 / succeeded / completed | none | 1 h; 1 GPU-h; CHF 2 | ~78 s train + ~11 s eval per epoch, same as 8 cores / 32 GiB |
| `bench-slgcn-pack2-001` / same | 1 / 2 | SL-GCN packing benchmark | L4, 8 cores, 32 GiB; app `ap-q5VjwGg3QQmOht2rWeZQvG` | joint + bone fold 0 on one GPU | 08:42:10Z / 08:56:39Z | 0 / succeeded / completed | none | 1 h; 1 GPU-h; CHF 2 | ~94 s per job-epoch versus ~89 s alone: packing saves nothing |
| `pose-full-001` / same | 1 / 2 | HRNet keypoints, all splits; supports Tables 2–3 | 64 shards on L4, 8 cores, 16 GiB; app `ap-d1wAwlgQl62pbFOPWMGU7m` | authors' `demo.py`, 4 processes per GPU | 08:58:05Z / 10:13:04Z | 0 / succeeded / completed | none | 6 h; 42 GPU-h; CHF 55 | 24,570 keypoint files; USD 48.46, 37.1 L4-h |
| `train-radar-full-001` / same | 1 / 2 | Radar CNN; Table 1 radar, Table 3 | L4, 4 cores, 64 GiB; app `ap-RMmXgPSbmvvBEJ5sgoQW5D` | `md_cnn_3branch.py` as published (80/20, seed 42) | 08:58:05Z / 09:07:40Z | 0 / succeeded / completed | none | 4 h; 3 GPU-h; CHF 5 | `relational_md_cnn_enhanced_final_valAcc_0.9099.h5` (early stop at epoch 28) |
| `train-video-full-001` / same | 1 / 2 | ResNet(2+1)D; Table 1 video, Table 3 | L4, 4 cores, 64 GiB; app `ap-FcHJZMfmoaWQPFW3f2Vaxa` | `resnet21D_V2.py` as published (95/5, seed 42) + patch 0002 | 08:58:05Z / 18:57:51Z | 0 / succeeded / completed (the local client's connection dropped at ~12:45Z; the detached function was unaffected) | none | 24 h; 20 GPU-h; CHF 35 | `r2p1d_final.h5`: early stop at epoch 113, epoch-83 weights restored (internal 5% split val_loss 0.2794, val_acc 0.9413); SHA-256 `21f3672e…08fa9e`. USD 15.00, 9.99 L4-h |
| `slgcn-data-full-001` / same | 1 / 2 | SL-GCN arrays | CPU 8 cores, 32 GiB; app `ap-4pPP6ZtQDRUTCxvQYRwBdN` | 5 folds, `StratifiedKFold(5, shuffle=True, random_state=1)` | 10:13:59Z / 10:18:44Z | 0 / succeeded / completed | none | 6 h; CHF 5 | 4 streams × (5 folds of 10,080 / 2,520, val 4,410, test 7,560) |
| `author-slgcn-weights-check-001` / same | 1 / 2 | Evidence only, no targets | L4, 8 cores, 32 GiB; 4 apps (`ap-KtYTqoAr1PzmhOS7xtPv4j`, `ap-tWXhRW2UJ6h38w3cKfzUEK`, `ap-v3U6qDhNqQD9Lz0rOEDgdQ`, `ap-ziw2LuE26ldTqVsgUoCJgY`) | authors' 20 fold checkpoints and test configs | 10:19:14Z / 10:45:41Z | 0 / succeeded / completed | none | 4 h; 3 GPU-h; CHF 5 | 100% argmax agreement with the authors' released SL-GCN logits (val 4,410, test 7,560; Pearson ≥ 0.99999) |
| `slgcn-train-full-001` / same | 1 / 2 | SL-GCN 4 streams × 5 folds; Tables 2–3 | 20 × L4, 4 cores, 16 GiB; app `ap-RoEhCWPKcgfyYuUagAJFVw` | SAM-SLR train configs, 250 epochs | 10:18:44Z / 12:41:22Z | 1 / interrupted / `external_interruption`; retry decision `resume` → `slgcn-train-full-002` | `interrupted_full_run` | 24 h; 170 GPU-h; CHF 185 | All 20 inputs cancelled at 12:40:38Z after 76–78 epochs, when the local orchestrator lost its network; checkpoints kept. USD 53.15, 47.5 L4-h |
| `pf-slgcn-resume-001` / same | 1 / 2 | Resume preflight | L4, 4 cores, 16 GiB; apps `ap-753svEgAqRUQrrt5GmkLZ4`, `ap-X6qrT2bImDARLlbIaAYMIB` | joint fold 0: 2 epochs, then resume to 3 | 13:04:42Z / 13:11:24Z | 0 / succeeded / completed | none | 1 h; 1 GPU-h; CHF 2 | `Load weights from …/epoch-1.pt`, continued at epoch 3; eval records for all 3 epochs |
| `slgcn-train-full-002` / same | 2 / 2 | Resume of `slgcn-train-full-001` | 20 × L4, 4 cores, 16 GiB; remote orchestrator; app `ap-WgJNGxPkRiwBHdgKsXXNIO` | as attempt 1, from each fold's last checkpoint (start epoch 76–78) | 13:13:32Z / 18:39:26Z | 0 / succeeded / completed (the local client lost its connection at 15:04:45Z; the remote orchestrator and all 20 trainings kept running) | none | 24 h; 120 GPU-h; CHF 130 (the group's remaining allocation) | All 20 folds reached epoch 250 (18,859–19,531 s each); best held-out epochs 162–249; fold summaries and SHA-256 in `reproduction.json`. USD 120.98, 107.75 L4-h. **Bone, jm and bm trained with `is_vector: True`: dead end, no targets** |
| `slgcn-test-full-001` / same | 1 / 2 | SL-GCN inference, authors' test configs | L4, 8 cores, 32 GiB; apps `ap-kRZVIcdB98V9PKkbq6bxw9` (val), `ap-ZZIW3lAFw6YBxMvRJAXln6` (test) | our selected checkpoints | 18:52:37Z / 19:08:46Z | 0 / succeeded / completed | none | 4 h; 3 GPU-h; CHF 5 | 4,410 / 7,560 rows; all 5 bone folds needed the `best_acc.pkl` fallback (no sample predicted as class 0), the first sign of the `is_vector` problem |
| `compare-slgcn-author-001` / same | 1 / 2 | Evidence only, no targets | Modal CPU, 2 cores, 8 GiB; 12 apps | `compare_logits`, ours vs the authors' released-weight logits | 19:09:11Z / 19:14:26Z | 0 / succeeded / completed | none | 1 h; no GPU; CHF 1 | Argmax agreement with the authors: joint 99.8/99.6%, jm 99.0/98.9%, bm 99.0/98.5%, **bone 0.86/0.85%** (val/test), ensemble 99.6/99.2%. Authors' bone vs their joint 99.16%; ours 0.86% |
| `slgcn-isvector-check-001` / same | 1 / 2 | Evidence only (option V1, val) | L4, 8 cores, 32 GiB; 7 apps | test `is_vector: True` for bone, jm, bm | 19:15:18Z / 19:32:26Z | 0 / succeeded / completed | none | 1 h; 0.5 GPU-h; CHF 2 | Bone vs the authors' 99.27% (from 0.86%); jm/bm unchanged; ensemble 99.89%. Conditional evidence, not targets |
| `slgcn-v1-test-001` / same | 1 / 2 | Evidence only (option V1, test) | L4, 8 cores, 32 GiB; app `ap-DsGDnHIY2oX8iMuNWkRR0R` | as above, test split | 2026-11-10T12:12:39Z / 12:28:44Z | 0 / succeeded / completed | none | 1 h; 0.75 GPU-h; CHF 2 | 7,560 rows; no `best_acc.pkl` fallback needed. Conditional evidence, not targets |
| `predict-full-001` / same | 1 / 2 | Radar and video inference; Tables 1 and 3 | L4, 8 cores, 32 GiB; apps `ap-x9Dtz5JLT1o95rfTAv62An` (val), `ap-QK1cTuWhLe889UDPhvoq83` (test) | authors' `val_fusion_pred.py`, `test_fusion_pred.py` with our models | 2026-11-09T19:05:36Z / 20:55:49Z | 0 / succeeded / completed | none | 4 h; 2 GPU-h; CHF 5. **GPU-hour ceiling exceeded:** 3.27 L4-h used (USD 4.68) | logit/prob xlsx and prediction CSVs for 4,410 val and 7,560 test samples; models loaded: radar `14946665…b22102`, video `21f3672e…08fa9e`. A first launch failed at CLI argument parsing (zsh did not split a variable holding the flags; nothing reached Modal) |
| `pf-guard-001` / same | 1 / 2 | Guard test | Modal CPU 0.25 core, 512 MiB; apps `ap-YaLrImQWdR3IdIrUpX8Oax` (A), `ap-yCqi4Bx0YJnwdSIJadi5yX` (B) | guard B must stop throwaway guard A | 2026-11-10T14:02:04Z / 14:12:56Z | 1 / failed / `command_failed`; retry → `pf-guard-002` | `deterministic_code` | 15 min; CHF 0.5 | The Modal CLI does not run inside containers (`No module named 'rich'`); B retried every second after its deadline. Both stopped by hand |
| `pf-guard-002` / same | 2 / 2 | Guard test, gRPC version | app `ap-mbZVaa4czmOoDRbkSUPNi0` (A) | as above | 14:13:49Z / 14:18:59Z | 1 / failed / `command_failed`; retry decision `stop` | `deterministic_code` | 15 min; CHF 0.5 | A hung on its first RPC (event-loop mismatch); B never launched (macOS has no `timeout`). A stopped by hand |
| `pf-guard-sync-001` / same | 1 / 2 | Guard test, synchronizer version | apps `ap-C2FDQavGG6259TtmVubuq6` (A), `ap-NcIrNjeO2jaqUr3N3cyc5r` (B) | as above | 14:19:33Z / 14:20:59Z | 0 / succeeded / completed | none | 15 min; CHF 0.5 | B stopped A at its deadline (requested 14:20:42Z, stopped by 14:20:57Z) and returned; A read the retrain's state correctly; retrain untouched |
| `pf-slgcn-retrain-001` / same | 1 / 2 | Retrain preflight | L4, 4 cores, 16 GiB; app `ap-N1uXvuH3HOQygV8etRvzfs` | bone fold 0, 2 epochs, `--authors-is-vector`, fold data of `slgcn-full-001` | 2026-11-10T13:14:44Z / 13:19:14Z | 0 / succeeded / completed | none | 40 min; 0.75 GPU-h; CHF 1 | Config written with `is_vector: false` in both feeders; held-out accuracy 2.6% → 9.7% (the `is_vector: True` run had 3.2% → 25.9%: slower start in warm-up, recorded, not tuned) |
| `slgcn-retrain-full-001` / same | 1 / 2 | SL-GCN bone, jm, bm × 5 folds retrained (R1); Tables 2–3 | 15 × L4, 4 cores, 16 GiB; remote orchestrator; app `ap-BLTNkE5vRv7KdWGHgpCoBi`; 15 of the 30-GPU limit | `is_vector: False`, SAM-SLR schedule, 250 epochs; run dir `slgcn-full-002` | 13:20:06Z / 19:26:10Z | 1 / interrupted / `external_interruption`; retry decision `resume` → `slgcn-retrain-full-002` | `interrupted_full_run` | 24 h; 135 GPU-h; CHF 150 — **enforced** by guard app `ap-hHzrQ1Puq2Drp6I4d9BjeG` (stop at 22:01:50Z; the cost ceiling binds: 15 × 8.70 h × USD 1.15) plus a local backup stop at 22:11:50Z; only the 24 h timeouts applied between launch and 14:21Z. Not reached. Neither the guard nor its backup stopped or affected the run: the guard returned `stopped_by_guard: false` at 19:26:59Z, after the app had stopped | Modal preempted bm folds 1–3 at ~17:05Z, after 120–127 of 250 epochs per fold. This attempt's glue turned that into a cancelled run (see Attempts). No checkpoint lost. USD 65.57, 58.34 L4-h (complete) |
| `pf-guard-tasks-001` / same | 1 / 2 | Guard test, task-count limit | apps `ap-mMmsizJm83aB4zBOZjgOIB` (A), `ap-X7IdLTuAIpffR86FjfJhHh` (B) | B with `--max-tasks 0` must stop A (1 task) | 19:33:39Z / 19:34:34Z | 0 / succeeded / completed | none | 15 min; CHF 0.5 | Two polls over the limit (19:33:57Z, 19:34:13Z), stop requested at 19:34:13Z, A stopped by 19:34:28Z; B returned |
| `pf-slgcn-all-001` / same | 1 / 2 | Orchestrator preflight | L4, 4 cores, 16 GiB; apps `ap-Lr2YCnOg2kBppcde7I1Ex5` (a), `ap-pzjmIVsyVOEM59zTHq8SW1` (b), each with a guard (`--max-tasks 2`) | bone fold 0, 1 epoch, scratch run id; then the same without `--resume` | 19:35:36Z / 19:49:49Z | 1 / failed / `command_failed`; retry → `pf-slgcn-all-002` | `deterministic_code` | 1 h; 1 GPU-h; CHF 2 | a trained 1 epoch (exit 0). b's fold refused as intended, but the orchestrator hung: the remote `SystemExit` killed Modal's client thread, and `return_exceptions` does not catch `SystemExit`. Stopped by hand |
| `pf-slgcn-all-002` / same | 2 / 2 | Orchestrator preflight, `RuntimeError` version | L4, 4 cores, 16 GiB; app `ap-HCloAQefYSkttk0AAC7wL3`, guard `ap-S58mAq9lznOhdNwUQirVVp` | invocation b again | 19:50:30Z / 19:51:55Z | 1 / failed / `command_failed` (the exit 1 is the tested outcome); retry decision `stop` | `deterministic_code` | 30 min; 0.5 GPU-h; CHF 1 | `failed_inputs` reported the fold's `RuntimeError`, and the orchestrator exited once after 85 s, with no hang or relaunch. Fix kept |
| `slgcn-retrain-full-002` / same | 2 / 2 | Resume of `slgcn-retrain-full-001`; Tables 2–3 | 15 × L4, 4 cores, 16 GiB; nonpreemptible remote orchestrator; app `ap-Or6VA4QMCT2DYpKBSlMDaI`; 15 of the 30-GPU limit | as attempt 1 with `--resume`, from each fold's last checkpoint (start epoch 120–127) | 19:54:25Z / 2026-11-11T00:18:56Z | 0 / succeeded / completed | none | 24 h; 76 GPU-h; CHF 84 (the group's remainder) — **enforced** by guard app `ap-0j8xcaIF5HJlInObqB6S0b` (stop at 2026-11-11T00:46:35Z; cost binds: 15 × 4.87 h × USD 1.15; at most 16 tasks) plus a local backup stop at 00:56:35Z. Not reached: at most 16 tasks seen; finished 27.7 min before the deadline | All 15 folds reached epoch 250 and exited 0; no preemption. Best held-out epochs 165–245; mean best held-out accuracy bone 98.71%, jm 98.48%, bm 98.10% (training-set folds, not targets). Fold summaries and checkpoints under `runs/slgcn-full-002/` on the results Volume (SHA-256 per fold in `reproduction.json`). USD 66.99, 59.56 L4-h; group total USD 132.56, 117.9 L4-h |

What the logit audit found:

- The authors' released logits reproduce all 16 published prediction CSVs exactly.
- The one exception is the Radar + Video row. Its published predictions use **probability** averaging; logit averaging, as the paper text describes, would differ on 79 val and 110 test samples.

**Preflight on real data (disposable, 2026-11-06).** These calls were launched without a contemporaneous stop-policy declaration, so they are recorded as evidence with Modal app IDs in `compute_assessment` rather than as run entries. All on NVIDIA L4:

| Stage | What ran | Measured | Full-run estimate |
| --- | --- | --- | --- |
| HRNet pose (`demo.py`) | 10, 40, 504 + 126 train and 10 test videos; 4 concurrent unmodified processes per GPU | ~675–710 videos per GPU-hour; peak 3.4 GiB | ~35 GPU-h for the remaining ~23,900 videos |
| Radar CNN (`md_cnn_3branch.py`) | 10 per class; early-stopped at epoch 14 | 49 ms/step; 79 s total | under 0.5 GPU-h |
| ResNet(2+1)D (`resnet21D_V2.py`) | 20 per class, capped at 1,200 s (11 epochs) | 0.73 s/step; first epoch 644 s (whole-set shuffle buffer in RAM) | 5–15 GPU-h (≤200 epochs, early stopping) |
| Radar/video inference (`val_fusion_pred.py`, `test_fusion_pred.py`) | preflight models reloaded on 20, 49 val and 10 test samples | 37 s for 20 | under 1 GPU-h |
| SL-GCN (`main.py` with SAM-SLR train configs) | 4 streams × 5 folds × 3 epochs on 630 samples; full-size speed benchmark (real samples replicated ×16, speed only) | ~110 s/epoch at full fold size; peak 8.4 GiB | ~154 GPU-h for 20 runs × 250 epochs (code protocol); ~33 for 4 full-data runs (paper protocol) |
| SL-GCN test, fusion, submissions | authors' 20 fold test configs + `run_fusion.py`, `fusion_all.py`, `Multimodal_fusion.py`; 11 `id,pred` files | the two fusion scripts agree | ~1–2 GPU-h |

Input sanity was also checked. The µ-D PNGs are 128 × 128 JET images and the video `.h5` volumes are 64 × 96 × 96; for both modalities, within-class distances are below between-class distances.

## Guesses and deviations

**Protocol decision (gate `protocol-training-recipe`, resolved 2026-11-09).** The operator chose **A1**: follow the published code as-is, plus the single output-only patch above.

**Why not B (the paper's text).** Section 5 says each model was retrained on the full training set, but it does not say how training was stopped in that final run, and the published code contains no full-training-set run at all. Its training loop depends on a held-out split: early stopping, learning-rate reduction and (for video) best-weight restoration all monitor validation loss. Training on 100% therefore needs a stopping rule the authors never state. One could infer it from the held-out runs, for example by reusing the epoch at which early stopping fired. That inference would be a behaviour-changing choice, so its result would be conditional evidence, not a produced target. Following the published code is the least invasive faithful option, and B stays documented as the rejected alternative.

**Paper/code discrepancies (recorded, not fixed):**

- Paper Section 5 describes an 80/20 split for tuning followed by retraining on the full training set; no retraining step exists in the code.
- The code holds out 20% for radar (`VALID_SPLIT = 0.2`, 100 epochs, early-stopping patience 10, `restore_best_weights=False`). For video it holds out only 5% (`VALID_SPLIT = 0.05`, 200 epochs, patience 30, `restore_best_weights=True`), so even the video tuning stage does not match the paper's 80/20.
- `MSLR_radar_video/readme.md` calls the Google Drive weights "the final trained weights", and the radar script saves `relational_md_cnn_enhanced_final_valAcc_*.h5`. Both are consistent with the released models being held-out-split models rather than 100% retrains.

| Detail | Paper/evidence says | This attempt used | Rationale | Effect on interpretation |
| --- | --- | --- | --- | --- |
| Final training data | Paper: retrained on the full training set | Code (A1): radar 80/20, video 95/5, SL-GCN five folds per stream | Operator decision; reasons above | The reproduced numbers test the released code, not the paper's retraining text |
| Video final model | Script saves none | `r2p1d_final.h5` from patch 0002 (the restored best-val-loss weights) | A1 | Exactly the model the script ends with |
| **Guess:** SL-GCN fold assignment | Unpublished | `StratifiedKFold(n_splits=5, shuffle=True, random_state=1)` over the 12,600 labelled training samples | Seed 1 matches `main.py`'s `init_seed`; stratification keeps all 126 classes in each fold | Fold membership differs from the authors' unknown folds |
| SL-GCN `is_vector` for bone, jm and bm (retrain) | Authors' fold configs: False in both feeders; their weights reproduce their logits under it | False in both feeders (`--authors-is-vector`) | Gate `slgcn-retrain-settings`, R1 | Corroborated, not a guess |
| SL-GCN training length (retrain) | Checkpoint names `…_250/epoch-176…249`; fold configs say 100 epochs (= SAM-SLR's test template) | 250 epochs | Corroborated by the checkpoint names; a 100-epoch run cannot produce them | Corroborated |
| **Guess:** SL-GCN learning-rate schedule (retrain) | Unpublished; fold configs say lr 0.01, step [50], warm-up 0, byte-identical to SAM-SLR's test template and possibly an unmodified copy | SAM-SLR train values: lr 0.1, steps [150, 200], warm-up 20 | Inferred from SAM-SLR (operator decision R1) | Consistent with, but not confirmed by, the authors' artifacts |
| **Guess (agent's), partly a dead end:** SL-GCN training config | No training configs published. The authors' fold configs are SAM-SLR's test templates with `is_vector: False` for bone, jm and bm (SAM-SLR: True), and a training section (100 epochs, lr 0.01, step [50], warm-up 0) that is SAM-SLR's unchanged test boilerplate and contradicts their own `_250` / `epoch-249` checkpoint names | SAM-SLR `train_{stream}.yaml` at `de6c53a6` (250 epochs, lr 0.1, steps 150/200, warm-up 20, batch 64, keep_rate 0.9, `is_vector: True` for bone/jm/bm), with 126 classes and one GPU | The agent's inference from the cited basis and the `_250` naming, made without a field-level comparison with the authors' fold configs | `is_vector: True` contradicts the published evidence: a **dead end** (see Attempts). The schedule is undetermined (gate `slgcn-retrain-settings`). Bone, jm and bm models from this config produce no targets |
| **Guess:** SL-GCN checkpoint per fold | Authors' selected epochs 176–249 of 250, odd and even | A checkpoint every epoch; the best held-out top-1 epoch (earliest on ties) | `main.py` tracks the best held-out accuracy | Inferred selection rule |
| `is_vector` at test time (conditional evidence only) | Authors' fold test configs: default False | `slgcn-isvector-check-001`, `slgcn-v1-test-001`: True for bone, jm and bm (`--vector-streams`) | Operator's step 1 (2026-11-10): show what our `is_vector: True` models give under the normalization they were trained with | None on targets; outputs carry a `_vec-bm-bone-jm` suffix |
| `predict-full-001` GPU-hour ceiling | Declared 2.0 GPU-hours | 3.27 L4-h used (USD 4.68, within its CHF 5) | The estimate extrapolated a 20-sample preflight; no timeout enforced the ceiling | None on outputs; recorded as a stop-policy overrun |
| SL-GCN `best_acc.pkl` | `main.py` writes it in test phase only if accuracy against dummy labels is above 0 | Glue copies the identical `epoch_*.pkl` score dict when it is missing, and records each use | Upstream execution quirk | None on scores |
| Table 2 per-stream logits | Paper reports them; code writes only the fused file | Each stream's 5-fold average, exactly as `run_fusion.py` forms it | Needed for the Table 2 targets | None on the method |
| Radar + Video fusion space | Paper: logits | Code path (softmax probabilities, 0.5/0.5) | A1; matches the authors' released predictions | Recorded discrepancy |
| Pose estimator | Paper cites HigherHRNet | Code's top-down HRNet-w48 COCO-WholeBody, with 4 unmodified processes per GPU | Published path; concurrency is infrastructure only | Recorded discrepancy |
| Test set size | Paper Sec. 5: 70 per class (its val figure of 35 is consistent) | The released 7,560-row test set (about 60 per class), matching the data, the authors' logits and every prediction CSV in their repository | Paper-text discrepancy, not a data mismatch | Denominator only (7,560, not 8,820) |
| Corpus size | Paper Sec. 3.1: 25,830 = 126 × (100 + 35 + 70) | The released partition: 24,570 = 126 × (100 + 35 + 60) | Possibly derived from the same per-class figures; may describe the full corpus (the organisers' summary gives 25,830 for 205 performances per item), so not treated as wrong | None |
| Video test prose value | 93.737 | Target 93.74 (Table 1) | 93.737 is unreachable with 7,560 rows | None on the target |
| Data location | Shared `datasets` Volume | Restricted Environment Volume | Competition rule 2.4b | None on results |
| SL-GCN training resumed once | One uninterrupted 250-epoch run per fold | Each fold resumed at epoch 76–78 (of 0–249) from its own last checkpoint, through `main.py`'s own `weights` and `start_epoch` options | External interruption of `slgcn-train-full-001`; AGENTS.md prescribes resuming from a verified checkpoint over a restart | Weights and the epoch-based learning-rate schedule carry over. SGD momentum buffers and the data-order RNG restart at the resume point. The authors' runs were not interrupted. The effect is expected to be small but is not measured |
| SL-GCN retrain resumed once | One uninterrupted 250-epoch run per retrained bone/jm/bm fold | Each of the 15 folds resumed exactly once, at epoch 120–127, from its own last checkpoint (`slgcn-retrain-full-002`, `--resume`), through `main.py`'s own `weights` and `start_epoch` options; no container was preempted in the resumed run | Modal preempted three containers, and this attempt's glue turned that into cancelling all 15 (`slgcn-retrain-full-001`; the guard was not involved). AGENTS.md prescribes resuming from a verified checkpoint over a restart; the resume stayed within the retrain group's 135 GPU-h / CHF 150 (used: 117.9 L4-h, USD 132.56) | Weights and the epoch-based learning-rate schedule carry over; every fold resumes after the 20-epoch warm-up and before the first learning-rate step at epoch 150. SGD momentum buffers (Nesterov, 0.9) and the data-order (shuffle) RNG restart at the resume point, so the trajectory differs from an uninterrupted run. The authors' runs were not interrupted. Every selected best epoch (165–245) lies after the resume. The effect is expected to be small but is not measured |

**Compute decision (gate `compute-full-plan`, resolved 2026-11-09).** The operator authorized A1 under attempt-wide ceilings of **CHF 300 total Modal spend and 250 L4 GPU-hours**. Both include the pose extraction and the preflights already run. The basis is that no plan-specific #repro-sign-team-r discussion has taken place yet, but on 2026-10-29 the project lead stated that Modal spending was well below budget and relaxed the spending restriction; the operator authorizes the ceiling on that basis as the responsible human reproducer. The conditions are:

- Every run's stop policy stays within the ceilings.
- If an estimate exceeds them, the run stops and the gate reopens.
- There are no repeated full restarts without a new diagnosis.

Spend before the full runs was USD 7.44 (about 2.1 L4 GPU-hours), taken from the Modal billing report and counted 1:1 as CHF. Two declared benchmarks then set the container sizes:

- `bench-slgcn-cpu4-001`: SL-GCN on 4 cores and 16 GiB is as fast as on 8 cores and 32 GiB.
- `bench-slgcn-pack2-001`: two SL-GCN jobs sharing an L4 save nothing, because the GPU is saturated.

The projection was about 210 L4 GPU-hours and about USD 250–260.

**Ceiling increase, 2026-11-10.** In #repro-sign-team-r the project lead approved retraining the SL-GCN streams whose training config mismatched the authors' published settings, and raised this attempt's ceilings to **CHF 450 and 380 L4 GPU-hours** (relayed by the operator). The CHF 300 / 250 decision above stays as history (`reproduction.json.compute_budget.ceilings.history`). The same day the operator set a concurrency limit: at most **30 GPUs at once** for this paper across all its jobs (see How to repeat this).

**Spend checkpoint, 2026-11-09T13:00Z: USD 116.31 and about 91.5 L4 GPU-hours.** The interrupted SL-GCN attempt accounts for USD 53.15 / 47.5 L4-h of this. The resume adds about 107 L4-h / USD 120: 172–174 remaining epochs per fold at the measured ~110 s per epoch. The worst-case projection is now about 219 of 250 L4-h and about USD 267 of CHF 300, counting the rest of video training at its full 20 GPU-h ceiling.

**Spend checkpoint, 2026-11-10T12:00Z: USD 251.88 and 209.1 L4 GPU-hours** (complete through 2026-11-09 20:00Z; nothing ran between 20:56Z and 12:12Z). That leaves about CHF 198 and 171 L4-h under the raised ceilings. The two SL-GCN training attempts cost USD 174.13 / 155.25 L4-h in total.

**Spend checkpoint, 2026-11-10T19:30Z: USD 317.93 and 267.8 L4 GPU-hours** (billed through the 18:00Z hour). Retrain attempt 1 (`slgcn-retrain-full-001`) accounts for USD 65.53 / 58.31 L4-h of it. Attempt 2 is expected to cost about USD 68 / 60.5 L4-h, with a ceiling of CHF 84 / 76 L4-h. Adding `slgcn_test` (ceiling CHF 5) and fusion, the worst case is about USD 410 of CHF 450 and about 347 of 380 L4-h.

**Spend checkpoint, 2026-11-12: USD 385.14 and 327.4 L4 GPU-hours** (complete through 2026-11-11; nothing of this paper ran after 00:19Z that day). The two retrain attempts cost USD 65.57 / 58.34 L4-h and USD 66.99 / 59.56 L4-h, together USD 132.56 / 117.9 L4-h of the group's CHF 150 / 135. About CHF 65 and 52.6 L4-h remain under the CHF 450 / 380 ceilings for `slgcn_test` and fusion.

**Released weights (evidence only, never used in place of training).**

- **SL-GCN `save_models.zip`** (public, 285,817,696 bytes, SHA-256 `6f33af6b…14921e`). It holds the 20 fold checkpoints named in the test configs (each 19,550,923 bytes, dated 2025-06-07 to 06-09). They are plain state dicts with no optimizer state, so they cannot show training length. They will be run through the authors' own test configs on this attempt's keypoints and compared with the authors' released `logit_slgcnn_{val,test}.csv`, an inference-only check of the pose pipeline.
- **Radar/video `.h5` models.** The Drive folder `1oSWKC3bShKOiI5yynSaQuwl1nnKQfRKf` is not publicly accessible: anonymous listing returned 401 and the folder page redirects to sign-in. Without the files, the requested optimizer-state check (iteration count and learning rate, hence epochs and effective training-set size) cannot be done. No access workaround was attempted. On 2026-11-09 the operator requested access from the authors (see Author and team contact).

## Attempts, failures, and dead ends

- **Label search.** Searched Hugging Face, Zenodo, GitHub, the web, the challenge site and later papers. Nothing found, and the host's 2026 statement points the other way. Result: gate `target-eval-route`, which the operator resolved by choosing Kaggle late submissions.
- **Leaderboard mapping.** Confirmed by the IMPRESS NCSU and CPAMI (UW) leaderboard entries, the competition metadata and the denominator arithmetic.
- **Artifact consistency.** The authors' logits reproduce their published predictions; see `audit-author-logits-001`.
- **Kaggle data listing.**
  - The anonymous internal-endpoint listing was refused by the session's permission policy and not pursued.
  - Two authenticated listing launches are recorded in `reproduction.json.unrecorded_attempts`. One failed because kaggle 2.2.4 imports `six` without declaring it; the kept fix pins `six==1.17.0` and `kagglesdk==0.1.38`. The other hit a 429 rate limit; listing was dropped, since the archive itself is the inventory.
- **Data download.** It returned 403 because the rules had not been accepted (gate `data-kaggle-download-forbidden`). After the operator accepted them, the rerun succeeded.
- **Radar preprocessing.** The first fan-out failed on a temp-directory collision in this attempt's glue code (fixed; `prep-radar-full-002` succeeded).
- **SL-GCN preflight.**
  - 5-fold stratification needs at least 5 samples per class, so keypoints were added for a fifth training sample per class.
  - `run_fusion.py` first failed on the `best_acc.pkl` quirk above; the glue fallback fixed it.
- **Early radar preflight behaviour.** On 8 samples per class, held-out accuracy stayed at one constant class while train accuracy rose to 1.0. The published early stopping (patience 10) then ended training at epoch 14. This is recorded, not tuned.
- **SL-GCN training interrupted (`slgcn-train-full-001`).**
  - At 12:40:38Z, about 2 h 22 min in, Modal cancelled all 20 fold trainings.
  - Cause: `slgcn_all` was a local entrypoint driving `slgcn_train.starmap` from the operator's machine, and that machine lost network access. The client later failed with `[Errno 8] nodename nor servname provided, or not known`.
  - `--detach` keeps an app alive when the client disconnects, but did not protect a local entrypoint that was itself waiting on the remote calls. The video run, a single remote call launched the same way, kept running.
  - Fix in this attempt's glue: `slgcn_all` is now a small remote function, and `slgcn_train` can resume each fold from its last checkpoint that has an eval record.
  - `pf-slgcn-resume-001` verified the resume path on a scratch run id. It also showed that the authors' `main.py` checks for a `save_models/{exp}/epoch` directory before its delete prompt; that path is a file prefix, so the prompt never fires and checkpoints are not at risk.
  - `slgcn-train-full-002` resumed all 20 folds. It is attempt 2 of 2 and stays within the group's original allocation; it is not a full restart.
- **Dead end: SL-GCN `is_vector: True` for bone, joint-motion and bone-motion (`slgcn-train-full-001/002`).**
  - *Hypothesis.* SAM-SLR's train configs, the paper's cited SL-GCN basis, reproduce the authors' unpublished training setup.
  - *How it surfaced.* `slgcn-test-full-001` ran our checkpoints through the authors' fold test configs. All five bone folds needed the `best_acc.pkl` fallback: none predicted class 0 for any val or test sample, unlike the authors' weights.
  - *Evidence* (`compare-slgcn-author-001`):
    - Our bone stream agrees with the authors' on 0.86% (val) and 0.85% (test), chance level for 126 classes, and with our own joint stream on 0.86%.
    - Joint, jm and bm agree with the authors' on 98.5–99.8%.
    - The authors' bone stream agrees with their joint stream on 99.16%.
  - *Cause.*
    - The authors' `gen_bone_data_test.py` leaves node 0 as an absolute position.
    - With `is_vector: False`, the authors' setting and the feeder default, normalization subtracts node 0's mean from every node.
    - SAM-SLR's train and test configs, which our training used, set `is_vector: True` for bone, jm and bm, which centres node 0 only.
    - The authors' fold configs carry `train_feeder_args.is_vector: False` for all four streams, and their released weights reproduce their logits under those configs. So the authors trained with False.
    - Motion streams have a near-zero-mean node 0, so the mismatch barely affects jm and bm.
  - *Classification.* A dead end, not an ambiguity: the inference contradicted published evidence (operator, 2026-11-10).
  - *Attribution.* The inference was the reproduction agent's (see Reproduction agents), not the human reproducer's.
  - *Why the section was missed.*
    - The session record shows no explicit assessment of it.
    - The agent described the repository as having "no training configs" (gate `protocol-training-recipe`) and treated the fold configs as test-phase files (`phase: test`) whose training fields are not authoritative.
    - Several of those fields do contradict the released checkpoints (`num_epoch: 100` and `step: [50]` against weights `sign_*_fold*_250/epoch-176…249`), but the record does not show that this conflict was weighed at the time.
    - The agent compared only the joint configs, where `is_vector` is False in both sources. It first diffed the bone, jm and bm configs field by field on 2026-11-09, after the bone stream failed.
  - *What should have happened.* The conflict should have been raised at protocol gate `protocol-training-recipe` before the full run.
  - *Kept or reverted.*
    - The runs are retained. Their `is_vector: True` bone, jm and bm models produce no targets.
    - Option V1, testing them with `is_vector: True`, is kept as conditional evidence only (`slgcn-isvector-check-001`, `slgcn-v1-test-001`). On val, V1 brings bone to 99.27% agreement with the authors' bone stream.
  - *Cost.* The two training attempts cost USD 174.13 / 155.25 L4-h.
  - *Next.* The full field diff (gate `slgcn-retrain-settings`) shows the authors' schedule fields conflict with their own checkpoint names, so the retrain's schedule awaits the operator.
- **Stop-policy enforcement guard.** `predict-full-001` overran its GPU-hour ceiling because only function timeouts bounded runs, so a deadline guard was added on 2026-11-10 while the retrain was already running (it could not be restarted to change its timeouts).
  - **First version** (`pf-guard-001`): ran the Modal CLI inside a container. It failed with `No module named 'rich'`, and it retried every second after its deadline instead of every poll interval.
  - **Second version** (`pf-guard-002`): called the CLI's `AppList`/`AppStop` RPCs directly. It hung, because the container's client lives on Modal's own event loop and the guard ran a fresh `asyncio.run` loop. The stop half never ran: macOS has no `timeout` command.
  - **Third version** (`pf-guard-sync-001`): runs the coroutine through `synchronize_api`. It works, and it guards the retrain.
  - **Cleanup.** Both failed tests' apps were stopped by hand through the wrapper. `app stop` needs `--yes` when no terminal is attached.
- **Retrain preempted, then cancelled by this attempt's glue (`slgcn-retrain-full-001`).**
  - *What happened.* At about 17:05Z Modal preempted three of the 15 L4 containers (bone-motion folds 1–3), after 120–127 of 250 epochs per fold. Modal retried those inputs automatically. The run had been launched without `--resume`, so each retry refused to overwrite the fold's checkpoints and raised `SystemExit`.
  - *Why epochs 120–127.* Every fold trained at about 112 s per epoch from its container start (13:20–13:22Z). The three preempted folds had evaluated 120 epochs when they were preempted; the other 12 had evaluated 122–127 when they were cancelled at 17:08:33Z. The spread comes from container start times and small speed differences between folds.
  - *Components.* The trigger was Modal's preemption of three preemptible L4 containers, which is external to the training code. The amplifier was this attempt's glue in `scripts/modal_app.py`, written by the agent: the missing `--resume`, `slgcn_train`'s `SystemExit` refusal, and `slgcn_all`'s re-raise of it (next point).
  - *How the glue made it worse.* `slgcn_all` re-raised that `SystemExit` while fetching map results, and it killed Modal's client thread (the log says "Synchronizer thread unexpectedly died", with `SystemExit` as the cause). The other 12 trainings were cancelled at 17:08:33Z. After each 900 s heartbeat timeout Modal reran the orchestrator, whose new inputs refused at once, until the app stopped at 19:26:10Z. No epochs were added after 17:10Z, and no checkpoint was lost.
  - *First fix and its test.* `slgcn_all` collects failed folds (`return_exceptions=True`) and runs nonpreemptible. `pf-slgcn-all-001` showed this was not enough: `return_exceptions` does not catch `SystemExit`, and the orchestrator hung.
  - *Kept fix.* `slgcn_train`'s two refusals now raise `RuntimeError`. `pf-slgcn-all-002` confirmed that the failed fold is reported and the orchestrator exits once.
  - *Guard.* The guard's deadline assumed at most 15 GPU containers. It now also stops the app when more than `--max-tasks` tasks run on two consecutive polls (`pf-guard-tasks-001`).
  - *Not involved.* The deadline guard (`ap-hHzrQ1Puq2Drp6I4d9BjeG`) only polled. Its deadline (22:01:50Z) was never reached, that version had no task-count limit, and it returned `stopped_by_guard: false` at 19:26:59Z after seeing the app stopped. The local backup watcher's stop at 22:14:35Z found the app already stopped. No human stopped the app. Training code, data, the 30-GPU limit and the cost ceilings played no part, and every checkpoint loaded when the folds resumed.
  - *Resume.* `slgcn-retrain-full-002` resumed every fold with `--resume`, so a preempted fold's automatic retry would continue from its own last checkpoint. It finished at 00:18:56Z on 2026-11-11: all 15 folds reached epoch 250, none was preempted, and it used USD 66.99 / 59.56 L4-h of its 76 GPU-h / CHF 84. The resume is recorded as deviation `slgcn-retrain-resumed`.
  - *Monitoring faults, no effect on the run.* The guard's local client printed "Connection lost" and exited 1 after the guard function had already returned. The launcher's backup loop did not recognise the finished app, because `app list` stopped listing it (state "missing", not "stopped"). After its deadline it kept re-issuing `app stop`, each answered "App is already stopped", until it was stopped by hand on 2026-11-12. The launcher now treats an app that drops out of `app list` after being listed as finished.
- **`tf_predict` launch typo.** The first launch of `predict-full-001` passed the model flags through a zsh variable, which zsh does not word-split. The Modal CLI rejected the arguments (exit 2) before anything reached Modal, and the relaunch with the flags written out succeeded.
- **Terminal classification of `data-populate-001`.** First recorded as `stopped`/`human_gate_opened`, which the validator rejects once the gate is resolved. At the operator's request (2026-11-09) it is now recorded as what happened: `failed`/`command_failed` (exit 1 on the 403), `license_access`, retry decision `stop`, referencing the resolved gate. This validates. The rerun has its own attempt group.

## Candidate flags, ethics, and human evaluation

- **Copied scores:** the queue says "no", and that is confirmed.
- **Human evaluation:** the queue says "no", and that is confirmed.
- **Ethics:** the queue says "no". The data shows one proficient signer in frontal RGB video, released by the challenge organisers under CC BY-NC-SA 4.0 and the competition rules. No new participants are involved. Access and processing are restricted to the operator's Restricted Environment, as described above.
- **Compute requirements:** the queue says "N/A". The paper reports no hardware or runtime; the preflight-based estimates are above.

## Author and team contact

| Date | Who → whom | Channel | Purpose | Timing | Outcome / what it changed |
| --- | --- | --- | --- | --- | --- |
| 2026-11-09 | Operator → the authors (owners of the Drive folder linked from the official README) | Google Drive access request for folder `1oSWKC3bShKOiI5yynSaQuwl1nnKQfRKf` (`radar_final.h5`, `video_final.h5`) | Released radar/video weights as **evidence only**: hashes, optimizer-state check (iteration count, learning rate → epochs and effective training-set size), optional check that they reproduce the committed predictions | Made **before the independent attempt finished**, while the full runs were in progress. AGENTS.md asks for author contact only after an independent attempt, so this is recorded as an exception to that timing | Pending. Nothing has changed. Any weights received will not replace training, produce or select target values, or alter protocol A1 |

Not a Team S data-access request. Alternative B of `target-eval-route` (a Team S request to the challenge organisers) was not requested.

## Gates

1. **`target-eval-route`** (target) — resolved 2026-11-05: alternative A, operator-made Kaggle late submissions; B not requested (reasons above).
2. **`data-kaggle-competition`** (data) — resolved 2026-11-05: Restricted Environment `$MODAL_ENV`, deviation `restricted-dataset-environment`.
3. **`data-kaggle-download-forbidden`** (data) — resolved 2026-11-06: the competition rules had not been accepted yet; the operator accepted them and the rerun succeeded.
4. **`protocol-training-recipe`** (protocol) — resolved 2026-11-09: **A1**, the published code plus the output-only video save patch; A2 and B rejected (reasons under Guesses and deviations). *Disclosure note (2026-11-10):* A1 described the SL-GCN training config only as "SAM-SLR's train config". It did not disclose that the authors' fold configs carry `is_vector: False` where SAM-SLR's carry True, or that their schedule fields conflict with their checkpoint names. The approval was given without that evidence.
5. **`compute-full-plan`** (compute) — resolved 2026-11-09 at CHF 300 / 250 L4 GPU-hours (operator). Raised 2026-11-10 to **CHF 450 / 380 L4 GPU-hours** by the project lead in #repro-sign-team-r, with the operator's limit of at most 30 GPUs at once; the original decision is kept as history.
6. **`is-vector-normalization`** (protocol) — resolved 2026-11-10 by the operator, choosing none of the options offered (V1 test as trained, V2 authors' configs as-is, V3 retrain bone only). `is_vector: True` is a dead end. V1 runs as conditional evidence only. Every field of the authors' fold configs is diffed against the config used. Then bone, jm and bm (and joint if it mismatches) are retrained with the authors' settings in a new attempt group, reusing the keypoints, unless the authors' own configs conflict.
7. **`slgcn-retrain-settings`** (protocol) — resolved 2026-11-10 by the operator: **R1**. Bone, jm and bm (15 jobs) are retrained with `is_vector: False` and SAM-SLR's schedule, on the same folds and keypoints, in a new attempt group.
   - **Why R1.**
     - `is_vector: False` is corroborated independently: the authors' released weights reproduce their logits under it.
     - The schedule fields in their fold configs (100 epochs, lr 0.01, step 50, warm-up 0) are contradicted by their own checkpoint names (`…_250/epoch-176` to `epoch-249`), which a 100-epoch run cannot produce. Following them literally would mean choosing evidence known to be false.
     - The 250-epoch length is corroborated by those names. lr 0.1, steps 150/200 and warm-up 20 are inferred from SAM-SLR, consistent with but not confirmed by the authors' artifacts (listed under guesses).
     - The fold configs' schedule fields may simply be an unmodified SAM-SLR test template.
   - **Joint is not retrained.** Its feeder settings match the authors' (`is_vector: False` in both), and its only other differences are the schedule fields resolved above.
   - **Rejected: R2.** It would retrain all four streams with every field from the authors' configs taken literally, which contradicts their checkpoint names.
   - **Estimate.** About 117 L4-h, USD 132 and 7.9 h wall time with 15 GPUs at once, within the 30-GPU limit. The attempt total comes to about 327 L4-h / USD 386.
