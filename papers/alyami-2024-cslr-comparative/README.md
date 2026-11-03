# Alyami & Luqman (2024) — Comparative Study of CSLR Techniques: reproduction

**Paper ID:** `8d2b672d06ddb561f890798d7c1e434389c34202`

**Citation:** Sarah Alyami and Hamzah Luqman. *A Comparative Study of Continuous Sign Language Recognition Techniques.* arXiv:2406.12369v1, 2024. A revised version appeared in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, 2025, pp. 4923–4932.

**Paper:** https://arxiv.org/abs/2406.12369v1 · **Code/artifacts:** the study did not publish its own code. Each evaluated model's official repository is used (see Source provenance).

**Preference level:** 2

The preference level is expected rather than confirmed; see `status.preference_level_basis`.

**Pipeline status:** `partial` — interim: 5 of 30 in-scope targets have been produced while the blocks run. The compute gate `compute-full-table` was resolved on 2026-11-02 with an absolute ceiling of USD 2,000.

**Numerical agreement:** `does_not_agree` — interim. The tolerance is ±1.0 WER point, from the paper's own "around 1% fluctuations" for its Phoenix re-runs. SMKD is within it. VAC (+7.73), TLP (+4.58), SEN (+4.28) and CorrNet (+1.64) are not. This will be reassessed as targets complete; see `status.numerical_agreement_basis`.

**Attempt dates:** 2026-10-26 (intake and gates); continued from 2026-11-02

## Reproduction agents

| Agent ID | Model and version | Agent application | Contribution | Attribution evidence / unknowns |
| --- | --- | --- | --- | --- |
| `claude-opus-5-5-claude-code` | Claude Opus 5.5 (`claude-opus-5-5`) | Claude Code 2.1.310 | Assignment intake, version resolution, source search, dataset inspection on Modal `repro-sign`, target/protocol/compute gates | Session system context names the model and ID. `claude --version` returned 2.1.310. No runs executed. |
| `claude-opus-5-5-claude-code-2026-11-02` | Claude Opus 5.5 (`claude-opus-5-5`) | Claude Code 2.1.316 | From 2026-11-02: gate updates, data population, container, preflights, estimates, launching and monitoring full runs | Session system context names the model and ID. `claude --version` returned 2.1.316. |

## Scope and target contract

The assignment is a single "reproductions" export record together with the arXiv v1 PDF. The record's `what_to_reproduce` is "Table 1".

The record has paper `status: final` but no `confirmation` field, and it is not a top-level candidate array. `ingest_candidate.py` therefore cannot accept it. It is recorded as a `direct_user_request` that preserves the export with emails redacted, following the precedent in `papers/pu-2024-siformer`.

**Target version (gate `target-table-version`, resolved).** The two paper versions have materially different Table 1s:

- **arXiv v1 Table I:** 5 models × 6 settings.
- **ICCVW 2025 Table 1:** 7 models × 8 settings. It adds Swin-MSTP, SlowFastSign and CSL-Daily, uses different epochs, and changes some values.

The operator chose the arXiv v1 Table I, which is the supplied PDF. That gives 30 targets: VAC, SMKD, TLP, SEN and CorrNet, each evaluated on six settings:

- Phoenix2014 Signer-Dep
- Phoenix2014 Signer-Indep (SI5)
- ArabSign Signer-Dep
- ArabSign Signer-Indep (mean of 6 leave-one-signer-out folds, Table II)
- GrSL Signer-Indep (signer 3)
- GrSL Unseen-Sent

The metric is test WER in percent: sclite corpus-level (S+D+I)/N, Eq. 1. None of the values is a copied baseline. The paper re-ran the Phoenix Signer-Dep column itself, and the other columns are new benchmarks. The paper states no seeds or checkpoint rule.

**Protocol (gate `protocol-lr-epochs`, resolved).** The paper says it "followed the same experimental settings described in each study". The same paragraph then states Adam, lr 1e-3, 40 epochs, ×0.2 at epochs 20 and 35, and batch 2 for all models. Every pinned upstream config instead uses lr 1e-4, and TLP/SEN train for 80 epochs (steps 40/60). The operator chose the paper-stated protocol for every model: lr 1e-3 and 40 epochs with steps 20/35. Everything else is each repo's unchanged config.

The ICCVW 2025 version (Sec. 3.3) instead says TLP and SEN were trained for 80 epochs, with steps 30/60. The decision stands because arXiv v1 is the assigned paper. Yet the TLP and SEN Table I values are the same in both versions: all six TLP cells and five of six SEN cells are identical at one decimal. So the change in stated schedule did not come with new numbers, and the evidence does not show which schedule produced them. Before the full TLP and SEN runs, the measured cost of an 80-epoch schedule will be reported to the operator for a decision.

| Target ID | Paper location | Published WER |
| --- | --- | ---: |
| vac-phoenix2014-signer-dep / -signer-indep | Table I | 21.94 / 40.40 |
| vac-arabsign-signer-dep / -signer-indep | Table I (SI = Table II avg) | 0.2787 / 6.97 |
| vac-grsl-signer-indep / -unseen-sent | Table I | 29.60 / 74.92 |
| smkd-phoenix2014-signer-dep / -signer-indep | Table I | 22.31 / 43.08 |
| smkd-arabsign-signer-dep / -signer-indep | Table I | 0.2091 / 4.24 |
| smkd-grsl-signer-indep / -unseen-sent | Table I | 30.80 / 61.10 |
| tlp-phoenix2014-signer-dep / -signer-indep | Table I | 22.00 / 35.70 |
| tlp-arabsign-signer-dep / -signer-indep | Table I | 0.2439 / 2.90 |
| tlp-grsl-signer-indep / -unseen-sent | Table I | 6.40 / 75.50 |
| sen-phoenix2014-signer-dep / -signer-indep | Table I | 20.20 / 34.40 |
| sen-arabsign-signer-dep / -signer-indep | Table I | 0.2962 / 7.22 |
| sen-grsl-signer-indep / -unseen-sent | Table I | 8.09 / 74.80 |
| corrnet-phoenix2014-signer-dep / -signer-indep | Table I | 19.90 / 33.60 |
| corrnet-arabsign-signer-dep / -signer-indep | Table I | 0.2265 / 7.65 |
| corrnet-grsl-signer-indep / -unseen-sent | Table I | 1.50 / 64.70 |

## Source provenance

| Artifact | Canonical source | Pinned revision / SHA-256 | Role |
| --- | --- | --- | --- |
| Paper PDF (target) | https://arxiv.org/pdf/2406.12369v1 | `db652a81563f9e5c7901ba37df4b69164c28c9acfae858ffff682441ae9eecf6` | Target and protocol |
| Paper PDF (ICCVW 2025) | openaccess.thecvf.com, ICCV2025W/MSLR | `922a2038faac61b3eacbefb5be417e5dbf126b3655814005189f06391dd5aa63` | Version comparison only |
| VAC + SMKD code | https://github.com/VIPL-SLP/VAC_CSLR | `71f3e0334fbc8cecc7ce9816ec69781068abaac0` | VAC; SMKD via `share_classifier`/`weight_norm` (README) |
| TLP code | https://github.com/hulianyuyy/Temporal-Lift-Pooling | `21be215c6ba749d98ce6a51f4b84cc9d5029bfe6` | TLP |
| SEN code | https://github.com/hulianyuyy/SEN_CSLR | `c9a4c7ebd0489e22496c00f9f31d930c0d968ad8` | SEN |
| CorrNet code | https://github.com/hulianyuyy/CorrNet | `a812814f4deefda0b75ab1dcc90b106c62912263` | CorrNet |
| ArabSign split CSVs | https://github.com/Hamzah-Luqman/ArabSign `ArabSign-EncoderDec.zip` | repo `8f6e127`, zip `6980aec2…eb0` | Per-signer train/test lists |

Locations searched on 2026-10-26:

- The arXiv abs page, which has only v1 and no code link.
- The ICCVW open-access page, which has no code or supplement.
- The dataset author's GitHub repositories.
- GitHub searches for SMKD, self-mutual distillation, ArabSign and GrSL CSLR.
- An exact-title web search.

None of these turned up a repository for the comparative study itself.

## Results

Five of the 30 targets have been produced so far (block 1, Phoenix SD); the other 25 are running or not yet launched. The WER is sclite's exact (S+D+I)/N on the test split. The difference is reproduced minus published (arXiv v1 Table I), so positive means a higher error.

| Target | Published | Reproduced | Difference | Selected epoch (dev WER) | Runs |
| --- | ---: | ---: | ---: | --- | --- |
| VAC, Phoenix2014 Signer-Dep | 21.94 | 29.67 | +7.73 | 35 (29.64) | `full-vac-phoenix-sd-001`, `eval-vac-phoenix-sd-001` |
| SMKD, Phoenix2014 Signer-Dep | 22.31 | 22.14 | −0.17 | 36 (21.57) | `full-smkd-phoenix-sd-001`, `eval-smkd-phoenix-sd-001` |
| TLP, Phoenix2014 Signer-Dep | 22.0 | 26.58 | +4.58 | 39 (26.19) | `full-tlp-phoenix-sd-001`, `eval-tlp-phoenix-sd-002` |
| SEN, Phoenix2014 Signer-Dep | 20.2 | 24.48 | +4.28 | 36 (24.75) | `full-sen-phoenix-sd-001`, `eval-sen-phoenix-sd-001` |
| CorrNet, Phoenix2014 Signer-Dep | 19.9 | 21.54 | +1.64 | 38 (21.34) | `full-corrnet-phoenix-sd-001`, `eval-corrnet-phoenix-sd-002` |

Counts on the 629 test sentences (6,504 words): VAC 1,930 errors, SMKD 1,440, TLP 1,729, SEN 1,592, CorrNet 1,401.

**Block 1 observations.** These are recorded as evidence, without interpretation, and are also in `compute-full-table.resolution.block_ledger`:

- The lowest dev WER came late for all five models. The best epochs, 0-based out of 40, were VAC 35, SMKD 36, TLP 39, SEN 36 and CorrNet 38.
- TLP's dev WER was still falling at its final (40th) epoch: 26.7, 26.8, 26.5 and 26.2 at epochs 36–39. The ICCVW 80-epoch schedule for TLP was not adopted (see Guesses and deviations).
- VAC is +7.73 against the paper, while SMKD is within 0.17. SMKD comes from the same repository, with the same data lists and scoring, and differs only in `share_classifier` and `weight_norm`. Both use SeqCTC 1 + ConvCTC 1, the loss weights recorded as a guess. This rules out a data or scoring fault for this block.

**Convergence curves.** Every full run's upstream per-epoch dev WER log (`work/dev.txt`) is copied verbatim to `artifacts/{run_id}-dev.txt`. Each copy is recorded as artifact `{run_id}-dev-log` with its sha256 and its source URI on the results Volume. Later blocks do the same.

## How to repeat this

Run from the repository root. Every Modal command goes through the `repro-sign` wrapper.

```bash
W=.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh
A=papers/alyami-2024-cslr-comparative/modal_app.py

papers/alyami-2024-cslr-comparative/data.sh       # populate/verify the datasets Volume (idempotent)
$W run $A::launch_prepare_frames --dataset phoenix2014   # 256x256 frame archives on the results Volume
$W run $A::launch_prepare_frames --dataset arabsign
$W run $A::launch_prepare_frames --dataset gsl
```

The preflight, training and evaluation commands are listed per run in `reproduction.json.runs`. Full runs and their evaluation have this form:

```bash
$W run --detach $A::launch_train --model {vac|smkd|tlp|sen|corrnet} --setting {setting} --run-id {run_id} [--dev-rule {rule}]
$W run --detach $A::launch_evaluate --train-run-id {run_id} --eval-run-id {eval_run_id}
$W run $A::launch_arabsign_split_record   # re-derives the ArabSign split hashes recorded in the gates
$W run --detach $A::launch_run_full --model {model} --setting {setting} --run-id {run_id} --eval-run-id {eval_run_id} --max-gpu-hours {ceiling} [--dev-rule {rule}]   # blocks 2-5: train + evaluate in one detached call
```

The settings are `phoenix-sd`, `phoenix-si`, `grsl-si`, `grsl-unseen`, `arabsign-sd` (`--dev-rule stratified-709-seed42`) and `arabsign-loso-01` … `-06` (`--dev-rule final-epoch`).

## Data provenance and permissions

| Dataset | Splits | License / cloud basis | Path in Volume `datasets` | State |
| --- | --- | --- | --- | --- |
| Phoenix2014 v3 | multisigner 5672/540/629; SI5 4376/111/180 | CC BY-NC-SA (RWTH), non-commercial research | `rwth-phoenix-2014/raw/phoenix2014-release` | Present; checked. All SI5 IDs resolve to multisigner frame folders. Corpus CSV hashes are in `reproduction.json`. |
| ArabSign (RGB) | Signer-Dep 7,489 train+val / 1,841 test (of 7,492 / 1,843 listed); 6 LOSO folds | Non-commercial; author-granted permission via Team S | `arab-sign/RGB/videos` | Extracted 2026-11-02 by `data.sh`: 9,330 mp4 with per-file SHA-256. 5 listed videos are absent from the RGB release. The 709-sample validation subset is unpublished. |
| Continuous GSL | SI (signer 3) 8821/588/881; unseen-sentence 8189/1063/1043 | CC BY 4.0 (Zenodo 3941811) | `gsl-continuous` | Populated 2026-11-02 by `data.sh`: 15 RGB zips plus `supplementary.zip`, 40.17 GB, md5-verified. The 6 split files match their recorded SHA-256. |

`data.sh` runs two population steps:

- `populate_gsl` downloads the Zenodo RGB zips and the Drive split files into `gsl-continuous/`. It writes nothing elsewhere.
- `extract_arabsign` extracts the six RGB 7z archives already on the Volume to `arab-sign/RGB/videos/`. It checks each archive against the import manifest and reconciles the videos with the dataset author's split CSVs.

Both steps are the only ones that mount `datasets` writable. Every other function mounts it read-only.

The GSL Drive folder labelled "GSL SD" actually holds the `*_unseen` files. None of its 1,043 test sentences occurs in training, and they cover 45 unique sentences, so this folder matches the paper's Unseen-Sent split. "GSL SI" holds signer 3 out with seen sentences, matching the paper's Signer-Indep split.

The five ArabSign videos missing from the RGB release are `01/test/0006`, `01/test/0044`, `01/train/0020`, `02/train/0020` and `05/train/0042`, all with `(timestamp)_c.mp4` stems. Signer-Dep test therefore has 1,841 of 1,843 clips. In addition, 35 file stems carry a signer/sentence prefix that differs from their folder. The folder path is what the author's CSVs list, so it defines signer, split and sentence.

Frames for all three datasets are resized to 256×256 PNG with the upstream `resize_img` operation (`cv2.resize`, `INTER_LANCZOS4`) and archived on the paper's results Volume `alyami-2024-cslr-results`:

- **Phoenix2014:** VAC's `preprocess/dataset_preprocess.py`, byte-identical in all four repos, runs unchanged on a local copy.
- **ArabSign:** mp4s are decoded with `simple-video-utils` 0.7.4 (rgb24, checked against PyAV), then converted to BGR and resized.
- **GSL:** jpg frames are decoded from the zips and resized.

## Environment and patches

`Dockerfile` starts from `ghcr.io/sign-language-processing/reproduction:latest` (NGC PyTorch 26.02: torch 2.11.0a0, CUDA 13.1, cuDNN 9.19, Python 3.12, numpy 2.1). On top of that it adds:

- SCTK `sclite` (`9688a26`);
- `ctcdecode` (`c90ad94`);
- `simple-video-utils` 0.7.4 with `av` 18.1.0;
- `lmdb` 1.5.1, which TLP's and SEN's `dataset/dataloader_video.py` import at module level;
- the four repos at their pinned commits under `/opt/repos/{vac,tlp,sen,corrnet}`, each with one correctness patch (`patches/{repo}-ctc-without-cudnn.patch`, below).

The repos' own Python pins (numpy 1.20, scipy 1.7, opencv 4.5.5) have no Python 3.12 wheels. They are replaced by the oldest releases that do and that still ship `scipy.misc`, as in `papers/yu-2025-cvsign`.

Each run works on its own copy of one repo. The harness adds only data files and configs to that copy:

- the setting's info lists, gloss dict and STMs, written with the repo's own `generate_gt_stm` and `sign_dict_update`;
- a `configs/{dataset}.yaml`;
- `configs/run.yaml`. This is the pinned `configs/baseline.yaml` with only the gate-approved keys changed, and each run records the exact delta.

Tracked upstream files stay unchanged, and each run records the hash of `git diff HEAD`.

Phoenix Signer-Dep uses each repo's own `phoenix2014` dataset config. SEN and CorrNet use their committed lists, which are identical to each other. VAC and TLP commit no lists, so they use the output of the byte-identical `preprocess/dataset_preprocess.py`.

Every other setting is written in the same format under a dataset name containing `phoenix`. In TLP, SEN and CorrNet that substring is the only thing that selects the train/dev/test loaders and the frame-folder glob. VAC always reads `./preprocess/phoenix2014/`. So no upstream code needs changing for the new datasets.

Patches and environment fixes, each justified by this paper's own build or preflight:

| Item | Why | Evidence |
| --- | --- | --- |
| `patches/ctcdecode-cxx17.patch` | Unpatched `ctcdecode` does not compile against current PyTorch: `#error C++17 or later compatible compiler is required to use PyTorch`. The patch builds as C++17 and drops five KenLM dynamic exception specifications. No decoder logic changes. It is identical to the yu-2025-cvsign patch. | Image build `im-zwwdoz4GF7RwJO4BnSPdG6` failed 2026-11-02 |
| `simple-video-utils==0.7.4` installed in the Dockerfile | The published `reproduction:latest` image lacks it, although the root Dockerfile pins it. | Build `im-L4hfuloTR54towJj5WD1nD`: `ModuleNotFoundError` |
| `av==18.1.0` | `simple-video-utils` requires `av>=18`, which resolved to 19.0.2. That version rejects the `metadata_errors` argument `simple-video-utils` passes. | App `ap-OMIRjJ8IzhzSco6yzVHx7u`: `open() got an unexpected keyword argument 'metadata_errors'` |
| `patches/{vac,tlp,sen,corrnet}-ctc-without-cudnn.patch` | On this stack `nn.CTCLoss` sends the repos' CTC call (CPU int32 targets) to cuDNN even when the two clips in a batch differ in length. PyTorch 1.x used cuDNN only when every input length equalled T. cuDNN's loss value is right, but its gradient is wrong for unequal lengths. Disabling cuDNN for the CTC call alone restores the native computation; nothing else changes. Same change as yu-2025-cvsign. | Fixed-batch gradients vs a float64 CPU reference: cosine 0.06–0.96 with cuDNN, 1.0000 without, and equal-length batches match. Under AMP, GradScaler skipped 26–50 of 60 updates with cuDNN against 5–8 without. With the patch, the as-is and cuDNN-off arms coincide (`ctc-diagnostics-*-phoenix-001`, `-patched-001`, `preflight-*-002/003`). |
| `lmdb==1.5.1` | TLP's and SEN's loaders `import lmdb`; no config uses the lmdb datatype. | `preflight-{tlp,sen}-*-001`: `ModuleNotFoundError` |
| `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` | Upstream resumes with a bare `torch.load` of checkpoints that hold numpy RNG state. PyTorch ≥ 2.6 refuses these by default. This is an environment variable, not a code change. | As in yu-2025-cvsign |

Two full-run settings change no computation:

- `PYTORCH_ALLOC_CONF=expandable_segments:True`. VAC and SMKD reached 45.2 GB of the L40S's 46 GB with the longest clips, and this setting only changes how freed memory is reused.
- 32 GiB of host memory instead of 64. Preflight host peaks were at most 19.5 GiB.

`ctc_check.py` is a diagnostic only. It builds each repo's own `Processor` from `main.py`, runs its own `seq_train` on real preflight batches, and compares CTC gradients on fixed batches.

## Execution evidence

All runs are in `reproduction.json.runs`, with Modal app and function-call IDs, image IDs, timestamps and declared ceilings. Outputs live under `runs/{run_id}/` on the Modal Volume `alyami-2024-cslr-results`.

**Data jobs, 2026-11-02:**

| Job | App | Result |
| --- | --- | --- |
| `extract_arabsign` | `ap-3EEqRJGxaxKfMN9NLO8W1O` | 9,330 mp4 |
| `populate_gsl` | `ap-Sg1meM9jpuHIbOLcF5I5bd` | 40.17 GB |
| Phoenix frames | `ap-Iew7yQSxjZSIdEXuS1iphB` | 70.2 GB, sha256 `3833a87f…` |
| ArabSign frames | `ap-0LKt5eSKIfpberp3xYdJon` | 1,104,581 frames, 91.5 GB, sha256 `27bf5694…` |
| GSL frames | `ap-RIAXU8Tv7XUNX8wBsYkvB3` | 1,036,155 frames, 91.1 GB, sha256 `09f596dd…` |

**Preflights (15, one per model × dataset).** Each uses 120 random plus the 16 longest training clips, and 40 random plus the 8 longest dev and test clips. It runs upstream `main.py` for 2 epochs, resumes from the epoch-1 checkpoint for a 3rd, then runs the test phase with sclite scoring. Every model × dataset passed. ArabSign used a provisional random 709-clip dev list, for cost measurement only.

Rates per model (train / eval ms per frame, with peak GPU memory from nvidia-smi):

| Model | Phoenix | GrSL | ArabSign | Peak GPU (GB) |
| --- | --- | --- | --- | --- |
| VAC | 0.73 / 1.01 | 0.46 / 2.13 | 0.77 / 1.01 | 45.2 |
| SMKD | 0.73 / 1.01 | 0.51 / 1.85 | 0.77 / 1.01 | 42.9 |
| TLP | 0.72 / 0.63 | 0.80 / 0.85 | 0.81 / 0.58 | 22.6 |
| SEN | 1.00 / 0.76 | 0.79 / 1.14 | 1.04 / 0.87 | 21.8 |
| CorrNet | 0.86 / 0.88 | 0.70 / 1.28 | 0.83 / 1.01 | 29.8 |

Each upstream checkpoint is about 385 MB.

**Scoring check (`score_check`).** A perfect hypothesis CTM scores 0.0% on Phoenix SI5, both GrSL splits, ArabSign SD and one LOSO fold. Dropping each sentence's first gloss gives exactly sentences ÷ reference words, so upstream's `preprocess.sh` plus sclite handles Greek and Arabic glosses unchanged. Apps are `ap-KcrIWUrzHeRYOQbuw2fxzc` and `ap-INgCesOLEX4oXS0KpfsCMZ`. Because sclite logs WER to one decimal and the paper reports ArabSign Signer-Dep WERs like 0.2787, evaluation also records sclite's raw error and word counts.

**Measured estimate (`compute-full-table.resolution.measured_estimate`).** This assumes 40 epochs for all models at USD 2.96/h (L40S, 16 cores, 32 GiB), 15 min overhead per run, and no retry margin:

| Block | Runs | USD |
| --- | ---: | ---: |
| Phoenix SD | 5 | 122 |
| Phoenix SI | 5 | 88 |
| GrSL SI | 5 | 115 |
| GrSL Unseen-Sent | 5 | 123 |
| ArabSign SD | 5 | 129–161 (by dev rule) |
| ArabSign LOSO | 30 | 803–965 (by checkpoint rule) |
| **Total** | 55 | **about 1,380** (cheapest rules) |

Per run this is 5.3–10.6 h. The paper reports about 10 h per Phoenix/GrSL run on an RTX A6000. The per-frame rates come from 136-clip epochs with worker start-up included, so they are more likely high than low.

**Spend so far:** USD 16.08 on 2026-11-02, from Modal billing summed over this paper's apps, covering data jobs, image builds, preflights and diagnostics.

**Block 1, Phoenix SD (complete, 2026-11-03 01:45 UTC).**

| Model | Training time (h) | Estimate (h) |
| --- | ---: | ---: |
| VAC | 6.15 | 7.65 |
| SMKD | 6.16 | 7.65 |
| TLP | 4.61 | 7.23 |
| SEN | 8.32 | 9.85 |
| CorrNet | 6.94 | 8.75 |

Each model trained in a single attempt. Two evaluations needed a second attempt, both rerun unchanged:

- The first TLP evaluation hit a GPU hardware fault (XID 154, CUDA unknown error) before loading anything; the rerun is `eval-tlp-phoenix-sd-002`.
- The first CorrNet evaluation was stopped when the local Modal client lost its connection, before the function had written anything; the rerun is `eval-corrnet-phoenix-sd-002`. Evaluations now run with `modal run --detach`.

The block cost USD 98.15 (Modal billing per app, training and evaluation), against an estimate of USD 122. Cumulative spend is USD 114.69 of USD 2,000. The block ceiling is USD 228. That is the sum of the per-run GPU-hour ceilings (14 + 14 + 16 + 14 + 18 GPU-h) at USD 3.0/h, against an estimate of USD 122.

| Run | Launched (UTC) | App | Declared ceiling |
| --- | --- | --- | --- |
| `full-vac-phoenix-sd-001` | 14:05 | `ap-Bq2sMC5UcEDiiAY8EoOOXV` | 14 GPU-h |
| `full-smkd-phoenix-sd-001` | 14:05 | `ap-HCnhtQKUVw36UvqaRMevLa` | 14 GPU-h |
| `full-corrnet-phoenix-sd-001` | 14:05 | `ap-LpdMiHskC3BGNTMPJhaVmR` | 16 GPU-h |
| `full-tlp-phoenix-sd-001` | 15:28 | `ap-brLTkY5v3XGxA5aUMV3ioZ` | 14 GPU-h |
| `full-sen-phoenix-sd-001` | 15:28 | `ap-q7NTK645rUEFQURf7rlJGn` | 18 GPU-h |

**Blocks 2–5 (Phoenix SI, GrSL SI, GrSL Unseen-Sent, ArabSign SD): 20 unattended runs, launched 2026-11-03 08:36–08:41 UTC.**

Each run is a single detached Modal call of `run_full`, launched as `modal run --detach … ::launch_run_full`, one call per app. The call:

1. trains with the unchanged protocol (lr 1e-3, 40 epochs, steps 20/35);
2. evaluates the dev-best checkpoint on dev and test;
3. writes `runs/{eval_run_id}/evaluation.json`, sclite's raw reports and the hypothesis CTMs to the results Volume, with one record per attempt in `runs/{run_id}/run_full.jsonl`.

No step depends on this session, a local process or an open connection.

Each run's declared GPU-hour ceiling is enforced inside the function. A heartbeat records container time per Modal attempt in `gpu_seconds.json`, and upstream is stopped when the ceiling is reached. A stopped run is recorded as `stopped` and is not retried. Modal retries (up to 3) resume from the newest checkpoint that loads.

The 20 runs together are estimated at 154 GPU-h (about USD 465 at USD 3.03/h). The sum of their ceilings is 276 GPU-h, a worst case of USD 836. Every app and function-call ID is in `reproduction.json.runs`, with a dashboard link.

**Proof of detachment.** `full-vac-phoenix-si-001` was launched first, at 08:36:37. Its local client exited at 08:36:51. At 08:39:28, with no local `modal` process, its app (`ap-TExBCfqcV25PagMad0KbLc`) was `ephemeral (detached)` with one running task. Its call (`fc-01M6G8VAYT4NYXFWJNAZ9SSKDV`) was running, and it had written its own start record to the Volume. Only then were the other 19 launched.

**Harness checks.**

- `check-run-full-complete-002` ran `run_full` on the finished SMKD Phoenix SD run. It skipped training, re-evaluated epoch 36 and got 1,440 / 6,504 = 22.1402%, identical to `eval-smkd-phoenix-sd-001`.
- `check-run-full-ceiling-002` exercised the ceiling stop with a 0.05 GPU-h budget. Upstream was killed after 1.4 s, the attempt was recorded as `stopped` (`gpu_hour_ceiling_reached`), and the call returned with no retry.

The first Phoenix SI launch, at 08:18, used the train-only function. It was stopped during frame staging, before any checkpoint, and replaced by `run_full`. The stopped calls are listed under each run's `compute.superseded_calls`.

At 08:41:52, 5 runs had a GPU and 15 were waiting for one. Queueing is allowed and is recorded when the runs are collected.

## Guesses and deviations

| Detail | Paper/evidence says | This attempt uses | Rationale |
| --- | --- | --- | --- |
| Learning rate, TLP/SEN epochs | arXiv v1: lr 1e-3, 40 epochs (steps 20/35) for all. Upstream: lr 1e-4, TLP/SEN 80 epochs (steps 40/60). ICCVW 2025: lr 1e-3, TLP/SEN 80 epochs (steps 30/60). | arXiv v1 values for all five models, in every block including ArabSign LOSO | Operator decision at gate `protocol-lr-epochs`, confirmed on 2026-11-02 for TLP/SEN after the measured 80-epoch cost was reported (about USD +578 over all blocks). arXiv v1 is the assigned paper. ICCVW's 80 epochs (steps 30/60) is a known alternative that was not adopted. |
| TLP/SEN schedule evidence across versions | Table I vs ICCVW Table 1 (checked 2026-11-02 from both PDFs). All six TLP cells and five of six SEN cells are identical at one decimal. SEN GrSL Signer-Indep is 8.09 in v1, 8.0 in ICCVW Table 1 and 8.1 in ICCVW Fig. 3. Across all 30 shared cells, 26 are identical at one decimal, 3 match only by truncation (SMKD Phoenix SI 43.08→43.0, SEN GrSL SI 8.09→8.0, CorrNet ArabSign SI 7.65→7.6), and 1 differs (SMKD Phoenix SD 22.31→22.1). | 40 epochs (arXiv v1) | The numbers did not change when the stated schedule went from 40 to 80 epochs, so the evidence does not show which schedule produced them. Recorded as a known alternative, not adopted. |
| VAC / SMKD loss (guess) | The paper says each study's settings were followed and names no loss weights. VAC_CSLR `README.md` line 36 at `71f3e03` states "The VAC result is corresponding to the setting of `loss_weights: SeqCTC: 1.0, ConvCTC: 1.0`", and VAC+SMKD adds `share_classifier`/`weight_norm: True`. `baseline.yaml` comments out ConvCTC and a `Dist: 10.0`. | SeqCTC 1 + ConvCTC 1; SMKD adds share_classifier and weight_norm | A guess from the README's own mapping of config to result. Line 33 also labels the VAC row "ResNet18+VAC (CTC only)". Dist is not part of either. |
| Seed | Not stated | 0 (upstream default) | Single reported number per cell |
| Checkpoint selection | Not stated | Lowest dev WER, earliest epoch on ties. This is TLP/SEN/CorrNet's `_best_model.pt` rule. VAC_CSLR has no best-model rule, so per-epoch checkpoints (`--save-interval 1`) let the same rule apply. | Upstream behavior; uniform across models |
| GPUs | v1: "RTX A6000 with two 48GB GPUs". ICCVW: batch 2 "on a single GPU". | One L40S (48 GB) | Matches ICCVW and avoids VAC's documented multi-GPU BatchNorm issue |
| Phoenix SI5 lists | SI5 corpus files. Upstream drops pandas row 2390 of the multisigner train CSV, the broken clip `13April_2011_Wednesday_tagesschau_default-14`, which has 16 frames and is not resized. | SI5 lists drop that clip by ID | In the SI5 file row 2390 is a different, valid clip, so dropping by index would drop the wrong clip |
| ArabSign gloss labels | ArabSignGroundTruth.txt "Sign Language Sentence". The author's CSVs carry it as `caption`, plus a processed `procCaption` that strips the definite article from some words. The ArabSign paper (Luqman, FG 2023; arXiv 2210.03951v1, sha256 `7be6d45d…`) states in Sec. III-B "The dataset's sentences consist of 155 signs, and the dataset's vocabulary consists of 95 signs" and "The average sentence length is 3.1 signs" (Table II: "Vocab. size 95"). | The published "Sign Language Sentence" (= `caption`) | Neither version of the paper says which labels were used. The chosen labels give 50 sentences, 155 sign tokens and 3.1 signs per sentence, matching the ArabSign paper exactly, and 93 distinct glosses against its 95. `procCaption` gives 90, further away. The two-gloss gap is not explained by either column, so this supports the choice but does not prove it. Details: `reproduction.json.datasets[arabsign].vocabulary_check`. |
| ArabSign videos | 9,335 samples | 9,330 (5 absent from the RGB release; test 1,841 of 1,843) | Data as released; listed in `reproduction.json.datasets` |
| ArabSign Signer-Dep validation | 6,783 train / 709 validation (ICCVW only). The subset itself is unpublished. The dataset author's code draws validation with `train_test_split(random_state=42, stratify=sentence ID)`. | 709 of the 7,489 usable training rows, from `train_test_split(test_size=709, random_state=42, stratify=sentence ID)` with scikit-learn 1.8.0 on rows in sorted archive-path order. That gives 6,780 train / 709 dev / 1,841 test, all 50 sentences in each. Dev-best selection. | Operator decision at gate `arabsign-sd-validation`, 2026-11-02. Split hashes are in the gate's resolution. |
| ArabSign LOSO checkpoint | No dev set. arXiv v1 Sec. V, on CorrNet: "Better results are likely to be obtained by … adopting early stopping". | Train 40 epochs on all clips of the five training signers and score the final epoch (39). Upstream's per-epoch dev loop reads a fixed 50-clip token list (the first training clip of each sentence), which is never used for selection. | Operator decision at gate `arabsign-loso-checkpoint`, 2026-11-02, based on the early-stopping remark. Per-fold hashes are in the gate's resolution. |
| Frame resizing for ArabSign/GrSL | "Resized to 256x256" | Upstream `resize_img` (`cv2.resize`, `INTER_LANCZOS4`) to 256×256 PNG, the Phoenix pipeline's own operation | Same operation the repos apply to Phoenix |
| Checkpoint retention | Not applicable | Full runs keep only the newest epoch checkpoint (for resume) and the current dev-best (for selection), and delete superseded ones during training | Declared before launch. About 385 MB × 40 epochs × 55 runs would otherwise cost about USD 75 per month in Volume storage. |

## Attempts, failures, and dead ends

The 2026-10-26 session stopped before data work at the compute gate. On 2026-11-02 the following happened, in order:

- **Image build: `ctcdecode`.** The first build failed in `ctcdecode`, which justified reusing the C++17 patch.
- **Image build: `simple-video-utils`.** The second build failed because `reproduction:latest` lacks `simple-video-utils`.
- **ArabSign frames, first two failures.** The first failed on PyAV 19 (`metadata_errors`), which led to the pin `av==18.1.0`. The second failed on 35 file stems whose prefixes differ from their folders, so sample IDs now come from the path.
- **First preflights:**
  - `preflight-smkd-gsl-001` lost its GPU mid-run (`CUDA error: unknown error`, and `nvidia-smi` stopped reporting the device), and was retried unchanged.
  - TLP and SEN failed on the missing `lmdb` module.
  - The CorrNet runs passed, but the helper `ctc_check.py` itself raised, because CorrNet's `seq_train` returns `None`.
- **CTC diagnostics.** These found the cuDNN gradient error, which led to the four `ctc-without-cudnn` patches.
- **ArabSign labels.** `preflight-*-arabsign-001` was stopped after `score_check` showed that the harness compared the wrong CSV columns with the ground-truth file. The CSV columns are `caption` and `procCaption`, not an Arabic sentence and a sign sentence.
- **SEN log parsing.** SEN's preflights then failed twice in the harness's log parser, because SEN logs WER without a percent sign.
- **Unattended launch, 2026-11-03.** The first two `run_full` checks were launched without `--detach`. `launch_run_full` only spawns and returns, so their apps stopped 3 s later and the calls never ran. They were relaunched with `--detach` as `-002`. The five train-only Phoenix SI calls from 08:18 were stopped during staging and replaced by `run_full`. Full-run containers have 40 GiB RAM from block 2 on, because block 1 peaked at 26.8 of 32 GiB.
- **Block 1 evaluations.** `eval-tlp-phoenix-sd-001` hit a GPU hardware fault (XID 154). `eval-corrnet-phoenix-sd-001` lost its client connection. Both were reruns of the same command, and `evaluate()` now raises on a non-zero upstream exit instead of failing in the log parser.

Every model × dataset preflight passed within its declared attempt limit.

## Candidate flags, ethics, and human evaluation

The queue record flags no human evaluation and no ethical concerns, and `copied_scores` is `no`. The text confirms the paper re-ran every model and copied no baselines. All three datasets are existing recordings released for research, and no participants are involved.

## Author and team contact

None. The ArabSign access shown in the queue record was coordinated earlier by Team S.

## Gates

### Compute gate (resolved)

`compute-full-table` covers about 55 single-GPU trainings: Phoenix 10, ArabSign Signer-Dep 5, ArabSign LOSO 30, GrSL 10.

The estimate presented in `#repro-sign-team-r` on 2026-10-30 was about 420–610 L40S GPU-hours, or USD 1,350–2,000. It included about 15 GPU-hours of preflights and a 10% retry margin, at USD 3.22/h effective for an L40S with 16 cores and 64 GiB. The 30 ArabSign LOSO runs account for about USD 550–1,050 of that.

That estimate replaces the earlier GPU-only figure of USD 800–1,600, which left out CPU and memory charges, preflights and retries.

The project lead approved the full Table I on 2026-11-02, including the 30 ArabSign LOSO runs. The absolute ceiling is **USD 2,000**: about 620 L40S GPU-hours at USD 3.22/h, or about CHF 1,650. Blocks run in this order:

1. Phoenix SD
2. Phoenix SI
3. GrSL SI
4. GrSL Unseen-Sent
5. ArabSign SD
6. ArabSign LOSO

Each block declares its own ceilings, and cumulative spend is tracked against USD 2,000.

### Protocol decisions of 2026-11-02 (resolved)

- **TLP/SEN epochs** (`protocol-lr-epochs` follow-up). The measured cost of 80 epochs for TLP and SEN was reported: about USD +578 over all blocks, or +238 outside LOSO. TLP and SEN stay at 40 epochs with steps 20/35 in every block, as arXiv v1 states. ICCVW's 80 epochs (steps 30/60) is recorded as a known alternative that was not adopted.
- **`arabsign-sd-validation`: `stratified-709-seed42`.** 709 dev clips, sentence-stratified, seed 42, as in the guesses table. Split record: app `ap-4HwGUsgGtScHCYmacWBOAP`.

  | Split | Clips | fileid-list sha256 |
  | --- | ---: | --- |
  | train | 6,780 | `bd960085…` |
  | dev | 709 | `6407bcd9…` |
  | test | 1,841 | `d19c5fdc…` |

  The lists hash is `1d36ba96…`.
- **`arabsign-loso-checkpoint`: `final-epoch`.** No dev-based selection, based on v1's early-stopping remark. The per-fold train/test/token-dev hashes are in the gate's resolution.

No gate is open. With these decisions the projected total is about USD 1,396 before retries, including the USD 16 already spent (`compute-full-table.resolution.projection_after_decisions`).
