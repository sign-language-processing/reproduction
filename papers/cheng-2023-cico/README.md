# CiCo: Domain-Aware Sign Language Retrieval via Cross-Lingual Contrastive Learning

**Current work:** training reproduction reopened on 2026-10-03. Source and training-data preparation are in progress; no new full training run has launched.

**Completed evidence:** released-checkpoint evaluation only. The prior run did not train the visual encoders or CLCL model. The author archive contains test features, without train/dev features.

**Pipeline status:** partial

The 24 released-checkpoint targets are preserved; 24 independent training targets are also in scope and pending.

**Numerical agreement:** does_not_agree

**Preference level:** 2

**Claim-level assessment (2026-10-02):** The released-checkpoint evaluation corroborates the paper’s central reported advantage over SPOT-ALIGN (Section 4.2, Tables 1–2): How2Sign T2V/V2T R@1 remain 22.5/28.0 percentage points above published SA-COMB, and PHOENIX remains 13.7/17.1 points above it. Recall@5/10 ordering is also preserved; PHOENIX median ranks tie the baseline. CSL-Daily retains the performance level of the baseline supplied in Table 3, which has no competing-system row.

These comparisons use the published baseline scores; those systems were not rerun. Training reproducibility, component ablations (Tables 4–9), causal explanations of the gains and training-seed variability remain unassessed. This interpretation adds no numerical tolerance and leaves the declared exact-agreement result unchanged. A separate GPT-6/Codex agent reviewed these claims and updated the report on 2026-10-02; it executed no experiments, and its exact model/application versions were unavailable.

Yiting Cheng, Fangyun Wei, Jianmin Bao, Dong Chen and Wenqiang Zhang. CVPR 2023. [Paper](https://arxiv.org/abs/2303.12793). The paper PDF has SHA-256 `5a0bb54fe956baaad50d474c5a065d977801413f50c2b6dc5a2dfd9b3f44be44`. Tables 1–3 are on **PDF page 7**. The assignment targets the three Ours rows: 24 numbers across text-to-video (T2V) and video-to-text (V2T) recall at 1/5/10 and median rank. Another 48 copied comparison numbers are retained out of execution scope.

PHOENIX evaluation completed on all 642 test videos and queries, matching all eight published values. CSL-Daily evaluation completed on 1,176 videos grouped into 798 unique text queries; its recall differences range from −0.6 to +0.1 percentage points, with both median ranks unchanged. How2Sign full evaluation also completed on 2,348 videos and 1,969 text query groups. Its T2V R1 and R5 exceed the published values by 0.1 percentage points; the other six metrics match. All **24 requested checkpoint-evaluation targets** are produced. Pipeline completion refers to this evaluation scope and does not establish end-to-end training reproducibility.

| Dataset / direction | Metric | Published | Evaluated | Difference |
|---|---|---:|---:|---:|
| how2sign / T2V | R1 | 56.6 | 56.7 | +0.1 pp |
| how2sign / T2V | R5 | 69.9 | 70.0 | +0.1 pp |
| how2sign / T2V | R10 | 74.7 | 74.7 | +0.0 pp |
| how2sign / T2V | MedianR | 1.0 | 1.0 | +0.0 ranks |
| how2sign / V2T | R1 | 51.6 | 51.6 | +0.0 pp |
| how2sign / V2T | R5 | 64.8 | 64.8 | +0.0 pp |
| how2sign / V2T | R10 | 70.1 | 70.1 | +0.0 pp |
| how2sign / V2T | MedianR | 1.0 | 1.0 | +0.0 ranks |
| phoenix2014t / T2V | R1 | 69.5 | 69.5 | +0.0 pp |
| phoenix2014t / T2V | R5 | 86.6 | 86.6 | +0.0 pp |
| phoenix2014t / T2V | R10 | 92.1 | 92.1 | +0.0 pp |
| phoenix2014t / T2V | MedianR | 1.0 | 1.0 | +0.0 ranks |
| phoenix2014t / V2T | R1 | 70.2 | 70.2 | +0.0 pp |
| phoenix2014t / V2T | R5 | 88.0 | 88.0 | +0.0 pp |
| phoenix2014t / V2T | R10 | 92.8 | 92.8 | +0.0 pp |
| phoenix2014t / V2T | MedianR | 1.0 | 1.0 | +0.0 ranks |
| csl-daily / T2V | R1 | 75.3 | 75.1 | -0.2 pp |
| csl-daily / T2V | R5 | 88.2 | 87.6 | -0.6 pp |
| csl-daily / T2V | R10 | 91.9 | 91.7 | -0.2 pp |
| csl-daily / T2V | MedianR | 1.0 | 1.0 | +0.0 ranks |
| csl-daily / V2T | R1 | 74.7 | 74.8 | +0.1 pp |
| csl-daily / V2T | R5 | 89.4 | 88.9 | -0.5 pp |
| csl-daily / V2T | R10 | 92.2 | 92.3 | +0.1 pp |
| csl-daily / V2T | MedianR | 1.0 | 1.0 | +0.0 ranks |

Recall values are percentages; median ranks are one-based and lower is better. The pinned author evaluator and its handling of ties, query groups and retrieval direction remain unchanged. Numerical agreement is assessed at the paper's one-decimal precision, as declared before evaluation. Section 4.3 reverses directions in its prose; Table 1 headers and the executable author evaluator agree, so the table headers determine target mapping. Numerical agreement or disagreement is not a scientific success/failure judgment.

The CSL-Daily discrepancy was investigated without score tuning. The code and data content are unchanged from the original `0322431` release; subsequent CiCo changes concern file modes and the README. All official labels have both domain-agnostic and domain-aware features, and the released checkpoint, dataset-specific alpha, batch size 256, seed 42 and upstream metrics are retained. Environment drift or differences between the released checkpoint and the one used for the table remain possible, unproven explanations. The domain-aware archive contains 1,628 CSL test files, but the official labels select 1,176 videos; the extra 452 files are not silently added to evaluation.

The six copied rows below account for all 48 excluded comparison numbers. Every cell was verified against [SPOT-ALIGN, arXiv:2201.02495v1](https://arxiv.org/pdf/2201.02495v1), **Tables 6–7** (PDF SHA-256 `3665f77714b69d2f806b5cc49e320f050c9c8b9add957366625ff32a4600a379`). The later published version changes the How2Sign values, so substituting that version would obscure the provenance. The Translation V2T median rank of 56.1 is preserved literally despite being inconsistent with an ordinary median of integer ranks.

| CiCo table / dataset / copied system | T2V R1 / R5 / R10 / MedianR | V2T R1 / R5 / R10 / MedianR |
|---|---|---|
| Table 1 / how2sign / SA-SR | 18.9 / 32.1 / 36.5 / 62.0 | 11.6 / 27.4 / 32.5 / 69.0 |
| Table 1 / how2sign / SA-CM | 24.3 / 40.7 / 46.5 / 16.0 | 17.9 / 40.1 / 46.9 / 14.0 |
| Table 1 / how2sign / SA-COMB | 34.2 / 48.0 / 52.6 / 8.0 | 23.6 / 47.0 / 53.0 / 7.5 |
| Table 2 / phoenix2014t / Translation | 30.2 / 53.1 / 63.4 / 4.5 | 28.8 / 52.0 / 60.8 / 56.1 |
| Table 2 / phoenix2014t / SA-CM | 48.6 / 76.5 / 84.6 / 2.0 | 50.3 / 78.4 / 84.4 / 1.0 |
| Table 2 / phoenix2014t / SA-COMB | 55.8 / 79.6 / 87.2 / 1.0 | 53.1 / 79.4 / 86.1 / 1.0 |

The [official CiCo release](https://github.com/FangyunWei/SLRT/tree/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo) is pinned to commit `38a4f7b00da7a858d59b7fabe5093876a84db8e0`. Its dataset-specific evaluation commands are the recipe. No upstream model or metric code was patched. The wrapper supplies mounted paths, checkpoint locations, output directories and bounded execution. It calls the original `main_task_retrieval.py` under single-process `torch.distributed.run`; training is disabled with `--do_eval`.

Two official archives were acquired through the committed `data.sh`:

| Artifact | Author source | SHA-256 |
|---|---|---|
| Test features | [sign_features.zip](https://drive.google.com/file/d/1Vb-HFZd-rhjN49sB5WwLRpIbyhiC6xTy/view) | `9ba1956cf416df9a31ae3d1a71a3fa9a2d1e2b3724670288b608c8d4eb895c51` |
| Trained checkpoints | [final_models.zip](https://drive.google.com/file/d/1Hpcn5obCcG5JHa3nLvHqX9pfrp7g6wDu/view) | `f02ea0b2a64123b2c8386ccb467c00143454ee6405ed4c07e074dcc712224c6c` |

The exact released features occupy canonical v2 Volume `datasets`, under `cico-features/sign_features/`. Each dataset has paired `_domain_agnostic` and `_domain_aware` directories, with `h2s`, `ph` and `csl` prefixes. Pinned upstream `data_h2/test.pkl`, `data_ph/test.pkl` and `data_csl/test.pkl` define the evaluated membership. Their SHA-256 values, archive manifest and per-run coverage audits are recorded in `reproduction.json`; every selected video was checked for both feature files. No raw video decoding or additional feature extraction is introduced.

How2Sign processing uses its CC BY-NC 4.0 non-commercial research terms. PHOENIX research processing follows the CC BY-NC-SA basis documented in the existing project reports and official dataset release. CSL-Daily uses the existing signed institutional research access grant documented in [the project's gloss-pretrained report](../chen-2026-gloss-pretrained/README.md) at base commit `0ff14a2`. These permissions cover the existing project processing; restricted data and predictions remain in project storage and are not redistributed. No new participants, human evaluation or author contact were involved.

The dataset-specific checkpoints are `H2S_sota.pth`, `ph_sota.pth` and `csl_sota.pth`. The author recipe uses alpha 0.8 for How2Sign/CSL-Daily and **0.9 for PHOENIX**, despite the paper's general implementation paragraph stating 0.8. The explicit dataset-specific command is followed. The upstream loader also uses its pinned CLIP initializer, SHA-256 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`, before loading the released CiCo state. No optimizer, epoch count or checkpoint selection was inferred for this evaluation-only attempt.

The GPU environment starts from the study image pinned to `ghcr.io/sign-language-processing/reproduction@sha256:305b6165d306192996358ca312d9a751fa409f43063a76dc7758880a8f905291`. The retained image is `im-5Bm6i5wypKGHVHK96xT4kP`, with Python 3.12, PyTorch `2.11.0a0+eb65b36`, CUDA 13.1, driver 580.95.05 and one NVIDIA A10G. This modernizes the author environment of Python 3.7, PyTorch 1.7.1 and CUDA 11. The wrapper pins its added dependencies, including textblob 0.17.1; each execution retains its resolved `pip-freeze.txt` and `hardware.txt`.

All Modal operations use the `repro-sign` wrapper and environment `main`. Training data storage is mounted read-only for evaluation. The shared v2 `huggingface-cache` Volume is mounted read-write at `/cache/huggingface`, with `HF_HOME` and `HF_HUB_CACHE` set. Outputs and released checkpoints are retained in the separate v2 `cheng-2023-cico-results` Volume. Native console/metric logs, commands, hardware records, dependency freezes, feature/checkpoint audits and diagnostic subset membership have immutable content hashes and external URIs in `reproduction.json`. Nothing in the cache alone establishes provenance.

Two inexpensive failures preceded the successful PHOENIX preflight. The first wrapper matched both actual feature directories and `__MACOSX` metadata; selecting the exact archive root fixed the path ambiguity. The second encountered textaugment's import of `textblob.translate`, removed in textblob 0.18; pinning 0.17.1 fixed the import. Neither fix changed the upstream model or metric implementation. The third preflight evaluated 16 real examples with the released weights: its perfect subset retrieval scores are diagnostics, not paper results. A 256-query How2Sign preflight then exercised the full batch size and informed the larger retrieval estimate. The CSL preflight likewise used real released features and weights.

| Retained run | State | Runtime (s) | Process GPU-hours | Sampled peak memory (MiB) |
|---|---|---:|---:|---:|
| [data-1](https://modal.com/apps/repro-sign/main/ap-RyXszdj6zO8XQEAwMsM7c8) | succeeded | 194.95 | — | Not recorded |
| [preflight-ph-1](https://modal.com/apps/repro-sign/main/ap-RUNvxALkbGZ5aruz14cUwQ) | failed | 303.62 (wrapper) | Unknown | Not recorded |
| [preflight-ph-2](https://modal.com/apps/repro-sign/main/ap-eGmNFk8iuLzFeIi3SMndO1) | failed | 11.15 | 0.00310 | 4 |
| [preflight-ph-3](https://modal.com/apps/repro-sign/main/ap-N55WM2cD4DoJgFgPBiV2Pn) | succeeded | 36.77 | 0.01022 | 706 |
| [full-ph-1](https://modal.com/apps/repro-sign/main/ap-RB297o8duhnTsI04MiD7of) | succeeded | 191.34 | 0.05315 | 1437 |
| [preflight-h2s-1](https://modal.com/apps/repro-sign/main/ap-OyKwCFnXPPh60M4AEslvv0) | succeeded | 242.28 | 0.06730 | 1955 |
| [preflight-csl-1](https://modal.com/apps/repro-sign/main/ap-ePuf5LNGzw6jQrahR2a3no) | succeeded | 67.03 | 0.01862 | 733 |
| [full-csl-1](https://modal.com/apps/repro-sign/main/ap-JQOysWax5EqHCEEe7z66PV) | succeeded | 310.35 | 0.08621 | 6733 |
| [full-h2s-1](https://modal.com/apps/repro-sign/main/ap-IGtSMOKofhTs4MC4SAHk5Q) | succeeded | 701.37 | 0.19482 | 4315 |

GPU-hours above are measured process wall time on one GPU, excluding container provisioning. The first failed preflight has only wrapper timestamps, which include image construction and startup; its GPU duration is unknown and is not inferred from that interval. Actual billed cost is unavailable. The PHOENIX full evaluation had a predeclared one-hour / one-GPU-hour / CHF5 ceiling after a conservative 20-minute estimate. CSL-Daily used the same ceiling after a 30-minute estimate. The How2Sign preflight took 242.28 seconds with 1,955 MiB sampled memory; its full run was estimated at 50 minutes, with a two-hour / two-GPU-hour / CHF5 ceiling and a 6,900-second subprocess limit. The prospective attempt and stopping policies remain unchanged in the ledger; a retry requires a specific diagnosis and remaining allowance.

Run `./setup.sh` once per clone and authenticate the `repro-sign` workspace, then use the repository root. The data step verifies pinned archive checksums and reuses complete extraction. Every evaluation requires a fresh output `--run-id`; the wrapper rejects an existing run directory to protect retained evidence. The following example IDs must be changed if already used:

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/cheng-2023-cico/modal_app.py --stage data --run-id data-repeat
.agents/skills/reproduce-paper/scripts/check_modal_dataset.sh cico-features manifest.json
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/cheng-2023-cico/modal_app.py --stage eval --dataset ph --subset 16 --run-id preflight-ph-repeat
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/cheng-2023-cico/modal_app.py --stage eval --dataset ph --subset 0 --run-id full-ph-repeat
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/cheng-2023-cico/modal_app.py --stage eval --dataset csl --subset 0 --run-id full-csl-repeat
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/cheng-2023-cico/modal_app.py --stage eval --dataset h2s --subset 0 --run-id full-h2s-repeat --max-seconds 6900
python3 .agents/skills/reproduce-paper/scripts/validate_reproduction.py papers/cheng-2023-cico
```

The evaluation entry point handles setup, checkpoint loading and scoring; it does not provide a verified training reproduction. Any training attempt requires its own data/recipe investigation and prospective run records.

The direct batch assignment preserves the redacted tracker export and its original SHA-256 `61a45da0c6d25859bc8abba10df4096806e371ca2b326b5d3881043781fcf780`. Its current schema has final paper status, a separate database ID and no legacy confirmation field; none was invented. `codex-orchestrator`—GPT-6 using Codex—performed source investigation, implementation and execution. `codex-reviewer`—also GPT-6 using Codex—independently reviewed the wrapper against upstream and edited this report on 2026-10-01; it did not execute the reported runs. These identities are explicitly attested by the active session instructions. Exact model IDs and application versions were not exposed, and execution attribution is preserved unchanged.

Training continuation is separately recorded in `reproduction.json.training_continuation`. GPT-6 using Codex (`codex-training-orchestrator`) is investigating raw-data preparation, sign-encoder adaptation, and native CLCL training; exact model and harness versions are not exposed. Earlier 24 checkpoint-evaluation results and their executor attribution remain unchanged. The new trained-model results will be evaluated against the published baselines without score-driven tuning.

## Independent training continuation (2026-10-03)

A GPT-6 agent using Codex is executing this continuation separately from the earlier checkpoint evaluation (`cico-training-agent`; exact model ID and harness version are unavailable). All 24 original target objects remain unchanged. Another 24 `trained-` targets track independent training. Published baseline scores will be compared; those systems are not rerun.

CPU acquisition verified the official BSL5K initialization (`6430592464a357dfdaa7f31973cb684663237655fdf23f3999608d162167fc6f`) and released How2Sign domain-aware encoder (`99e101d696ff63131b5d44fa6e465201216604ba5d8cc773f3cefa4a96ebd518`). Their classifier heads differ (5,383 versus 1,432 classes), but both expose 1,024-dimensional pooled embeddings. The released adapted encoder does not establish independent encoder training.

PHOENIX is the first full-chain candidate. A read-only audit verified SHA-256 and frame metadata for all 8,257 videos: 7,096 train, 519 dev, and 642 test. The manifest is `modal://cheng-2023-cico-results/phx-raw-manifest-v1/manifest.json`, SHA-256 `4974f59634d771679d32c7b7031115506286e9dd1247ba42d41900323cb8d53a`. The train split contains 720,914 native 16-frame, stride-one windows. Author `dev.pkl` combines train and dev identities, so it is not a held-out development split. CSL has all 18,401 training and 1,176 test filenames requested by the author labels. How2Sign raw videos are absent.

Native I3D inference and three diagnostic SGD steps passed on real PHOENIX videos; fresh model/optimizer restoration was exact. The resumed diagnostic step used the previous batch's pseudo labels, so it proves finite mechanics only. The actual threshold-0.6/NMS-24 pseudo-label microcase produced two clips from 126 windows. Closed-rank replay and recovery after a directory rename but before its completion receipt regenerated identical clip hashes. These checks are diagnostic evidence, not trained paper results.

Training-only patches replace video decoding with `simple-video-utils`, preserve RGB values and native end-exclusive clip encoding, use the real 260×210 source dimensions, and convert native `range` objects to lists for the existing segment-concatenation operation. Decoded bytes exactly matched native OpenCV on initial, interior and tail windows; short-video padding also passed. The original checkpoint experiment remains unchanged.

The continuation ceiling is 24 GPU-hours / CHF 90, including up to two GPU-hours / CHF 10 for diagnostics and four CPU-hours / CHF 8 for acquisition/audits. The proposed training-pseudo stage has a four-hour absolute deadline, at most two execution segments, and a CHF 16 reservation. Full adaptation and CLCL training require separate native preflights and measured forecasts. No trained target value is claimed yet.

The first CPU wrapper failed before native execution because a sibling module was missing remotely; a standalone wrapper fixed it. A GPU probe then stopped at the missing OpenCV import; pinned OpenCV 4.11.0.86 passed CPU checks before the corrected GPU probe. Every retained attempt, original policy, source hash, native exit and Modal identifier is recorded in `reproduction.json`.

Repeat bounded preparation and diagnostics through the project wrapper:

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/cheng-2023-cico/scripts/training_modal.py::prepare --run-id training-input-preparation-v2
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/cheng-2023-cico/scripts/training_modal.py::decoder_audit --run-id decoder-audit-v2
```

Existing run IDs are immutable and cannot be reused for a fresh execution. New inputs and outputs use `cheng-2023-cico-results`; datasets remain read-only. Native CLCL uses a true global contrastive batch of 512 and evaluates test retrieval every epoch, selecting by test R@1. This selection protocol is explicitly disclosed; ordinary gradient accumulation is not an equivalent negative pool.
