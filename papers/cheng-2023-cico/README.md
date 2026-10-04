# CiCo: Domain-Aware Sign Language Retrieval via Cross-Lingual Contrastive Learning

**Completed:** independent 15-epoch PHOENIX encoder adaptation and 200-epoch CLCL training produced T2V/V2T R@1 **69.31%/68.07%**, above published SA-COMB **55.8%/53.1%**. All six recalls exceed the baseline; both median ranks tie. No immediate human blocker.

**Pipeline status:** partial — **32/48** in-scope values produced (24 preserved released-checkpoint evaluations plus 8 trained PHOENIX values); 16 How2Sign/CSL training values are not produced within the recorded compute budget.

**Numerical agreement:** does_not_agree

**Preference level:** 2

The numerical assessment uses prospectively declared exact one-decimal equality. Comparative advantage and exact numerical agreement are separate conclusions.

The preserved checkpoint evaluation also corroborates the reported How2Sign/PHOENIX advantage over published SPOT-ALIGN values. Baselines were not rerun. PHOENIX training is now assessed for one seed; How2Sign/CSL training, component ablations, causal explanations and seed variability remain unassessed. Native test-based checkpoint selection is disclosed below.

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

The [official CiCo release](https://github.com/FangyunWei/SLRT/tree/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo) is pinned to commit `38a4f7b00da7a858d59b7fabe5093876a84db8e0`. Its dataset-specific evaluation commands are the recipe. For the preserved released-checkpoint experiment, no upstream model or metric code was patched. The wrapper supplies mounted paths, checkpoint locations, output directories and bounded execution. It calls the original `main_task_retrieval.py` under single-process `torch.distributed.run`; training is disabled with `--do_eval`.

Two official archives were acquired through the committed `data.sh`:

| Artifact | Author source | SHA-256 |
|---|---|---|
| Test features | [sign_features.zip](https://drive.google.com/file/d/1Vb-HFZd-rhjN49sB5WwLRpIbyhiC6xTy/view) | `9ba1956cf416df9a31ae3d1a71a3fa9a2d1e2b3724670288b608c8d4eb895c51` |
| Trained checkpoints | [final_models.zip](https://drive.google.com/file/d/1Hpcn5obCcG5JHa3nLvHqX9pfrp7g6wDu/view) | `f02ea0b2a64123b2c8386ccb467c00143454ee6405ed4c07e074dcc712224c6c` |

The exact released features occupy canonical v2 Volume `datasets`, under `cico-features/sign_features/`. Each dataset has paired `_domain_agnostic` and `_domain_aware` directories, with `h2s`, `ph` and `csl` prefixes. Pinned upstream `data_h2/test.pkl`, `data_ph/test.pkl` and `data_csl/test.pkl` define the evaluated membership. Their SHA-256 values, archive manifest and per-run coverage audits are recorded in `reproduction.json`; every selected video was checked for both feature files. That preserved evaluation path introduces no raw decoding or feature extraction; the separate training continuation below does both.

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

The evaluation entry point above handles the preserved checkpoint experiment. The separately verified training entry points and prospective records are documented below.

The direct batch assignment preserves the redacted tracker export and its original SHA-256 `61a45da0c6d25859bc8abba10df4096806e371ca2b326b5d3881043781fcf780`. Its current schema has final paper status, a separate database ID and no legacy confirmation field; none was invented. `codex-orchestrator`—GPT-6 using Codex—performed source investigation, implementation and execution. `codex-reviewer`—also GPT-6 using Codex—independently reviewed the wrapper against upstream and edited this report on 2026-10-01; it did not execute the reported runs. These identities are explicitly attested by the active session instructions. Exact model IDs and application versions were not exposed, and execution attribution is preserved unchanged.

## Independent PHOENIX training (completed October 4, 2026)

This continuation trained the visual encoder and retrieval model independently of the released retrieval checkpoints. Its eight PHOENIX targets are separate from the 24 preserved evaluation targets above. The 16 requested How2Sign/CSL training targets are terminal `not_produced`, reason `compute_budget_blocked`, with measured forecasts below. All 48 in-scope targets have terminal results; 32 produced comparable values.

| PHOENIX direction / metric | Paper CiCo | Independently trained | Difference | Published SA-COMB |
|---|---:|---:|---:|---:|
| T2V R@1 | 69.5 | 69.3146 | −0.1854 pp | 55.8 |
| T2V R@5 | 86.6 | 87.2274 | +0.6274 pp | 79.6 |
| T2V R@10 | 92.1 | 91.4330 | −0.6670 pp | 87.2 |
| T2V median rank | 1 | 1 | 0 | 1 |
| V2T R@1 | 70.2 | 68.0685 | −2.1315 pp | 53.1 |
| V2T R@5 | 88.0 | 87.5389 | −0.4611 pp | 79.4 |
| V2T R@10 | 92.8 | 91.2773 | −1.5227 pp | 86.1 |
| V2T median rank | 1 | 1 | 0 | 1 |

All six independently trained recalls exceed the published SA-COMB values; median ranks tie. R@1 gains are +13.5146/+14.9685 percentage points. This corroborates the reported comparative advantage on PHOENIX using the published comparison numbers. Those baseline systems were not rerun. The six recall values do not exactly match at the prospectively declared one-decimal precision; both median ranks match. No tolerance was introduced after seeing results. Component causality, ablations, other-dataset training and seed variability remain unassessed.

**Selection disclosure:** native CLCL evaluates all 642 test queries/videos after every epoch and chooses maximum test T2V R@1, replacing the selected checkpoint on ties. The selected zero-based epoch is 106 (the 107th epoch); all 200 epochs and 2,600 updates nevertheless completed. This is test-based selection, not a validation-selected estimate.

### Data, recipe and recovery

A read-only audit verified all 8,257 canonical PHOENIX video hashes and frame counts: 7,096 train, 519 dev and 642 test. All selected train/test counts match the pinned author labels. The raw manifest SHA is `4974f59634d771679d32c7b7031115506286e9dd1247ba42d41900323cb8d53a`. The BSL5K teacher/init weights have SHA `6430592464a357dfdaa7f31973cb684663237655fdf23f3999608d162167fc6f`; the CLIP initializer remains pinned as above. The released How2Sign adapted encoder was inspected during acquisition but never substituted for independent PHOENIX adaptation.

Native threshold-0.6/NMS-24 teacher generation processed 720,914 training windows and produced 8,608 unique pseudo clips (319,278,600 bytes). Its closed manifest is `d0509e382d73920ccc6768a26cf0a3c531f714b957f53c5c288f9b2c9aa9539a`. Native I3D adaptation used all clips, 15 epochs/32,280 updates, batch4, SGD learning rate0.01/momentum0.9/weight decay0 and seed0. Author `dev.pkl` includes training identities, so it is not held-out validation. The complete training split with empty validation and final epoch15 checkpoint preserves the native scheduler/selection; milestones20/40 do not occur. Empty-validation accuracy logs are not scientific measurements. Final adapted checkpoint SHA: `c89b5384ec86515cde41b7fbf3172bdeab8f170acb1a042677e8c2b9d4ca2c35`.

Native 16-frame, stride-one feature extraction preserves 1,024 float32 channels. Each stream contains exactly 7,738 files (7,096 train/642 test), 775,911 windows. All aware feature hashes differ from the corresponding agnostic ones and pin the exact independently adapted checkpoint. Closed manifests are agnostic `8a51ffe369071939d249d0b92f4f081d71e878d6db3f4597b6a21613046a3da3` and aware `60c86cecf796321310215aa7d465a73b93853d442a5665317155db2d3348b5b3`. Byte-preserving ZIP_STORED archives, each3,181,370,649bytes, have hashes `c60bdd14c7082015ac3e0d6b96ee03962654390915362d61b391dfdf9563db96` and `9ac9d90f134346d83ac32f18010b646ab61218848f3505e13320e07169ab47fa`. Archive staging changes I/O location only; every member, shape and hash was verified.

CLCL uses the pinned author trainer/optimizer, true contrastive batch512, accumulation1, seed42, sampler seed0, alpha0.9 and 13 batches per epoch with native drop-last. The configured horizon is 200 epochs/2,600 updates. All 300 compatible CLIP tensors load exactly; the image convolution and positional embeddings have the two expected image-to-feature shape mismatches and retain native seeded initialization. No released retrieval checkpoint initializes this run. The continuation uses the same pinned study base image on one A100-80GB, four requested CPU cores and32GiB RAM, PyTorch `2.11.0a0+eb65b36914.nv26.02`; closed hardware, dependency freeze, command and source/patch hashes identify the actual execution.

Training patches preserve native computation while adapting paths, video decoding and recovery. `simple-video-utils` decoding matched native OpenCV RGB bytes on initial/interior/tail windows and short-video padding; generated clips preserve native end-exclusive slicing and real260×210 dimensions. Converting native `range` objects to lists fixes segment concatenation. Every retained patch and source hash is recorded in the run ledger; no model or metric was tuned toward the paper score.

Representative I3D preflight trained32 real clips with empty validation and checked exact fresh-process model/optimizer/RNG/start-epoch restoration and matching next-batch labels. Small floating-point drift existed between independent native executions before resume; disabling cuDNN benchmarking did not fix it and was discarded. This proves restored state/data order, not bitwise identity of independently computed trajectories. Completed-rank replay and interrupted-rank regeneration passed for pseudo clips and feature extraction. The actual adapted checkpoint also passed a126-window feature probe with finite38/75/13×1024 arrays.

Actual paired CLCL preflight ran two full-loader epochs/26 updates and full642-query evaluation, with exact fresh-process model/optimizer/RNG restoration. Peak CUDA allocation was45.178GB (47.857GB reserved); losses were finite6.3010→5.7803. Its150-file closure and native CPU metric replay passed. The earlier duplicated-stream/512-example diagnostic remains mechanics evidence only. Native tied V2T sorting and PyTorch lower-median handling are preserved.

### Terminal runs and evidence

| Stage / run ID | Modal app | Runtime seconds | GPU-hours |
|---|---|---:|---:|
| Teacher `phx-pseudo-full-v1` | `ap-ptrUKZx3D3xVTdREcJSgrj` | 7,046.30 | 1.9573 |
| Agnostic `phx-agnostic-features-full-v1` | `ap-4Lpt8BmHT5kvqaDKkdzfOI` | 8,338.91 | 2.3164 |
| Adaptation `phx-i3d-adaptation-full-v1` | `ap-CXTLdOIOuUQT49IWJ9Z2WL` | 7,748.60 | 2.1524 |
| Aware `phx-aware-features-full-v1` | `ap-9ft4aGcdrhpGchpd98FZ0S` | 13,086.00 | 3.6350 |
| CLCL `phx-clcl-full-seed42-v1` | `ap-dZxnrcMB1x5YIYBEGFqamR` | 6,170.04 | 1.7139 |
| Full artifact closure `phx-clcl-full-closure-v1` | `ap-KVtFe1yloq2cyNrrXPHqv8` | 31.40 | 0 |
| Native metric replay `phx-clcl-full-native-metric-audit-v1` | `ap-1wnScAFWSXfL3uRviMmhHv` | 7.68 | 0 |

Every full GPU stage completed in one segment, native exit0, before its original immutable deadline. CLCL call `fc-01M43FCG5PX4SY599MS0DSW71C` ran October4 12:50:29–14:33:19UTC, before16:50:28UTC. Its selected checkpoint SHA is `40754b2a636eb012f53c4f3fac024bf8212657f962e0c12422ca36dd28711a4a`; final recoverable state is `d5e739466ca589a7598f155917d83a5fc097327af5e4288abef872d5b9b45f63`.

The CPU collector hashed all 405 unique full-run files,28,792,764,695bytes, under `modal://repro-sign/main/cheng-2023-cico-results/phx-clcl-full-closure-v1/manifest.json`, SHA `c0e256af1008491fb33f09a10598ed2e67c588e6929f7ebac957d79d99f92899`. It retains all selected/tied checkpoints, decadal snapshots, optimizer/RNG states, histories, similarity arrays, native code/patches, commands and environment receipts. Nothing was pruned. Selected raw metrics SHA is `0ee7716d12da0570f089b662811f89bdedbb14e862b74cd6f9d1af0b2bfa0d31`; similarities SHA is `70fe0afd0283185df9de2bf8317a0eb0aeda77e9f2a28d2008586e0cb8963fca`. The final CPU audit replayed all eight exact native values and verified200 finite loss/test histories,2,600 updates, final epoch199 and latest maximum selection106. Its pinned metric source SHA is `103e93090de14f55d1db61e40ecf1fbc814670975bd9b9ab1fe341d0c95f0a6f`.

Independent review rehashed downloaded small artifacts and the selected similarity arrays, replayed T2V locally, and verified native Torch V2T tie handling against the CPU report. The remote collector hashed full checkpoints; the reviewer did not separately download every checkpoint. The earlier684-file preparation closure and66-file adaptation closure remain in the ledger, with all retained failures and original execution attribution.

The first CPU preparation failed before native execution because remote sibling-module hydration was missing; a standalone wrapper fixed it. A GPU probe exposed a missing OpenCV import; pinned4.11.0.86 passed CPU checks before retry. A recovery-proof instrumentation error removed the restored epoch assignment and was corrected before full adaptation. A feature client transport error occurred after a native success receipt, so GPU work was not repeated. A CSL directory traversal hit its1,800-second cap; direct-path counting completed in253.56seconds. These failures, rejected hypotheses, native/client exits and retry limits remain recorded.

### Bounded omissions and accounting

How2Sign author labels contain31,085 train/2,348 test videos,4,988,373/376,538 native windows. Teacher plus two feature passes require15,718,195 windows, forecasting45.58GPU-hours at measured completed PHOENIX rates, before raw download/realignment, adaptation and CLCL. This exceeds the whole24GPU-hour allowance. Raw videos remain unacquired; SPOT-ALIGN features were not substituted.

CSL author code consumes a separate frame-image representation. Existing raw MP4s imply2,721,567 train/185,718 test windows; author labels imply1,909,322/132,608. Four deterministic samples match released feature lengths to the shorter labels, while two decoders confirm the longer videos. One canonical annotation identity is absent and three frame counts differ. The `csl-daily-frames-512x512` parts00–09 remain an unacquired provenance lead, not an inherently unavailable dataset. Even the shorter representation forecasts17.38GPU-hours for teacher and two feature passes alone, before adaptation/CLCL, exceeding remaining capacity reserved for PHOENIX. No arbitrary temporal trimming, archive substitution, CSL training or author contact occurred.

The whole continuation retained its24GPU-hour/CHF90 ceiling, including diagnostics2GPU-hours/CHF10 and CPU4hours/CHF8. Known GPU function/process time totals43,414.84seconds (12.0597hours); charging another900seconds conservatively for an earlier pre-native failure remains below24hours. Known CPU time totals2,427.52seconds; another6,000seconds conservatively charged for unknown bounded CPU attempts gives2.3410hours, below4hours. These timings exclude provisioning and are not invoices.

Reservations reconcile to CHF90: completed teacher8, agnostic10, adaptation10, aware14, CLCL14, diagnostics10, CPU8 and storage16. Original executed stop policies were never extended. Measured full-loader preflight invalidated the unlaunched3-hour CLCL proposal; a reviewed new4-hour/CHF14 plan forecast11,600seconds including all allowed segment startup, full-test evaluation, native checkpoint writes/commits and margin. It was authorized before launch and finished in6,170seconds. The actual32,280-update adaptation forecast similarly replaced an unlaunched3-hour proposal with4.5hours before execution.

A conservative117.63GiB component forecast fits the150GiB one-month storage reservation, at the recorded [Modal rate](https://modal.com/pricing) $0.09/GiB/month ($13.50). No artifacts are automatically deleted; retention beyond one month is outside this forecast. Requested A100/CPU/RAM resources estimate$2.94/hour; four hours plus120seconds load/idle per allowed segment estimated$12.07 against CHF14. CHF1/USD is a conservative planning assumption, not an exchange-rate claim. Actual billed costs are unavailable. No further compute or immediate human judgment is required to close this bounded report.

### Repeat commands and attribution

`reproduction.json.runs[].command` preserves every exact preparation, diagnostic, full execution and CPU audit command. Existing run IDs/output directories are immutable; a new independent attempt requires fresh IDs, prospective ceilings, canonical data checks and new closed input manifests. Preparation entry points use `training_modal.py`; the native teacher/feature/adaptation wrappers and CLCL wrapper pin exact sources and reject mismatched inputs. Repeat the documented retained commands in dependency order: canonical raw audit → native teacher/closed pseudo manifest → adaptation/final checkpoint → agnostic and adapted feature extraction/closed archives → actual paired preflight → CLCL → closed evidence and CPU metric replay. The final execution/audit commands are:

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/cheng-2023-cico/scripts/clcl_full_modal.py::run --run-id phx-clcl-full-seed42-v1 --agnostic-manifest /outputs/phx-agnostic-collection-v1/manifest.json --agnostic-sha 8a51ffe369071939d249d0b92f4f081d71e878d6db3f4597b6a21613046a3da3 --aware-manifest /outputs/phx-aware-collection-v1/manifest.json --aware-sha 60c86cecf796321310215aa7d465a73b93853d442a5665317155db2d3348b5b3
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/cheng-2023-cico/scripts/training_modal.py::close_evidence --run-id phx-clcl-full-closure-v1 --source-runs phx-clcl-full-seed42-v1
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/cheng-2023-cico/scripts/training_modal.py::audit_clcl --run-id phx-clcl-full-native-metric-audit-v1 --source-run phx-clcl-full-seed42-v1 --closure-manifest /outputs/phx-clcl-full-closure-v1/manifest.json --closure-sha c0e256af1008491fb33f09a10598ed2e67c588e6929f7ebac957d79d99f92899 --expected-epochs 200
```

All agents identify as GPT-6 using Codex; exact model IDs and harness versions are not exposed, and these unknowns are explicit in the machine-readable evidence. `codex-training-orchestrator` planned/investigated the continuation; `cico-training-agent` implemented and executed earlier preparation, diagnostics, teacher and agnostic stages. `cico-recovery-executor` later collected earlier evidence, executed adaptation/aware/paired preflight/full CLCL and final CPU audits, and maintained this report. `cico-independent-launch-reviewer` independently reviewed native source/guards, lineage, measured preflight/recovery/runtime/storage, final closed evidence and metrics; it did not implement or run experiments. Earlier released-checkpoint executors and all 24 target objects remain unchanged. Reviews and later collection do not retroactively confer execution credit.
