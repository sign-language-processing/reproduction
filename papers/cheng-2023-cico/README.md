# CiCo: Domain-Aware Sign Language Retrieval via Cross-Lingual Contrastive Learning

**Current work:** native 15-epoch PHOENIX sign-encoder training completed, and its adapted-feature probe/recovery checks passed. Full domain-aware extraction completed; both7738-file feature archives are now closed and verified. Actual paired-feature CLCL preflight and fresh-process recovery passed. The full training proposal is under internal review.

**Completed evidence:** released-checkpoint evaluation only. The prior run did not train the visual encoders or CLCL model. The author archive contains test features, without train/dev features.

**Pipeline status:** partial

The 24 released-checkpoint targets are preserved. Eight PHOENIX independent-training targets remain pending; the 16 How2Sign/CSL training targets are terminal not-produced under the original compute allowance.

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

The continuation ceiling is 24 GPU-hours / CHF 90, including up to two GPU-hours / CHF 10 for diagnostics and four CPU-hours / CHF 8 for acquisition/audits. The completed [training-pseudo stage](https://modal.com/apps/repro-sign/main/ap-ptrUKZx3D3xVTdREcJSgrj) started at 2026-10-03 19:34:27 UTC. It completed at 21:31:54 UTC in 1.9573 GPU-hours, preserving its original 23:34:27 UTC deadline. Domain-agnostic extraction completed at 23:04:37 UTC in 2.3164 GPU-hours. Both used one execution segment. The reviewed 15-epoch adaptation started on October 4 at 05:44:27 UTC with an immutable 10:14:24 UTC deadline and a 4.5 GPU-hour / CHF18 ceiling. CLCL still requires actual paired-feature preflight and a measured full-run forecast. No trained retrieval target value is claimed yet.

The first CPU wrapper failed before native execution because a sibling module was missing remotely; a standalone wrapper fixed it. A GPU probe then stopped at the missing OpenCV import; pinned OpenCV 4.11.0.86 passed CPU checks before the corrected GPU probe. Every retained attempt, original policy, source hash, native exit and Modal identifier is recorded in `reproduction.json`.

Repeat bounded preparation and diagnostics through the project wrapper:

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/cheng-2023-cico/scripts/training_modal.py::prepare --run-id training-input-preparation-v2
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/cheng-2023-cico/scripts/training_modal.py::decoder_audit --run-id decoder-audit-v2
```

Existing run IDs are immutable and cannot be reused for a fresh execution. New inputs and outputs use `cheng-2023-cico-results`; datasets remain read-only. Native CLCL uses a true global contrastive batch of 512 and evaluates test retrieval every epoch, selecting by test R@1. This selection protocol is explicitly disclosed; ordinary gradient accumulation is not an equivalent negative pool.


The native I3D trainer microcase runs the actual author train/validation loops on two generated pseudo clips, repeated only to exercise batch size four. Its validation rows reuse these same diagnostic clips and are never reported as held-out measurements. Independently repeated runs exhibit small floating-point state differences before any resume despite identical RNG states. Disabling cuDNN benchmarking did not resolve this, so that hypothesis was discarded. Recovery is instead checked by exact model/optimizer/all-RNG restoration in a fresh native process, matching next-batch IDs and labels, and finite continuation. The corrected native recovery check passed: model, optimizer, all RNG state, restored epoch, and next-batch indices/labels matched. A proof-only instrumentation error that removed the restored epoch assignment was diagnosed and fixed before any full adaptation. This establishes restored state and data order, not a bitwise-identical floating-point trajectory.

A representative native run then trained on 32 distinct generated training clips for two epochs with zero validation rows. Empty validation executes without changing selection or scheduling. The second epoch averaged 0.36148 seconds per batch of four, with 13.87 GB peak allocated GPU memory. Full adaptation used all closed training pseudo clips, native SGD (learning rate 0.01, momentum 0.9, weight decay zero), seed zero and the final epoch-15 checkpoint. Native milestones 20 and 40 do not occur. Zero-row validation logs are not scientific accuracy measurements. The final clip count, staging forecast, source review and immutable run policy passed before launch.

The native feature microcase produced finite float32 arrays of shapes (38,1024), (75,1024) and (13,1024); all file hashes matched closed receipts. Completed-rank replay and interrupted-rank regeneration passed. Its native exit was zero; a client transport error occurred after the completion receipt and did not justify repeating GPU work. Closed diagnostic GPU subprocess time for this continuation totals approximately 0.114 hours as of this report update, excluding provisioning.


Domain-agnostic PHOENIX feature extraction completed under a separate immutable four-hour / CHF16 ceiling (app `ap-4Lpt8BmHT5kvqaDKkdzfOI`, call `fc-01M41R78RCHNKWFWBBXJBRVFA9`, deadline 2026-10-04 00:45:37 UTC). It preserves the native 16-frame stride-one windows and 1,024 float32 channels for all 7,096 train and 642 test videos. This is preparation, not a trained target result.

The native CLCL diagnostic completed on 512 real training feature/query pairs with true global batch 512, accumulation one, the author mixed parameter precision and BertAdam. Fresh resume and uninterrupted execution produced identical model tensors, optimizer state, next-batch tensors and loss/metric histories after two updates; 299 parameter tensors changed. All 300 compatible CLIP tensors loaded exactly. The image convolution and positional embeddings have the two expected image-to-feature shape mismatches, so their native seeded initialization is retained. Peak CUDA memory was 45.18 GB allocated / 47.75 GB reserved, with 1.7423 seconds for the warm batch. The complete diagnostic took 301.50 seconds, mostly remote feature staging. Its duplicated domain-agnostic streams and 16 training-query evaluation are mechanics only; final paired-feature/full-loader preflight remains required. The full 7,096-row loader has 13 batches per epoch and 2,600 optimizer updates over 200 epochs, unlike the 200-update diagnostic horizon.

CPU collection preserves every native feature-pickle byte in a checksum-verified archive for container-local staging. This changes I/O location only. Every native best-test checkpoint, the decadal snapshots, optimizer state, raw metric history and similarity matrices remain retained; no trained paper result is inferred from preprocessing or diagnostics.


The full PHOENIX teacher stage completed natively with exit zero in 7,046.30 seconds (1.9573 GPU-hours), processing all 720,914 training windows. CPU collection verified all 256 rank receipts and 8,608 unique generated clips (319,278,600 bytes); the closed training manifest SHA-256 is `d0509e382d73920ccc6768a26cf0a3c531f714b957f53c5c288f9b2c9aa9539a`. The actual count implies 32,280 SGD updates and 3.241 hours at the representative rate, exceeding the unlaunched provisional three-hour proposal before staging. The revised 4.5-hour / CHF18 plan passed internal review and launched on October 4 at 05:44 UTC. Its first execution established a new immutable deadline at 10:14 UTC; no prior running deadline was extended.

CSL investigation found a material input distinction. The shared raw videos contain 2,721,567 training and 185,718 test sliding windows, while native-label frame counts imply only 1,909,322 and 132,608. Four deterministic test examples have released feature lengths matching the shorter labels minus 15; both OpenCV and `simple-video-utils` confirm the longer shared raw videos. The canonical dataset documentation mentions a separate frame archive. We do not invent temporal crop offsets or claim these representations are interchangeable. All 19,577 requested raw-video metadata records were counted via direct paths in 253.56 CPU seconds after a directory-traversal attempt hit its 1,800-second limit. The canonical annotation also omits one author training identity and differs on three other frame counts; these are retained explicitly. Three extraction passes over the current raw representation alone forecast about 23.2 GPU-hours, before adaptation and CLCL, exceeding the remaining continuation allowance. No CSL training, bulk archive acquisition or author contact has occurred.


The terminal preparation/diagnostic evidence is closed by `modal://repro-sign/main/cheng-2023-cico-results/training-evidence-closure-v1/manifest.json`, SHA-256 `e2dfc7f4feeb69c91311761d25cc141a5fa2840e2796f7897248f73c9c65a277` (684 files, 8,081,531,898 bytes). This CPU collection does not rerun models. Two failures before native launch have no output directories and retain their known CLI/build exits separately from null native exits. Pseudo clip bytes remain covered by the independently verified closed clip manifest and rank receipts.

A further read-only check confirms all 7,096 PHOENIX training and 642 test frame counts exactly match the author labels. For CSL, the pinned README explicitly requires frame images passed through `gather_frames.py`, which encodes all supplied images at 25 fps and defines no trimming offsets. The separately distributed `csl-daily-frames-512x512` archive parts 00–09 remain the precise acquisition/version lead; their absence from the present Modal inventory does not establish that they are inherently unobtainable.

A separate GPT-6/Codex recovery executor continues on 2026-10-04, collecting earlier evidence and launching subsequent stages. Earlier run attribution is preserved; exact model ID and harness version are unavailable. The full 8,608-clip adaptation has an approved prospective 4.5 GPU-hour / CHF18 ceiling within the unchanged 24 GPU-hour / CHF90 continuation budget.

The later CPU archive audit completed in 685.48 seconds on October 4: all 7,738 native float32 feature files (7,096 train / 642 test, 775,911 windows) passed per-file hashes, finite-value/shape checks and exact membership. The byte-preserving ZIP is 3,181,370,649 bytes with SHA-256 `c60bdd14c7082015ac3e0d6b96ee03962654390915362d61b391dfdf9563db96`; its closed manifest is `8a51ffe369071939d249d0b92f4f081d71e878d6db3f4597b6a21613046a3da3`. This archive audit is attributed to the later recovery executor and does not change earlier model-execution attribution.

The independent I3D adaptation completed all 15 epochs / 32,280 SGD updates on 8,608 clips at 2026-10-04 07:53:37 UTC, native exit zero, one execution segment, 7,748.60 seconds (2.15239 GPU-hours). Selection uses final epoch 15 with no validation or test selection. The final checkpoint SHA-256 is `c89b5384ec86515cde41b7fbf3172bdeab8f170acb1a042677e8c2b9d4ca2c35`. A separate CPU audit closed all 66 files / 572,255,987 bytes under manifest `be5c9e8b80ee9aac2da67a9e7df4bd0c40c89226e923dda26301d70a8082bcc6`. The actual adapted encoder then passed native extraction on 126 windows, finite 1,024-channel feature checks and exact recovery checks. The full aware stage started at 07:59:25 UTC with an immutable 11:59:24 UTC deadline, four GPU-hours / CHF16 and at most two segments. These encoder/feature results do not yet establish retrieval accuracy.

A later CPU audit counted the exact pinned How2Sign labels: 31,085 training / 2,348 test videos, with 4,988,373 / 376,538 native 16-frame windows. Teacher generation plus two train/test feature passes require 15,718,195 windows. At the measured completed PHOENIX full-stage throughputs this projects about 45.58 GPU-hours for extraction alone, excluding download, realignment, encoder adaptation and CLCL. This is a forecast, not a measured How2Sign execution; it already exceeds the unchanged 24 GPU-hour continuation ceiling. Raw videos therefore remain unacquired, and existing SPOT-ALIGN features are not substituted.

The How2Sign and CSL independent-training targets are recorded as `compute_budget_blocked` within this bounded attempt. Even the shorter native-label CSL representation projects 17.38 GPU-hours for teacher generation and two feature passes, before adaptation or CLCL; at review, at most 10.37 GPU-hours remained unreserved after completed work and the PHOENIX stages. This does not claim that CSL frame archives are unobtainable: the separate image archive remains an unacquired provenance lead, and the longer shared MP4 corpus is not silently substituted. No immediate human action or budget expansion is requested.

Compute reservation accounting remains CHF90: completed pseudo8, agnostic10, adaptation10, aware16, CLCL12, diagnostics10, CPU8 and storage16. The original run ceilings are unchanged. Measured checkpoint sizes bound a conservative 117.63 GiB component forecast; 150 GiB is reserved for one month at the [published Modal rate](https://modal.com/pricing) of $0.09/GiB/month ($13.50), with no artifact pruning or automatic deletion. Retention beyond that month is outside this cost forecast. An A100-80GB with four CPU cores and 32 GiB requested RAM costs an estimated $2.94/hour; three hours plus120seconds of load/idle allowance per allowed segment estimates $9.12 against the CHF12 CLCL reservation. The CHF1/USD conversion is a conservative planning assumption; actual billed cost is unavailable. Full CLCL remains conditional on actual paired-feature, end-to-end preflight measurements including evaluation, checkpoint writes and volume commits.

Before actual paired CLCL preflight, the driver pins the aware stream to the exact closed adapted-checkpoint SHA `c89b5384ec86515cde41b7fbf3172bdeab8f170acb1a042677e8c2b9d4ca2c35`, in addition to both closed feature manifests and archive hashes. The planned diagnostic runs one full 13-update epoch and all 642 test queries, then a second epoch after fresh-process restoration under the same 1,800-second ceiling. Its conservative epoch forecast includes evaluation, checkpoint/similarity writes, final hashes and volume commits; warm training throughput alone is insufficient.

Domain-aware extraction finished on October4 at11:37:33UTC in13085.9996seconds (3.6350GPU-hours), one segment with native/stage exit0 before its original11:59:24deadline. The declared CPU collector was submitted automatically at11:38; its function started at11:39:05UTC to verify every feature and build the closed archive. This completes encoder training and extraction, not trained retrieval evaluation.

Aware CPU closure completed at11:48:57UTC in592.319seconds: all7,738native feature files and775,911windows passed exact membership, finite shape/dtype, checkpoint/source identity and ZIP readback hashes. Its manifest SHA is `60c86cecf796321310215aa7d465a73b93853d442a5665317155db2d3348b5b3`; archive SHA is `9ac9d90f134346d83ac32f18010b646ab61218848f3505e13320e07169ab47fa`. Paired full-loader CLCL preflight `phx-clcl-paired-preflight-v1` has a prospectively declared combined1,800-second/0.5GPU-hour/CHF2 ceiling and at most two segments, within the unchanged diagnostic budget. It executes one native epoch and then one resumed epoch; full200epoch training still requires independent measured-path review.

Actual paired CLCL preflight completed two full-loader epochs (26 updates) with exact fresh-process model, optimizer and RNG restoration and finite losses6.3010→5.7803. Both epochs evaluated all642 test queries. Peak CUDA allocation was45.178GB; source/runtime receipts and all150artifact hashes are closed. These diagnostic retrieval values are not final target results. The second epoch conservatively took at most48.145seconds including test evaluation and checkpoint/output overhead, exceeding the first three-hour planning condition when projected across200epochs. A revised, still unlaunched four-GPU-hour/CHF14 plan forecasts11,279seconds (rounded11,600), including all three segments’ staging/startup,20extra decadal saves and900seconds margin. Internal review remains pending; the requested200epoch recipe is unchanged. The proposed aware14/CLCL14 reservation shift keeps the totalCHF90 and24GPU-hour ceiling. Terminal CPU collection reserves900seconds within the conservatively accounted4CPU-hour budget.

Independent launch reviewer `cico-independent-launch-reviewer` identifies its session as GPT-6 using Codex; exact model and harness versions are unavailable. It independently checked the paired inputs, native source, closed two-epoch training and recovery receipts, runtime, storage and budget plan. Technical review passed for commit `0ae059a`; it performed no implementation or experiment execution. The revised full-run allocation awaits the orchestrator’s authorization.
