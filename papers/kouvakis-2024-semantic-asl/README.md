# Semantic Communications for Image-Based Sign Language Transmission

**TL;DR:** All three recognition runs finished: 78/81 targets produced; test accuracy is MNIST 88.39%, LEXSET 97.92%, RGB 97.83%.
**Conclusion:** RGB and LEXSET are nearly tied; strict RGB-best ranking and absolute paper scores were not reproduced. No human action is needed; three undefined validation scores remain unproduced.

**Pipeline status:** partial

**Numerical agreement:** does_not_agree

**Preference level:** 3

All published MNIST (90), RGB (80), and LEXSET (60) epoch schedules completed, producing **78 of 81 requested numbers**. The three historical validation partitions remain undefined. There were no failed GPU training attempts or full restarts; CPU acquisition fixes and the stopped remote-file audit are documented below.

| Dataset / split | Published accuracy (%) | Reproduced (%) | Difference (pp) |
|---|---:|---:|---:|
| MNIST / training | 99.61 | 100.00000 | +0.39000 |
| MNIST / validation | 100 | Not produced | — |
| MNIST / testing | 97.12 | 88.38539 | -8.73461 |
| LEXSET / training | 98.39 | 99.48611 | +1.09611 |
| LEXSET / validation | 98.52 | Not produced | — |
| LEXSET / testing | 98.95 | 97.91667 | -1.03333 |
| RGB / training | 99.21 | 100.00000 | +0.79000 |
| RGB / validation | 99.81 | Not produced | — |
| RGB / testing | 99.72 | 97.83333 | -1.88667 |

Table 3(c), class order `A B C D E F G H I K L M N O P Q R S T U V W X Y`. Each official test class has 75 examples. Values are percent; each cell shows **published → reproduced**.

| Class / letter | Precision | Recall | F1 |
|---|---:|---:|---:|
| 0 / A | 98.68 → 96.05 | 100.00 → 97.33 | 99.34 → 96.69 |
| 1 / B | 100.00 → 100.00 | 100.00 → 98.67 | 100.00 → 99.33 |
| 2 / C | 100.00 → 98.67 | 100.00 → 98.67 | 100.00 → 98.67 |
| 3 / D | 100.00 → 100.00 | 100.00 → 96.00 | 100.00 → 97.96 |
| 4 / E | 100.00 → 98.68 | 100.00 → 100.00 | 100.00 → 99.34 |
| 5 / F | 100.00 → 97.40 | 100.00 → 100.00 | 100.00 → 98.68 |
| 6 / G | 100.00 → 100.00 | 100.00 → 97.33 | 100.00 → 98.65 |
| 7 / H | 100.00 → 97.40 | 100.00 → 100.00 | 100.00 → 98.68 |
| 8 / I | 100.00 → 100.00 | 100.00 → 100.00 | 100.00 → 100.00 |
| 9 / K | 100.00 → 96.10 | 98.67 → 98.67 | 99.33 → 97.37 |
| 10 / L | 100.00 → 98.68 | 100.00 → 100.00 | 100.00 → 99.34 |
| 11 / M | 100.00 → 100.00 | 98.67 → 96.00 | 99.33 → 97.96 |
| 12 / N | 100.00 → 97.33 | 97.33 → 97.33 | 98.65 → 97.33 |
| 13 / O | 100.00 → 100.00 | 100.00 → 97.33 | 100.00 → 98.65 |
| 14 / P | 100.00 → 100.00 | 100.00 → 100.00 | 100.00 → 100.00 |
| 15 / Q | 100.00 → 100.00 | 100.00 → 92.00 | 100.00 → 95.83 |
| 16 / R | 100.00 → 96.15 | 100.00 → 100.00 | 100.00 → 98.04 |
| 17 / S | 96.10 → 92.86 | 98.67 → 86.67 | 97.37 → 89.66 |
| 18 / T | 100.00 → 90.36 | 100.00 → 100.00 | 100.00 → 94.94 |
| 19 / U | 100.00 → 93.75 | 100.00 → 100.00 | 100.00 → 96.77 |
| 20 / V | 98.68 → 98.57 | 100.00 → 92.00 | 99.34 → 95.17 |
| 21 / W | 100.00 → 98.68 | 100.00 → 100.00 | 100.00 → 99.34 |
| 22 / X | 100.00 → 98.68 | 100.00 → 100.00 | 100.00 → 99.34 |
| 23 / Y | 100.00 → 100.00 | 100.00 → 100.00 | 100.00 → 100.00 |

| Retained run | App / task | Measured process runtime | GPU-hours |
|---|---|---:|---:|
| mnist-full (90 epochs) | [ap-whLo78NxvmUwKfoYqZ0C2Q](https://modal.com/apps/repro-sign/main/ap-whLo78NxvmUwKfoYqZ0C2Q) / `ta-01M3VGW8W24N09Z1X49352VRMR` | 691.10 seconds | 0.19197 |
| rgb-full (80 epochs) | [ap-Hun5BDvgQbbY4q0gPtR4Nk](https://modal.com/apps/repro-sign/main/ap-Hun5BDvgQbbY4q0gPtR4Nk) / `ta-01M3VGWDHDHKMW3430TA1YGG0R` | 299.26 seconds | 0.08313 |

The final checkpoint test counts are 6339/7172 for MNIST and 1761/1800 for RGB. Full precision values and mechanical differences remain in `reproduction.json`. Process runtime includes input loading, training, checkpoint commits and evaluation, excluding container provisioning. The native environment reports PyTorch `2.11.0a0+eb65b36914.nv26.02`, CUDA 13.1, driver 580.95.05, and `NVIDIA A100-SXM4-80GB`. The immutable training image is `im-jIjEGkfzAuYeo3qOEA7KbC`. Final metrics, predictions and all checkpoint hashes were collected only after terminal completion.

After observing the mismatch, the native split counts, class mappings, published epoch counts, final checkpoint reload and metric formulas were inspected. No concrete implementation defect was found. The unpublished optimizer settings/initialization and inferred MNIST resizing remain possible sources of difference; no score-driven adjustment or restart was performed.

Kouvakis, V., Trevlakis, S. E. and Boulogeorgos, A.-A. A. (2024), IEEE Open Journal of the Communications Society 5:1088–1100. [Paper](https://doi.org/10.1109/OJCOMS.2024.3360191). The direct assignment covers recognition Table 2 and Table 3(c), excluding communication/channel simulations. All 81 requested numbers have individual target records; none are copied literature baselines.

The architecture is reconstructed independently because the authors' related [SemCom-XAI repository](https://github.com/InnoCubePC/SemCom-XAI/tree/ae6d16e959f2b291cb1c4e3f26792fbce70caea3) supplies XAI examples that require an external model path, with no training implementation or weights. The paper, author/project pages, exact-title/author repository searches, main repository history and releases were inspected. The pinned related repository is GPL-3.0; its source is evidence, not vendored code. No upstream patch was necessary.

Figure 3 resolves the architecture precisely: valid convolutions `(128,7), (128,5), (128,2), (128,2), (32,2)`, ReLU, four 2×2 max pools after the first four convolutions, flatten 288, dense 128, and 24 outputs. This gives exactly **616,504 parameters**, including the published per-layer counts. The prose's claim of a pool after every convolution conflicts with the figure and count; the latter agree and determine the implementation. Cross-entropy on logits is equivalent to sparse categorical cross-entropy on softmax probabilities for training, and argmax supplies class predictions.

The full runs use the published 90 MNIST epochs and 80 RGB epochs, Adam, all official training samples, and the final checkpoint. Missing routine details are declared defaults: learning rate 0.001, Adam betas 0.9/0.999, epsilon 1e-8, batch 64, seed 42, float32 and PyTorch default layer initialization. There is no augmentation, test-based checkpoint selection, early stopping, or tuning toward the published score. Missing validation membership remains confined to the validation targets. Online example-weighted training accuracy is the conventional training-loop measure; test metrics use the final saved model on every official test example.

The RGB input is 100×100 RGB divided by 255, as specified by the paper and supported by the related author loader. For RGB, Pillow `Image.Resampling.BILINEAR` performs the resize; the author's related loader uses OpenCV bilinear, so pixel rounding can differ. MNIST uses a separate implementation: `torch.nn.functional.interpolate(mode="bilinear", align_corners=False)` resizes grayscale images from 28×28 to 100×100, then `expand` repeats them over three identical channels. This MNIST input shape is inferred from Figure 3's input and exact parameter count. The seed and these ordinary missing details are assumptions rather than recovered author configuration.

Data identity and permission were checked before training. [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist), Kaggle version 1, is CC0 and has 27,455 training/7,172 test examples in 24 classes. Its native label gap at J is remapped into contiguous classes. The source archive SHA-256 is `fa1b513570d4348c6d6860e04e5854c59ef8eadb8c42a45d36bd1286ce3d489f`. The training and test CSV hashes, exact split counts and manifest hash are recorded in `reproduction.json`.

The authors' [RGB dataset, Zenodo 14635573](https://zenodo.org/records/14635573), is CC BY 4.0. `ASL_SemCom.zip` has MD5 `831ff816c3bb36ffc3b0c9f248cf5033` and SHA-256 `686e818063a1a81bb90fb2cbb319f8c9dc8d368650faf71351d836e508ba0582`. Direct inventory established **10,490 training and 1,800 test images**, precisely the paper's counts, with 75 test images per letter. The landing page's statement of 440 training images per class is inaccurate; it does not override the actual published archive. Both datasets were absent from their intended canonical directories and were acquired directly inside Modal through the committed, idempotent `scripts/data.sh`. Their manifests enumerate every file checksum; the datasets were then mounted read-only for training. Existing public hand images and rendered hand landmarks are processed under their public licenses, with no new human participants or identification task.

[LEXSET](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet), by Lexset, version 3 lists **Data files © Original Authors**. On 2026-10-03, project authorization confirmed attribution-based internal acquisition, Modal storage and processing. This resolves the permission gate without relabeling the public metadata or inferring broader redistribution rights. The pinned archive SHA-256 is `fee9a105ef0785ded6c795031fc0b25252263e782a13836645d06f175cd04373` (7,067,002,276 bytes). The verified 24-class subset excludes J, Z and background, giving 21,600/2,400 native train/test examples. The canonical archive audit completed: all 27,000 sample images and one auxiliary overview passed ZIP CRC, SHA-256 and decoding. The manifest SHA-256 is `de9923d8de3ae1e16dc6f4da7fd91251d26af19689a8fb1ecdb8656438698ca0` at `modal://repro-sign/datasets/synthetic-asl-alphabet/manifest.json`. The 24-letter subset has 900 training/100 test images per letter, with no exact cross-split duplicate. Actual sample dimensions are **513×512**, whereas the paper says 512×512; the declared 100×100 resize is unchanged. No dataset payload is committed or redistributed. Public metadata responses for all three sources are retained under `artifacts/` with hashes. The tracker mentions historical Team S correspondence about RGB; this attempt did not contact authors or rely on unverified correspondence, because the public Zenodo license independently permits the work.

The other open issue is the validation set: section IV-A/Table 1 allocates every listed sample to training or testing, while Table 2 reports validation accuracy without defining its allocation or membership. Creating an arbitrary validation partition would not recover those three validation scores. All published train/test samples remain in their original partitions for the permitted full runs.

The environment starts from the prescribed `ghcr.io/sign-language-processing/reproduction:latest`, resolving to an immutable Modal image recorded with each run. Native dependency freezes, hardware logs, model/optimizer/RNG checkpoints, metrics with epoch history, and predictions/confusion matrices live in v2 Volume `repro-2248c066-results`; every retained artifact has a SHA-256 in `reproduction.json`. Canonical `datasets` and `huggingface-cache` are both v2, with the cache mounted read-write at `/cache/huggingface` and both required HF environment variables set. For the original MNIST/RGB runs an OCI digest was not exposed; immutable Modal image IDs and native package freezes are preserved. The LEXSET continuation pins `ghcr.io/sign-language-processing/reproduction@sha256:305b6165d306192996358ca312d9a751fa409f43063a76dc7758880a8f905291` and also retains the native dependency freeze. All operations use the `repro-sign` wrapper.

The representative preflight trained four real batches from each dataset, saved and reloaded model/optimizer state, took another optimizer step, and evaluated 128 held-out examples. Its 1.5625% MNIST and 0% RGB values are engineering diagnostics, not target results. It measured 194.55/226.24 examples per second and 0.93/1.27 GB peak allocated memory. These conservatively projected 3.53/1.03 GPU-hours for the exact full schedules, approximately $9.35/$2.73 at the inspected [Modal rates](https://modal.com/pricing), before storage. Each detached single-A100-80GB full run had a predeclared six-hour/six-GPU-hour/CHF25 ceiling, one attempt, and an epoch checkpoint. The CHF ceiling uses a conservative 1 USD = 1 CHF budgeting allowance rather than claiming a current exchange rate. Actual billing is unavailable. Full training throughput improved after warm-up; observed durations are reported above.

Run `./setup.sh` once per clone, then repeat from repository root:

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/kouvakis-2024-semantic-asl/scripts/modal_app.py::populate
.agents/skills/reproduce-paper/scripts/check_modal_dataset.sh sign-language-mnist manifest.json
.agents/skills/reproduce-paper/scripts/check_modal_dataset.sh asl-semcom-rgb manifest.json
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/kouvakis-2024-semantic-asl/scripts/modal_app.py::preflight
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/kouvakis-2024-semantic-asl/scripts/modal_app.py::train --dataset mnist
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/kouvakis-2024-semantic-asl/scripts/modal_app.py::train --dataset rgb
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/kouvakis-2024-semantic-asl/scripts/modal_app.py::evidence
python3 .agents/skills/reproduce-paper/scripts/validate_reproduction.py papers/kouvakis-2024-semantic-asl
```

Existing completed output directories return their retained metrics without overwriting them; interrupted runs resume their model, optimizer, epoch history and RNG checkpoint. Setup, training and full-test evaluation are contained in the Modal entry points. A new scientifically independent attempt should use a new output directory and a new declared run record. Dataset acquisition verifies both pinned archive SHA-256 values and refuses a different manifest version.

GPT-6 using Codex performed source discovery, implementation, execution and reporting. This identity comes from session instructions; exact model ID and harness version were not exposed. The tracker record's operator identifiers are redacted, its original export hash is preserved, and no legacy `confirmation` field is invented for a current schema that lacks it. This is a direct user-authorized tracker assignment, with final paper status and a separate database ID. No numerical closeness is called a scientific success or failure; the declared criterion is agreement at the published two-decimal precision for comparable produced metrics, leaving interpretation to review.

A second GPT-6/Codex agent, `codex-orchestrator-reviewer`, independently reviewed the report, metric formulas, label mapping and training implementation on 2026-10-01. Its explicit session attestation supplies those model/application names; exact model ID and harness version remain unavailable. This review clarified the documentation of MNIST interpolation versus RGB resizing. It did not change the implementation, run any experiment, or alter execution attribution.

## Conclusion assessment

All three recognition runs completed. **RGB 97.83333% and LEXSET 97.91667% are nearly tied**, and both exceed MNIST 88.38539%. RGB uses 10,490 training examples versus 21,600 for LEXSET and 27,455 for MNIST. Similar RGB accuracy with fewer training images is retained descriptively. The paper's strict RGB-best ordering is not reproduced: LEXSET is 0.08333 percentage points higher in this single-seed reconstruction. The reported absolute scores are also not recovered: LEXSET is 1.03333 points below 98.95%, RGB 1.88667 below 99.72%, and MNIST 8.73461 below 97.12%.

These different datasets do not provide a controlled causal comparison of image representations. The very small LEXSET/RGB gap is not evidence of statistically reliable superiority. Higher training than test accuracy also does not substantiate the paper's assertion of no overfitting. Three historical validation scores remain unproduced because their membership is unspecified. Communication/channel simulations were outside the assigned recognition scope.

The 2026-10-03 LEXSET continuation is executed by a separate GPT-6/Codex agent (`lexset-continuation-agent`); exact model and harness versions are not exposed. It preserves all 76 previously produced target records and does not rerun MNIST or RGB. Section IV-B specifies 60 LEXSET epochs; the same Figure 3 CNN, Adam defaults, batch 64, seed 42 and final-epoch selection apply. Its additional ceiling is six GPU-hours and CHF25 including diagnostics. The initial CPU acquisition exposed a directory-name assumption (`Test_Alphabet`, not `test`); the scoped mapping correction reuses the already checksum-verified archive. A second audit decoded all 27,000 sample images, then rejected an additional root `alphabet.jpg` overview as unassigned. The third audit attempted existing-file CRC/SHA/decode checks but was stopped after more than seven minutes without a 3,000-file milestone, projecting beyond its one-hour ceiling. The fourth audit stages the pinned archive locally, validates each ZIP member CRC, SHA-256 and image decode, and inventories the overview separately. The canonical data is the verified archive plus its manifest; the partially re-audited extracted tree remains an unused noncanonical leftover. Training reads exact manifest-selected archive bytes after archive and per-image SHA checks, using identical Pillow resizing.


LEXSET's representative preflight loaded all 21,600/2,400 images, trained four batches, restored a fresh CNN and Adam optimizer with exact tensor checks, took a resumed step, and evaluated 128 test images. Its 3.125% accuracy was diagnostic only. Peak allocated memory was 1.63 GB; cold throughput 361.685 images/s predicted approximately 4,151 seconds including loading/evaluation. The prospective full ceiling was five GPU-hours/CHF18 within the additional six GPU-hours/CHF25 aggregate allowance.

The full run `lexset-full-v1` completed in **542.535 seconds (0.150704 GPU-hours)**, app `ap-riaYpafepN5mW0emUdwkt6`, call `fc-01M41H02F0TGPGABJZWPG2JZH9`, one execution segment, native exit 0. It ran from 2026-10-03 18:39:44 to 18:48:46 UTC, before the immutable 23:39:43 UTC deadline. All 60 epochs completed; the final checkpoint produced **2,350/2,400 correct** with 100 test samples per letter. Final online training accuracy was 99.48611%; warm training throughput 6,594 images/s. Total new preflight plus full GPU time was 0.226769 hours. Estimated full compute cost is CHF0.453 using the recorded conservative conversion/rate; actual billing and storage cost are unavailable. CPU data acquisition used no GPU.

Final checkpoint SHA-256: `7568f82afbe19e43ccb34a05a9d48ae164193aa162665392cbb1f4e30434b43b`. Predictions SHA-256: `7d6b8f874d0a9397e76e80811c042a966a343c464ec21debb868dc57fc649df7`. Closed CPU hash collection independently verified the metrics, predictions and checkpoint after GPU execution ended, and preserved every original MNIST/RGB artifact hash and all 76 prior target objects. Raw artifacts are in `modal://repro-sign/repro-2248c066-results/lexset-full-v1/`; exact hashes, source pins, package freeze, GPU/driver details and execution records are in `reproduction.json`.

For the LEXSET path, run these commands from repository root. The acquisition is idempotent. A new independent training reproduction uses a new run ID and requires its own prospective ledger entry; the completed `lexset-full-v1` is immutable and is not restarted. Provider recovery is limited to the same call, two total segments and the original deadline, with a durable checkpoint required.

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/kouvakis-2024-semantic-asl/scripts/modal_app.py::populate_lexset --run-id lexset-data-repeat
.agents/skills/reproduce-paper/scripts/check_modal_dataset.sh synthetic-asl-alphabet manifest.json
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/kouvakis-2024-semantic-asl/scripts/modal_app.py::lexset_run --run-id lexset-full-repeat --expected-manifest-sha256 de9923d8de3ae1e16dc6f4da7fd91251d26af19689a8fb1ecdb8656438698ca0
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/kouvakis-2024-semantic-asl/scripts/modal_app.py::evidence
```
