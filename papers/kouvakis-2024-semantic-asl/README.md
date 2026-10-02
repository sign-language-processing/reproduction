# Semantic Communications for Image-Based Sign Language Transmission

**TL;DR:** Full MNIST and RGB runs produced 76 of 81 requested numbers; test accuracies were 88.39% and 97.83%, below 97.12% and 99.72%.
**Decision:** LEXSET requires permission. The two undefined validation scores remain unproduced; no completed run is repeated to invent validation membership.

**Pipeline status:** partial

**Numerical agreement:** does_not_agree

**Preference level:** 3

The full MNIST and RGB schedules completed and produced **76 of 81 requested numbers**. LEXSET remains gated by data permission, and two validation scores lack an identified validation partition. There were no failed training attempts or full restarts.

| Dataset / split | Published accuracy (%) | Reproduced (%) | Difference (pp) |
|---|---:|---:|---:|
| MNIST / training | 99.61 | 100.00000 | +0.39000 |
| MNIST / validation | 100 | Not produced | — |
| MNIST / testing | 97.12 | 88.38539 | -8.73461 |
| LEXSET / training | 98.39 | Not produced | — |
| LEXSET / validation | 98.52 | Not produced | — |
| LEXSET / testing | 98.95 | Not produced | — |
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

[LEXSET](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet) version 3 lists **Data files © Original Authors**, without affirmative project-cloud processing terms. Its three requested results are blocked on permission. The expected 24-class subset excludes J, Z and background, giving 21,600/2,400 train/test examples, but no restricted payload was downloaded. Public metadata responses for all three sources are retained under `artifacts/` with hashes. The tracker mentions historical Team S correspondence about RGB; this attempt did not contact authors or rely on unverified correspondence, because the public Zenodo license independently permits the work.

The other open issue is the validation set: section IV-A/Table 1 allocates every listed sample to training or testing, while Table 2 reports validation accuracy without defining its allocation or membership. Creating an arbitrary validation partition would not recover those two scores. All published train/test samples remain in their original partitions for the permitted full runs.

The environment starts from the prescribed `ghcr.io/sign-language-processing/reproduction:latest`, resolving to an immutable Modal image recorded with each run. Native dependency freezes, hardware logs, model/optimizer/RNG checkpoints, metrics with epoch history, and predictions/confusion matrices live in v2 Volume `repro-2248c066-results`; every retained artifact has a SHA-256 in `reproduction.json`. Canonical `datasets` and `huggingface-cache` are both v2, with the cache mounted read-write at `/cache/huggingface` and both required HF environment variables set. An OCI image digest was not exposed; the immutable Modal image ID and native package freeze are preserved instead. All operations use the `repro-sign` wrapper.

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

The tested RGB-versus-MNIST ordering is reproduced: RGB achieves 97.83333% test accuracy using 10,490 training images, versus MNIST 88.38539% using 27,455. The reported absolute test performance is not recovered, particularly MNIST: 88.38539% versus 97.12%. This supports only part of the recognition findings.

These are comparisons across different datasets, not a controlled demonstration that RGB representation alone causes the improvement. LEXSET is permission-blocked, so RGB superiority over all three datasets is not established. Undefined validation membership leaves two validation scores unproduced. Communication/channel simulations were outside the assigned scope; no conclusion about the proposed communication system is reproduced by these recognition runs alone.

This assessment uses the existing completed runs; no training, protocol or numerical-agreement criterion was changed.
