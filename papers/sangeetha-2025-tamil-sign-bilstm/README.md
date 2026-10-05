# Deep Learning-Based Tamil Sign Language Assistance for Speech Disabilities

**Paper ID:** `194facc5ad2db543ddf529ddc43d8afffccfc1e7`
**Citation:** M. Sangeetha and C. Divya Gowri, ICNGCS 2025, pp. 1-9. https://doi.org/10.1109/ICNGCS64900.2025.11183340
**Preference level:** 3

No code was published; the runs below are a reconstruction from the paper.

**Pipeline status:** `insufficient_information` — no reported number could be produced by a sufficiently specified pipeline.
**Numerical agreement:** `not_assessed` — no in-scope target produced a comparable value; the conditional runs are evidence only.

## Targets

The assignment points to section "iv. Result and Discussion". With the user's agreement (2026-10-05), every number reported there and in the learning-rate subsection that follows is a target: 16 in total.

| Source in the paper | Accuracy | Precision | Recall | F1 | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Sec. IV "iv", paragraph 2 (no configuration) | 85.77 | 65.54 | 85.77 | 75.79 | not produced: `target_ambiguous` |
| Sec. IV.G, lr 0.01 (Fig. 4) | 52.75 | 57.12 | 52.75 | 52.91 | not produced: `protocol_ambiguous` |
| Sec. IV.G, lr 1e-4 (Fig. 6) | 96.16 | 96.40 | 96.16 | 96.21 | not produced: `protocol_ambiguous` |
| Sec. IV.G, lr 1e-5 (Fig. 7) | 81.18 | 80.66 | 81.18 | 78.37 | not produced: `protocol_ambiguous` |

**Not targets:**
- **lr 0.001 (Figure 5):** the figure is a copy of Figure 7's screenshot (it shows accuracy 0.8118), and its paragraph repeats the lr 0.01 numbers.
- **The abstract's 95%:** it appears nowhere in the results.
- **The paragraph's own arithmetic:** its "569 out of 663" is 85.82%, not 85.77%, and no figure matches it.

The paper copies no baseline scores.

## Conditional runs

These were approved by the user before launch and are evidence only, not target results. All four learning rates ran with the same invented recipe (see "Reconstruction decisions"): 300 images per class, a stratified 80/20 split (780 test images, 60 per class), 30 epochs and seed 1. The saved model was reloaded and evaluated on the test split.

| lr | Accuracy | Precision | Recall | F1 | Final train acc. | Paper accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 7.69 | 0.59 | 7.69 | 1.10 | 5.7 | 52.75 |
| 0.001 | 7.69 | 0.59 | 7.69 | 1.10 | 7.2 | (no usable value) |
| 1e-4 | 35.38 | 32.99 | 35.38 | 30.91 | 22.4 | 96.16 |
| 1e-5 | 9.74 | 4.25 | 9.74 | 4.74 | 8.5 | 81.18 |

- **The reconstruction underfits at every learning rate.**
  - With lr 0.01 and 0.001 it collapses to one class (loss stays at ln 13).
  - With lr 1e-4 it learns slowly and reaches only 22% training accuracy after 30 epochs.
- **The inputs are not blank.** Applying the same preprocessing to sample images locally kept the hands clearly visible (the skin mask covers 10–23% of the frame).
- **Likely causes are choices the paper leaves open.** Candidates are the three Dropout(0.5) layers, the augmentation, the 3,328-feature LSTM input and the 30-epoch budget. I did not adjust them after seeing the results, because that would be tuning toward the paper.
- **What the numbers show.** They indicate how far an unguided reconstruction can land from the paper. They do not test the paper's claim.

## Why no target is produced

- **Protocol:** the paper names the components but almost none of their settings:
  - which 3,903 images were used and how they were split;
  - the convolution filter counts and pooling;
  - the LSTM size and how a single image becomes a Bi-LSTM sequence;
  - the dropout rate, epochs, batch size and augmentation;
  - the HSV thresholds, gamma value and blur kernel;
  - the seed and the checkpoint rule.
- **Section IV paragraph:** its numbers have no learning rate or figure attached and match nothing in Section IV.G.

## Data

| Item | Value |
| --- | --- |
| Dataset | TLFS23 v2, [Mendeley Data 39kzs5pxmk](https://data.mendeley.com/datasets/39kzs5pxmk/2), CC BY 4.0 |
| Stored | folders 1–13 only (அ ஆ இ ஈ உ ஊ எ ஏ ஐ ஒ ஓ ஔ ஃ), 1,000 JPEGs each at 640x480, each checked against Mendeley's SHA-256 |
| Location | Modal Volume `datasets`, path `tlfs23/`, mounted read-only |
| Identifier | `MANIFEST.sha256` (13,000 files), SHA-256 `408119efcd791d64ff144890cea44c8810968f1afff5a4b00cbd4e25085d7318` |

**How this differs from what the paper used:**
- **Size and folder names:** Figure 2 shows 3,903 images in 13 classes, with folders named like `1 அ(a)` and `10 ஒ(o)`. The public release has 1,000 images per class in folders named `1`–`13`. The authors' subset and copy are unknown, which supports the queue reviewer's note that the data is not identical.
- **Collection description:** the text describes the authors' own collection (224x224 RGB, indoor and outdoor), but the abstract calls the data TLFS23.
- **Class count:** Section IV.C lists 12 vowels labelled 0–11, while Figure 2 and every confusion matrix have 13 classes.
- **Duplicates:** 64 stored files are exact duplicates of another file, so a random image-level split, here and probably in the paper, can place the same image in train and test. The images are also video frames, so near-duplicates are common.

**Ethics flag:** the queue flags potential ethical concerns without giving a reason. Only the public TLFS23 release was used, with no new participants and nothing published.

## Reconstruction decisions

None of these is a paper fact.

| Item | Choice | Basis |
| --- | --- | --- |
| Framework | Keras 3.11.3 on the PyTorch backend | Figure 2 shows Keras output |
| Subset | 300 random images per class (seed 1) | Figure 2: 3,903 images over 13 classes |
| Split | stratified 80/20, no validation set | Figure 6 shows about 60 test images per class |
| Preprocessing | HSV mask H 0–20, S 20–255, V 70–255 → 5x5 Gaussian blur → gamma 1.5 → grayscale → 224x224 | steps and order from Sec. IV.D; values chosen |
| Augmentation | rotation 0.05 turns, zoom 0.1, translation 0.1 | "data augmentation" only |
| CNN | 3x3 conv 32/64/128, ReLU, 2x2 max pooling after each | Sec. IV.E: three 3x3 conv layers |
| Bi-LSTM | feature-map rows as 26 timesteps; Bidirectional(LSTM(128)) | mapping not described in the paper |
| Dense head | 256-128-64 ReLU, Dropout(0.5) after each, 13-way softmax | Sec. IV.F.ii widths; dropout rate chosen |
| Training | Adam, categorical cross-entropy, batch 32, 30 epochs, final epoch | Adam and loss stated; rest chosen |
| Metrics | weighted precision, recall and F1 (scikit-learn 1.7.2) | the paper's recall always equals its accuracy |

**Deviations:** one NVIDIA L4 and Keras 3 on PyTorch; the paper states neither hardware nor software.

## Sources and search

| Artifact | Pin | Use |
| --- | --- | --- |
| Full paper (IEEE Xplore, supplied by the user) | SHA-256 `9af5c3c5a6fcb94dd5d73e6645a8730bd5696a241226c841c086eb2dc248ceaf` | targets and protocol; not redistributed |
| [Institutional excerpt](https://ir.psgitech.ac.in/id/eprint/1607/) | SHA-256 `e541aaa00450b8a18932897bd1213a457641df170b0bd22d961ccb350ee1d612` | 3 of 9 pages; superseded |
| [TLFS23 v2](https://data.mendeley.com/datasets/39kzs5pxmk/2) | per-file SHA-256 in the manifest | data |

**Code search (dead end):** no code link in the paper. Web searches on 2026-10-05 for the title with the authors' names, and for the method's keywords with TLFS23, found only the IEEE record, the institutional repository and the TLFS23 data paper.

## Runs and evidence

All four learning rates ran in parallel in one Modal app, [`ap-qhm0er2vzgVMTWNMLFFafu`](https://modal.com/apps/repro-sign/main/ap-qhm0er2vzgVMTWNMLFFafu), from 10:25:20 to 10:44:14 UTC on 2026-10-05. Each ran on one L4 with 4 CPU and 16 GiB, under profile `repro-sign` and environment `main`.

| Run | Function call | Seconds (load to metrics) |
| --- | --- | ---: |
| `full-lr-0.01` | `fc-01M45SHFFPR1GJEMRTTGBZM7XD` | 940 |
| `full-lr-0.001` | `fc-01M45SHFKRW8VTP755RWVQ3FSK` | 991 |
| `full-lr-0.0001` | `fc-01M45SHFR66T0PZFTKAMM01QMP` | 960 |
| `full-lr-1e-05` | `fc-01M45SHFWJJEZMVGM7PYYC5D5P` | 1088 |

- **Stop policy:** declared at 10:25:18 UTC, before launch: 2 h wall time (the function timeout), 2 GPU-hours and CHF 5 per run, one retry allowed. No run failed.
- **Compute:** at most 1.26 GPU-hours in total, about USD 1.4 at list rates.
- **Artifacts:** `run.json` (config, metrics, confusion matrix, per-image predictions, training history), `freeze.txt` and `model.keras` for each run, under `modal://volume/sangeetha-2025-tamil-sign-bilstm-results/lr-*/`. SHA-256 hashes are in `reproduction.json`.
- **Preflight:** app `ap-qnO7UOMHR9oL4nHRg5vXlS` ran 20 images per class for 2 epochs at all four learning rates, covering loading, preprocessing, training, save/reload and metrics, with exit 0. It is documented in `reproduction.json` rather than as a run entry because no stop policy was declared for it.
- **Failed first download:** the first population attempt failed on a transient Mendeley TLS error before anything was uploaded.

## How to repeat

```bash
./setup.sh
papers/sangeetha-2025-tamil-sign-bilstm/data.sh
.agents/skills/reproduce-paper/scripts/check_modal_dataset.sh tlfs23 MANIFEST.sha256
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/sangeetha-2025-tamil-sign-bilstm/modal_app.py
```

- **`data.sh`:** idempotent; downloads and verifies TLFS23 folders 1–13.
- **`modal_app.py`:** launches the four learning rates in parallel and prints each function-call ID. A run whose `run.json` already exists is returned as is.

## Open gates

| Gate | Needed |
| --- | --- |
| `protocol-unspecified` | the authors' code, data subset and split, or acceptance that the runs stay conditional |
| `sec4-paragraph-configuration` | the configuration behind 85.77 / 65.54 / 85.77 / 75.79, or a decision to drop those targets |

## Author contact and closure

None. On 2026-10-05 the user decided to close the attempt without contacting the authors: most important hyperparameters are missing and the reported results are internally inconsistent, so a reply would be unlikely to make any target reproducible. The two gates above remain open, and no further runs are planned.
