# Novel classification layer technique for ASL analysis using CNN — reproduction

**Paper ID:** `03785db2dc052c9c21dfefc690b03b8ad0703d9d`

**Citation:** P. Silanon and S. Lertchuwongsa, “A Novel Classification Layer Technique for ASL Analysis Using CNN,” *International Joint Conference on Computer Science and Software Engineering*, 2023. DOI: [10.1109/JCSSE58229.2023.10202089](https://doi.org/10.1109/JCSSE58229.2023.10202089).

**Paper:** user-provided PDF (SHA-256 `73e8eb64c28f3aa4ea5ab86304cf9b2a6f224dd58c96a29bbb7249a82a3e732d`) · **Code:** none found after repository and author searches

**Preference level:** 3 — a conditional reimplementation is necessary because the paper provides no executable artifact.

**Status:** `partial` — real-data preflight completed; the full Table III run is pending.

**Attempt date:** 2026-08-27

## Scope and target contract

The requested target is every reported number in Table III: training and augmented-validation accuracy for SLR and OCNN; the two corresponding RBF-SVM variants; and augmented-validation accuracy for the averaged E-OCNN-SLR ensemble. The dataset is the 29-class ASL Alphabet release. The paper uses a random 80:20 image split (69,600/17,400), online augmentation for CNN training, and a fixed 10-fold offline augmentation of validation (174,000 images). It reports no test split, seed, or uncertainty interval.

The implementation below has only the two CNNs and their probability-average ensemble. The SVM variants are deliberately not fabricated: their feature layer, `C`, `gamma`, scaling, and exact offline augmentation instances are not reported. Those missing choices are recorded below before any conditional SVM reconstruction is attempted.

## Source provenance

| Artifact | Canonical source | Pin | Role |
| --- | --- | --- | --- |
| Paper PDF | User-provided [`03785db2dc052c9c21dfefc690b03b8ad0703d9d.pdf`](/Users/amitmoryossef/Downloads/03785db2dc052c9c21dfefc690b03b8ad0703d9d.pdf) | SHA-256 above | Table III and protocol |
| Bibliographic record | [IEEE DOI](https://doi.org/10.1109/JCSSE58229.2023.10202089) | accessed 2026-08-27 | Canonical citation |
| Dataset | [Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) | version 1 archive SHA-256 `7c572f14fbaff94f98835cfe71c7582dd379a5176e7c4f83dbf3a30e4b3f68c4` | Exact 29-class image release |

No author repository, release, configuration, or weights were found. A GitHub search and author/repository search were completed before selecting preference level 3.

## Results

The retained preflight uses only ten images per class and two validation augmentations, so it proves the real data/training/checkpoint/evaluation path but is not comparable to Table III. Full results will replace the `Not produced` cells after the fixed 80:20 / 10-fold-validation run reaches terminal state.

| Target ID | System | Split | Original | Reproduced | Evidence |
| --- | --- | --- | ---: | ---: | --- |
| slr-train | SLRNet-8 | checkpoint training | 98.41% | Not produced | Full run pending |
| slr-validation | SLRNet-8 | 10× augmented validation | 83.50% | Not produced | Full run pending |
| ocnn-train | OCNN | checkpoint training | 96.85% | Not produced | Full run pending |
| ocnn-validation | OCNN | 10× augmented validation | 84.68% | Not produced | Full run pending |
| slr-svm-train | SLRNet-8 + RBF SVM | checkpoint training | 99.45% | Not produced | Unspecified SVM protocol |
| slr-svm-validation | SLRNet-8 + RBF SVM | 10× augmented validation | 85.41% | Not produced | Unspecified SVM protocol |
| ocnn-svm-train | OCNN + RBF SVM | checkpoint training | 99.01% | Not produced | Unspecified SVM protocol |
| ocnn-svm-validation | OCNN + RBF SVM | 10× augmented validation | 85.74% | Not produced | Unspecified SVM protocol |
| e-ocnn-slr-validation | E-OCNN-SLR | 10× augmented validation | 87.01% | Not produced | Full run pending |

## How to repeat

The shared `datasets` Volume must contain `asl-alphabet/manifest.json`. The Modal wrapper fails closed unless the `repro-sign` profile is authenticated.

```bash
.agents/skills/reproduce-paper/scripts/check_modal_dataset.sh asl-alphabet manifest.json
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run \
  papers/silanon-2023-asl-cnn/modal_app.py::train
```

The result is an immutable run directory in the paper-specific Modal output Volume; `run.json` holds the metrics, selected epochs, per-epoch histories, dataset manifest hash, and exact inferred configuration.

## Data provenance and permissions

| Dataset | Version/subset/splits | Permission basis | Modal path | Counts / identity |
| --- | --- | --- | --- | --- |
| ASL Alphabet | Kaggle v1; all 29 stored classes; deterministic, seeded class-stratified 80:20 image split | Project’s existing authorized ASL Alphabet acquisition; original dataset metadata declares GPL-2.0 | `/datasets/asl-alphabet` | 87,000 images; 3,000/class; `manifest.json` SHA-256 `012e786c2f72e1f731f4384adbcf190c4e7084f80c64c8c17e3ad585693a453d` |

The 29 classes are `A`–`Z`, `SPACE`, `DELETE`, and `NOTHING`; none are excluded because the paper names and counts the full 29-class release. No videos are decoded.

## Environment and implementation

The Modal image is `tensorflow/tensorflow:2.15.0-gpu` with `Pillow==10.2.0` and `SciPy==1.11.4`, on an L4 GPU. It mounts the shared `huggingface-cache` at `/cache/huggingface` with `HF_HOME` and `HF_HUB_CACHE`; this image-only experiment does not download Hub artifacts.

There are no patches because no published code exists. `train.py` is the only implementation: Figure 4 supplies SLRNet-8’s visible layers; Section III.B/Table II supply OCNN’s five convolutions, pooling, batch normalization, dropout, dense layers, and output regularization; Section III.C supplies the average-probability ensemble. Comments identify the paper-derived items.

## Decisions not specified by the paper

| Detail | Paper says | This attempt uses | Why |
| --- | --- | --- | --- |
| Random split and augmentation seed | Random 80:20 split; no seed | `2026`, stratified by class | Makes the reported conditional reconstruction repeatable without claiming the authors’ split |
| Image resizing | SLRNet-8 source architecture takes 64×64; Table III does not repeat size | 64×64 RGB | Only explicit SLRNet-8 input evidence; applied consistently to both networks |
| Convolution padding | Not specified | `same` | Preserves the Figure 4 spatial path for the listed pooling layers |
| Optimizer / learning rate | Not specified | Adam, `1e-3` | Conventional minimal Keras default-like choice, recorded rather than inferred as author setting |
| CNN checkpoint rule | Table III pairs training accuracy with maximum validation accuracy | Select maximum augmented-validation accuracy | Matches the table’s stated comparison framing |
| Augmentation samples | Ranges/factor stated, sampled transforms not | Deterministic ImageDataGenerator transforms | Exactly implements stated ranges and tenfold cardinality |
| RBF-SVM details | Only “RBF kernel” and 696,000 augmented training images | Not run yet | Feature layer and hyperparameters are material, unreported protocol choices |

## Execution evidence and attempts

| Run | Hypothesis and result |
| --- | --- |
| `ap-tOF4t8PHgEABQJ6kJrh6GP` | Initial real-data preflight failed before training because Pillow was absent from the TensorFlow base image. Added the missing image dependency. |
| `ap-bVda6OAr7tS5O2p1CHbB0n` | The next preflight reached augmentation, then failed because ImageDataGenerator needed SciPy. Added the missing documented runtime dependency. |
| `ap-O2GZJS0UD3qCggsag1Nku3` | Real-data path completed: both models trained, saved/reloaded their best checkpoints, and evaluated. Tiny-subset ensemble validation accuracy was 8.6207%; it is path evidence only. |
| `ap-vrW6ASmr0dT7FhNnTWwn3k` | Eight loader workers made the tiny real-data path about eight times slower than the baseline due to process/IPC overhead. That speculative throughput change was reverted; the retained full recipe uses the faster single-process loader. |

All Modal calls use workspace/profile `repro-sign`. No author or Team S contact was needed; the already-authorized public dataset was present in the shared v2 dataset volume.
