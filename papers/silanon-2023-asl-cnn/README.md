# Novel classification layer technique for ASL analysis using CNN — reproduction

**Paper ID:** `03785db2dc052c9c21dfefc690b03b8ad0703d9d`

**Citation:** P. Silanon and S. Lertchuwongsa, “A Novel Classification Layer Technique for ASL Analysis Using CNN,” *International Joint Conference on Computer Science and Software Engineering*, 2023. DOI: [10.1109/JCSSE58229.2023.10202089](https://doi.org/10.1109/JCSSE58229.2023.10202089).

**Paper:** user-provided PDF (SHA-256 `73e8eb64c28f3aa4ea5ab86304cf9b2a6f224dd58c96a29bbb7249a82a3e732d`) · **Code:** none found after repository and author searches

**Preference level:** 3

**Status:** `complete`

**Numerical agreement:** not fully reproduced

Every Table III target was produced by a fully documented conditional reconstruction; the paper provides no executable artifact.

**Attempt date:** 2026-08-28

## Scope and target contract

The requested target is every reported number in Table III: training and augmented-validation accuracy for SLR and OCNN; the two corresponding RBF-SVM variants; and augmented-validation accuracy for the averaged E-OCNN-SLR ensemble. The dataset is the 29-class ASL Alphabet release. The paper uses a random 80:20 image split (69,600/17,400), online augmentation for CNN training, and a fixed 10-fold offline augmentation of validation (174,000 images). It reports no test split, seed, or uncertainty interval.

The implementation includes the two CNNs, their probability-average ensemble, and a conditional RBF-SVM attempt. The SVM feature layer, `C`, `gamma`, scaling, and exact offline transforms are not reported, so the selected values are recorded below rather than attributed to the authors.

## Source provenance

| Artifact | Canonical source | Pin | Role |
| --- | --- | --- | --- |
| Paper PDF | User-provided [`03785db2dc052c9c21dfefc690b03b8ad0703d9d.pdf`](/Users/amitmoryossef/Downloads/03785db2dc052c9c21dfefc690b03b8ad0703d9d.pdf) | SHA-256 above | Table III and protocol |
| Bibliographic record | [IEEE DOI](https://doi.org/10.1109/JCSSE58229.2023.10202089) | accessed 2026-08-27 | Canonical citation |
| Dataset | [Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) | version 1 archive SHA-256 `7c572f14fbaff94f98835cfe71c7582dd379a5176e7c4f83dbf3a30e4b3f68c4` | Exact 29-class image release |

No author repository, release, configuration, or weights were found. A GitHub search and author/repository search were completed before selecting preference level 3.

## Results

The retained runs used the stated 80:20 image split, 10 training epochs, Table I augmentation ranges, and the stated 10× augmented validation cardinality. They produced every Table III value below. These are valid results of this explicitly conditional reconstruction, but not a numerical match: the exact author split, saved offline validation images, convolution padding, optimizer, and SVM configuration are unavailable.

| Target ID | System | Split | Original | Reproduced | Evidence |
| --- | --- | --- | ---: | ---: | --- |
| slr-train | SLRNet-8 | checkpoint training | 98.41% | 97.20% | `full-threads/run.json` |
| slr-validation | SLRNet-8 | 10× augmented validation | 83.50% | 98.00% | `full-threads/run.json` |
| ocnn-train | OCNN | checkpoint training | 96.85% | 95.74% | `full-threads/run.json` |
| ocnn-validation | OCNN | 10× augmented validation | 84.68% | 96.11% | `full-threads/run.json` |
| slr-svm-train | SLRNet-8 + RBF SVM | checkpoint training | 99.45% | 99.81% | `svm-full/run.json` |
| slr-svm-validation | SLRNet-8 + RBF SVM | 10× augmented validation | 85.41% | 99.78% | `svm-full/run.json` |
| ocnn-svm-train | OCNN + RBF SVM | checkpoint training | 99.01% | 99.63% | `svm-full/run.json` |
| ocnn-svm-validation | OCNN + RBF SVM | 10× augmented validation | 85.74% | 99.45% | `svm-full/run.json` |
| e-ocnn-slr-validation | E-OCNN-SLR | 10× augmented validation | 87.01% | 98.67% | `full-threads/run.json` |

## How to repeat

The shared `datasets` Volume must contain `asl-alphabet/manifest.json`. The Modal wrapper fails closed unless the `repro-sign` profile is authenticated.

```bash
.agents/skills/reproduce-paper/scripts/check_modal_dataset.sh asl-alphabet manifest.json
cnn_run="cnn-$(date -u +%Y%m%dT%H%M%SZ)"
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run \
  papers/silanon-2023-asl-cnn/modal_app.py::train \
  --run-name "$cnn_run"

.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run \
  papers/silanon-2023-asl-cnn/modal_app.py::svm_full \
  --run-name "svm-$(date -u +%Y%m%dT%H%M%SZ)" \
  --weights-run-name "$cnn_run"
```

Each run name creates a new immutable directory in the paper-specific Modal output Volume; `run.json` holds the metrics, selected epochs, per-epoch histories, dataset manifest hash, and exact inferred configuration. Existing evidence is never overwritten.

## Data provenance and permissions

| Dataset | Version/subset/splits | Permission basis | Modal path | Counts / identity |
| --- | --- | --- | --- | --- |
| ASL Alphabet | Kaggle v1; all 29 stored classes; deterministic, seeded class-stratified 80:20 image split | Project’s existing authorized ASL Alphabet acquisition; original dataset metadata declares GPL-2.0 | `/datasets/asl-alphabet` | 87,000 images; 3,000/class; `manifest.json` SHA-256 `012e786c2f72e1f731f4384adbcf190c4e7084f80c64c8c17e3ad585693a453d` |

The 29 classes are `A`–`Z`, `SPACE`, `DELETE`, and `NOTHING`; none are excluded because the paper names and counts the full 29-class release. No videos are decoded.

## Environment and implementation

The Modal image is `tensorflow/tensorflow:2.15.0-gpu` (resolved digest `sha256:206b54412f00b02c0ebaf00f359281c4032a5b97b20a682b04496a29860230a6`) with `Pillow==10.2.0`, `SciPy==1.11.4`, and `scikit-learn==1.4.2`, on one NVIDIA L4 (20,833 MiB, compute capability 8.9). It mounts the shared `huggingface-cache` at `/cache/huggingface` with `HF_HOME` and `HF_HUB_CACHE`; this image-only experiment does not download Hub artifacts.

There are no patches because no published code exists. `train.py` implements the CNNs and ensemble: Figure 4 supplies SLRNet-8’s visible layers; Section III.B/Table II supply OCNN’s five convolutions, pooling, batch normalization, dropout, dense layers, and output regularization; Section III.C supplies the average-probability ensemble. `svm.py` is the small separate Table III SVM evaluator. Comments identify the paper-derived items.

## Decisions not specified by the paper

| Detail | Paper says | This attempt uses | Why |
| --- | --- | --- | --- |
| Random split and augmentation seed | Random 80:20 split; no seed | `2026`, stratified by class | Makes the reported conditional reconstruction repeatable without claiming the authors’ split |
| Image resizing | SLRNet-8 source architecture takes 64×64; Table III does not repeat size | 64×64 RGB | Only explicit SLRNet-8 input evidence; applied consistently to both networks |
| Convolution padding | Not specified | `same` | Preserves the Figure 4 spatial path for the listed pooling layers |
| Optimizer / learning rate | Not specified | Adam, `1e-3` | Conventional minimal Keras default-like choice, recorded rather than inferred as author setting |
| CNN checkpoint rule | Table III pairs training accuracy with maximum validation accuracy | Select maximum augmented-validation accuracy | Matches the table’s stated comparison framing |
| Augmentation samples | Ranges/factor stated, sampled transforms not | Deterministic ImageDataGenerator transforms | Exactly implements stated ranges and tenfold cardinality |
| RBF-SVM details | Only “RBF kernel” and 696,000 augmented training images | SLR global-average-pool / OCNN flatten; scikit-learn `SVC(C=1, gamma="scale")` | Final tensors before the inferred dense classifier stages; defaults are explicit rather than score-tuned |

## Execution evidence and attempts

| Run | Hypothesis and result |
| --- | --- |
| `ap-tOF4t8PHgEABQJ6kJrh6GP` | Initial real-data preflight failed before training because Pillow was absent from the TensorFlow base image. Added the missing image dependency. |
| `ap-bVda6OAr7tS5O2p1CHbB0n` | The next preflight reached augmentation, then failed because ImageDataGenerator needed SciPy. Added the missing documented runtime dependency. |
| `ap-O2GZJS0UD3qCggsag1Nku3` | Real-data path completed: both models trained, saved/reloaded their best checkpoints, and evaluated. Tiny-subset ensemble validation accuracy was 8.6207%; it is path evidence only. |
| `ap-oaR3Dx9lXq7L0TbgTNE1It` | The first full run was stopped during epoch 1: VolumeFS image reads were serial, the GPU was idle, and the projected run exceeded its 12-hour ceiling. Partial output is retained, but is not evaluated or reported. |
| `ap-aYHbFDLXKnLWTWk9OxndM1` | Measured the actual input bottleneck over 928 real JPEGs: 59.03 s serial versus 1.15 s with eight threads. |
| `ap-pgI9ZErpb0k0ZP6X5Pt74K` | Real-data threaded-loader preflight completed: both models trained, saved/reloaded checkpoints, and evaluated. Tiny-subset ensemble accuracy was 9.4828%; it proves the path only. |
| `ap-DuMtpPTQjVjbfF5EGtZlaA` | Terminal full real-data run on L4. SLR selected epoch 7 (97.20% training / 98.00% validation); OCNN selected epoch 10 (95.74% / 96.11%); probability-average ensemble was 98.67%. Raw JSON SHA-256 `d44eb667395866fc68727926d17b443a655f72c6beb7c8a10c3055e32d2746ff`. |
| `ap-N9rmp6VzHpboWeAXWZ9lii` / `ap-Ni5UKRAcpc6u845IhXbJFs` | Conditional exact-RBF SVC preflight and scale measurement. At 23,200 augmented training examples, fitting took 4.19 s (OCNN) and 4.78 s (SLR); feature extraction, not SVC fitting, dominates. |
| `ap-ZEdZmXIbJTSGEGTN76O1Z7` | Terminal full conditional RBF-SVM run on L4: 696,000 augmented training and 174,000 augmented validation examples. SLR-SVM: 99.81% training / 99.78% validation; OCNN-SVM: 99.63% / 99.45%. Feature extraction took 2,828.84 s (SLR) and 2,223.92 s (OCNN); fitting took 2,249.80 s and 1,032.21 s. Raw JSON SHA-256 `2f763720f6fd6d8f2bc4859c20190d04f9dceb36390ceb4ff27a1259c5040ed0`. |

The full CNN run lasted 4.29 L4-hours; the SVM evaluation lasted 9.12 L4-hours. `reproduction.json` records source commits, image digest, dependency manifest, timestamps, ceilings, and dashboard URLs. Modal exposed app IDs for these detached runs but not function-call IDs; those fields are explicitly `null`, not inferred. All Modal calls use workspace/profile `repro-sign`. No author or Team S contact was needed; the already-authorized public dataset was present in the shared v2 dataset volume.
