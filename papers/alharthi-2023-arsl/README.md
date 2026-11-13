# Arabic sign language recognition reproduction

Alharthi and Alzahrani, *Vision Transformers and Transfer Learning Approaches for Arabic Sign Language Recognition*. Applied Sciences, 2023. [Paper](https://doi.org/10.3390/app132111625).

**Preference level:** 3

**Pipeline status:** `partial` — all 16 models completed; 28 accuracy targets were assessed. The four VGG16/VGG19 accuracy targets are excluded (see [Implementation differences](#implementation-differences-from-the-paper)), and 96 other metric targets remain unresolved.

**Numerical agreement:** `does_not_agree` — the 28 assessed accuracy values differ from the published values.

**Attempt date:** 2026-11-04.

The paper's [official repository](https://github.com/SalhaAlzahrani/ArSL_Classification) is empty, and we found no alternative implementation. We therefore reconstructed the experiments from the paper. The scope covers all 128 scores in Table 7, including the four baseline models.

## Implementation differences from the paper

Two classifier-head choices in this reconstruction differ from what the paper states. Codex seems to have made a mistake here:

- **Transfer models.** The paper fine-tunes each pretrained model "using an added dense layer and one classification layer". Our implementation connects the pooled backbone features (or the ViT class token) directly to the 32-class softmax, with no added dense layer. The paper does not give that layer's width.
- **CNN-3.** Table 4 lists CNN-3 as five convolution layers followed by "fully connected layer + softmax". Our implementation adds a Dense(128) layer between the flattened features and the softmax. Our CNN-3 has 519,628 parameters, against 431,146 trainable parameters for CNN-3V4 in Table 4; it would have about 440,000 without the Dense(128) layer.

The reviewer considers these differences minor except for VGG16 and VGG19. Both VGG runs collapsed from the first epoch to predicting a single class, which the reviewer attributes to the missing dense layer leaving the classifier with too few parameters. Their four accuracy targets are therefore excluded; the raw outputs remain in [reproduction.json](reproduction.json) as conditional evidence. This attribution has not been tested with a corrected run.

## Results

All models completed training: 100 epochs for CNN-3 and ten for each transfer model. The table shows final validation accuracy on 5,404 images for the 14 models in the assessment.

| Model | Paper accuracy (%) | This attempt (%) |
| --- | ---: | ---: |
| CNN-3 | 63.23 | 97.02 |
| MobileNet | 96.41 | 99.28 |
| MobileNetV2 | 96.65 | 99.22 |
| Xception | 94.08 | 99.41 |
| InceptionV3 | 88.51 | 98.87 |
| InceptionResNetV2 | 98.17 | 99.24 |
| DenseNet169 | 55.87 | 99.43 |
| BiT-M R50x1 | 95.08 | 99.24 |
| BiT-M R50x3 | 90.04 | 99.09 |
| ViT-B/16 | 89.19 | 99.22 |
| ViT-L/32* | 88.68 | 99.11 |
| SwinV2 Tiny | 97.65 | 99.44 |
| ResNet50 | 98.24 | 99.11 |
| ResNet152 | 91.10 | 99.17 |

*The paper's `ViT_132` was interpreted as ViT-L/32.

All 14 models reached 97.02–99.44% accuracy. VGG16 and VGG19 are excluded, as explained above.

We treat the documented training choices as reasonable reconstruction assumptions and assess the 28 first-epoch and final accuracy targets of the remaining 14 models using ordinary classification accuracy. All differ from the published values beyond their four-decimal rounding precision; this is not a statistical significance test. The table above shows final accuracy. We completed the experiments using the documented assumptions below; differences in split membership and training settings may contribute to the numerical differences. Higher scores therefore do not establish an improvement over the paper. The remaining 96 loss, precision, recall, F1 and AUC targets remain unresolved because their definitions are unclear; the computed values are retained as conditional evidence. This is why target coverage is partial despite all training runs having completed. The first-epoch results, precision, recall, F1, AUC, both loss calculations and all differences from Table 7 are in [reproduction.json](reproduction.json).

## Setup

We used the 54,049-image, 32-class [ArASL dataset, version 1](https://data.mendeley.com/datasets/y7pckrw6z2/1), available under CC BY 4.0, through the pinned grayscale mirror already stored in the project. A stratified split gives 48,645 training and 5,404 validation images. Its membership is our choice; the paper does not supply the original split.

Training used TensorFlow/Keras 2.8, Python 3.10 and CUDA 11.2.2/cuDNN 8. The container is defined in [scripts/modal_app.py](scripts/modal_app.py), with resolved packages in [requirements.lock](requirements.lock). Each run records its environment, data and pretrained weight hashes.

## Guesses and deviations

The paper defines MSE but reports losses as high as 17.744, which probability-space MSE cannot explain. Its averaging and threshold rules are also unclear. Neither of our loss calculations is treated as the recovered original definition.

| ID | Kind | Description | Paper says | This attempt used | Basis | Effect | Scope | Status | Evidence | Related IDs | Affects | Dataset IDs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `split-and-randomness` | `guess` | Split membership and randomness. | "48645 images for training and 5404 files for validation" (§5.1); membership and seed not given | Stratified image split and seed 42. No claim of evaluation on unseen signers. |  |  |  |  |  |  |  | `arasl2018` |
| `optimizer-lr-batch` | `guess` | Optimizer, learning rate and batch size. | Not stated | Adam, learning rate 0.0001, batch size 32, no learning-rate schedule. |  |  |  |  |  |  |  |  |
| `classifier-head` | `paper_deviation` | Classifier heads of the transfer models and CNN-3. | Transfer models: fine-tuning "using an added dense layer and one classification layer" (§3.1), width not reported. CNN-3: Table 4 lists five convolutions followed by "fully connected layer + softmax" and 431,146 trainable parameters for CNN-3V4; §5.1 says "two fully connected layers were utilized" | Transfer models: the pooled backbone features or the ViT class token connect directly to the 32-class softmax, with no added dense layer (scripts/models.py). CNN-3: a Dense(128) layer before the softmax, giving 519,628 parameters (about 440,000 without Dense(128)) |  | Human reviewer judgment (2026-11-04): the head differences are minor for CNN-3 and all transfer models except VGG16 and VGG19. For VGG, the missing added dense layer left the classifier with too few parameters; the reviewer attributes the observed single-class collapse to it and excludes the VGG accuracy targets. The attribution has not been tested with a corrected run. |  |  | scripts/models.py |  |  |  |
| `trainable-layers` | `guess` | Which layers are trained. | Not stated | Train all layers |  |  |  |  |  | `classifier-head` |  |  |
| `pretrained-checkpoints` | `guess` | Exact pretrained checkpoints and ViT variant. | Model families only; ViT variants "ViT_b16 and ViT_132", BiT "BiT-m-r50x1 and BiT-m-r50x3", Swin "SwinV2Tiny256" | Pinned ImageNet weights; ImageNet-21k BiT-M; ImageNet-21k+ImageNet-2012 ViT weights. Interpret `ViT_132` as ViT-L/32. |  |  |  |  |  |  |  |  |
| `input-processing` | `guess` | Input processing. | "ArSL images with three RGB color channels were resized to 224 × 224 pixels" (§3.3, for the CNN models) | Convert to grayscale and repeat into RGB; bilinear resize to 224×224, or 256×256 for Swin; backbone-specific normalization. |  |  |  |  |  |  |  | `arasl2018` |
| `cnn3-architecture` | `guess` | CNN-3 architecture details. | Table 4, CNN-3V4: filters 256, 126, 64, 62, 16; 3 × 3 convolutions; 2 × 2 max pooling; "+1 batch normalization, 1 dropout"; leaky ReLU | Table 4's V4 filters: 256, 126, 64, 62, 16. Same-padded 3×3 convolutions, valid 2×2 pooling, LeakyReLU 0.01, batch normalization after the last convolution, dropout 0.5 and Dense(128), which Table 4 does not list (see Implementation differences). |  |  |  |  |  | `classifier-head` |  |  |
| `cnn-augmentation` | `guess` | CNN augmentation. | "random rotation with 0.2 degrees and random horizontal and vertical flipping" (§3.3) | Horizontal/vertical flips and rotation within ±0.2 degrees. No random augmentation for transfer models. | Literal reading of "0.2 degrees" (scripts/models.py: RandomRotation(0.2/360.)) | If the paper's 0.2 is the Keras RandomRotation factor, the intended range is ±0.2 × 360° = ±72°; the literal reading makes rotation negligible |  |  |  |  |  |  |
| `loss-and-metrics` | `guess` | Loss and multiclass metric definitions. | Loss defined as MSE, 1/N ∑(ypred − ytrue)² (§4.4), while reported losses reach 17.744 (VGG19, Table 7); precision, recall, F1 and AUC given as binary formulas with no averaging rule | Train with cross-entropy; report cross-entropy and probability MSE separately. Use argmax accuracy, macro precision/recall/F1 and macro one-vs-rest AUC. |  |  |  |  |  |  |  |  |
| `checkpoint-and-precision` | `guess` | Checkpoint selection and precision. | Not stated | Report epoch one and the final epoch, without best-score selection. Float32 with TensorFlow's default TF32 enabled. |  |  |  |  |  |  |  |  |

## Noncompliance

None.

## Execution and evidence

Runs used A100 GPUs on Modal, with up to four models training concurrently. Eight interrupted jobs resumed from verified checkpoints. The conservative total is **13.76 GPU-hours**, including preflights and interruptions, within the approved 22.33-hour ceiling. Actual billed cost was not retrieved.

All 250 epoch evaluations were retained. We independently recomputed seven metrics from each model's first and final predictions: all 224 values matched within floating-point precision. Checkpoint recovery preserved earlier epoch artifacts byte-for-byte.

Compatibility fixes added OpenCV for ViT and mapped Swin positional weights to the installed library's layer layout, checking every pretrained tensor. No settings were tuned to match published scores. The full source search, assumptions, failed attempts, commands and artifact locations remain in [reproduction.json](reproduction.json). No authors were contacted.

Implementation, execution and reporting: GPT-6 using Codex. Exact model and application versions were unavailable.

## How to run

The entry point is [scripts/modal_app.py](scripts/modal_app.py), using the existing dataset in the `repro-sign` workspace. Run Modal commands through `.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh`. Exact invocations are recorded in `reproduction.json`; a new experiment needs fresh run IDs and a declared compute budget.

To validate the report locally:

```bash
python3 .agents/skills/reproduce-paper/scripts/validate_reproduction.py papers/alharthi-2023-arsl
```
