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

## Decisions and assumptions

| Missing or unclear detail | Choice used here |
| --- | --- |
| Split membership and randomness | Stratified image split and seed 42. No claim of evaluation on unseen signers. |
| Optimizer, learning rate and batch size | Adam, learning rate 0.0001, batch size 32, no learning-rate schedule. |
| Classification head and trainable layers | Global pooling or the transformer class token, followed directly by a 32-class softmax, without the paper's added dense layer (see Implementation differences); train all layers. |
| Exact pretrained checkpoints and ViT variant | Pinned ImageNet weights; ImageNet-21k BiT-M; ImageNet-21k+ImageNet-2012 ViT weights. Interpret `ViT_132` as ViT-L/32. |
| Input processing | Convert to grayscale and repeat into RGB; bilinear resize to 224×224, or 256×256 for Swin; backbone-specific normalization. |
| CNN-3 architecture details | Table 4's V4 filters: 256, 126, 64, 62, 16. Same-padded 3×3 convolutions, valid 2×2 pooling, LeakyReLU 0.01, batch normalization after the last convolution, dropout 0.5 and Dense(128), which Table 4 does not list (see Implementation differences). |
| CNN augmentation | Horizontal/vertical flips and rotation within ±0.2 degrees. No random augmentation for transfer models. |
| Loss and multiclass metric definitions | Train with cross-entropy; report cross-entropy and probability MSE separately. Use argmax accuracy, macro precision/recall/F1 and macro one-vs-rest AUC. |
| Checkpoint selection and precision | Report epoch one and the final epoch, without best-score selection. Float32 with TensorFlow's default TF32 enabled. |

The paper defines MSE but reports losses as high as 17.744, which probability-space MSE cannot explain. Its averaging and threshold rules are also unclear. Neither of our loss calculations is treated as the recovered original definition.

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
