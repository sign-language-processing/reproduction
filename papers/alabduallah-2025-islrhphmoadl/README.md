# ISLRHP-HMOADL reproduction

Alabduallah et al., *Innovative hand pose based sign language recognition using hybrid metaheuristic optimization algorithms with deep learning model for hearing impaired persons*. Scientific Reports, 2025. [Paper](https://doi.org/10.1038/s41598-025-93559-4).

**Preference level:** 3

**Pipeline status:** `insufficient_information` — the reconstruction completed, but missing protocol details prevent a direct reproduction claim.

**Numerical agreement:** `not_assessed`

**Attempt dates:** 2026-11-01 to 2026-11-03.

No usable paper-specific code was found, so we implemented the proposed model from the paper. The scope includes all 60 scores in Table 4. Only the proposed model was run; the baseline targets remain unresolved, as detailed below.

## Results

The model correctly classified **17,395 of 17,400 test images (99.9713%)**, making five errors. All 50 training epochs completed. Validation selected the epoch-10 checkpoint, which was then evaluated on the test set.

| Metric (%) | Paper Table 4 | This attempt |
| --- | ---: | ---: |
| Accuracy | 99.57 | 99.9980 |
| Precision | 94.43 | 99.9713 |
| Sensitivity | 93.70 | 99.9713 |
| Specificity | 99.78 | 99.9990 |
| F1 | 93.89 | 99.9713 |

The table uses unweighted averages over 29 classes. Its accuracy is **one-vs-rest accuracy**, including true negatives; ordinary classification accuracy is the 99.9713% reported above. The paper's aggregation is unclear, so these values are not a verified like-for-like comparison.

## Baseline targets

All 55 baseline scores remain reproduction targets. None was produced in this attempt: the paper does not specify how these comparison results were obtained. We checked references 36–39 for their provenance.

| Baseline | Accuracy | Precision | Sensitivity | Specificity | F1 | What we found in the cited papers |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| DNN-Squeezenet | 90.78 | 91.46 | 92.44 | 97.65 | 92.53 | Shin et al. report 83.28% accuracy on Finger Spelling A. We did not find the claimed 90.78%. |
| CNN Classifier | 93.00 | 91.39 | 92.16 | 97.76 | 92.65 | Shin et al. report the same 93.00% accuracy, but on a different dataset: Finger Spelling A. |
| RBM Algorithm | 98.13 | 93.29 | 92.71 | 98.30 | 92.65 | Shin et al. list the same 98.13% accuracy under Finger Spelling A, a different dataset. |
| ASLAR-HPE | 98.45 | 91.90 | 93.01 | 96.61 | 93.04 | Shin et al. report 98.45% on Finger Spelling A. Their result on ASL Alphabet, the dataset used here, is 87.60%. |
| StackGAN Model | 97.11 | 92.60 | 93.12 | 96.69 | 92.88 | Natarajan et al. evaluate the quality of generated videos. We found no classification accuracy of 97.11%. |
| VGG19 Method | 96.30 | 93.51 | 93.01 | 96.85 | 92.25 | Miah et al. report the same 96.30% accuracy, but on a different dataset: ASL KLP. |
| InceptionResnet | 99.36 | 93.12 | 92.25 | 97.07 | 91.99 | Miah et al. report 99.46% accuracy in their ASL comparison. We did not find the claimed 99.36%. |
| SVM Classifier | 96.89 | 93.76 | 91.94 | 97.89 | 92.60 | Miah et al. list SVM accuracies of 89.70% and 96.30%. Neither matches the claimed 96.89%. |
| MobileNetV2 | 93.58 | 91.89 | 92.74 | 98.28 | 93.22 | Song et al. report 85.66% on INCLUDE for MobileNetV2 combined with Bi-LSTM, a different model and dataset. |
| HA-GCN | 98.93 | 93.82 | 93.38 | 98.86 | 93.35 | Song et al. report 96.82% on AUTSL and 99.63% on INCLUDE. Neither matches the claimed 98.93%. |
| Transformer | 99.05 | 92.59 | 93.03 | 97.26 | 93.08 | Song et al. report 81.03% on AUTSL and 90.44% on INCLUDE. Neither matches the claimed 99.05%. |

Numbers above are the paper's percentages, not reproduced results. Sources: [Shin et al. (reference 36), Tables 8 and 10](https://pmc.ncbi.nlm.nih.gov/articles/PMC8434249/); [Natarajan et al. (reference 37), Tables 2–10](https://doi.org/10.1109/ACCESS.2022.3210543); [Miah et al. (reference 38), author manuscript, Tables 5 and 7](https://www.researchgate.net/publication/378554578_Hand_Gesture_Recognition_for_Multi-Culture_Sign_Language_Using_Graph_and_General_Deep_Learning_Network); [Song et al. (reference 39), Tables 4–6](https://doi.org/10.1016/j.jiixd.2024.08.001).

Four accuracies match prior results on different datasets, suggesting reuse. We could not trace the other accuracies or any of the 44 precision, sensitivity, specificity and F1 values to corresponding results. Reference 38 does contain 99.05%, but for Ma et al. on Senz3D, not a Transformer row. These checks neither prove copying nor establish that the baselines were rerun under a shared protocol.

## Setup

- **Data:** 58,000 ASL Alphabet images, 2,000 per class. A seeded, stratified image split gives 36,540 training, 4,060 validation and 17,400 test images. No exact image duplicates cross the splits.
- **Input:** Wiener filtering, resize to 224×224 and ImageNet normalization. No random augmentation.
- **Model:** ImageNet-pretrained ResNeXt101, VGG19 and ViT-B/16 features feed a BiGRU classifier. The two CNNs stay frozen; ViT is fine-tuned after five warm-up epochs.
- **Search and training:** 20 CS-IGWO trials, each training the classifier for ten epochs on cached features. The selected configuration is trained afresh for 50 epochs, batch size five, using SGD and seed 42. Selected classifier learning rate: 0.002386; fixed ViT rate: 0.001.

The paper does not specify the original split, feature-fusion layout or search budget. These choices were agreed for this reconstruction. The image-level split also does not establish performance on unseen signers, and exact-duplicate checks do not rule out near-duplicate images. The high scores therefore do not demonstrate an improvement over the published method.

## Decisions and assumptions

This table records the choices used in the completed run. Some fill gaps; others resolve conflicting text or equations. The last two rows describe practical substitutions. Earlier choices that were later replaced are noted below the table.

| Detail | Missing or unclear in the paper | Choice used here |
| --- | --- | --- |
| Dataset size and selection | 2,000 images × 29 classes implies 58,000, but Table 1 prints 30,000; no subset list. | 2,000 images per class, sampled without replacement from sorted filenames. |
| Train/validation/test split | Reports 70/30 train/test; no split files or validation partition. | Stratified split; reserve 10% of the training portion for validation: 36,540 / 4,060 / 17,400. |
| Randomness | Seeds and shuffle policy unspecified. | Seed 42; NumPy PCG64 for sampling/search; deterministic PyTorch; seeded shuffle each epoch. |
| Wiener filter | Window, channel handling and noise estimate unspecified. | 3×3 per RGB channel; estimate noise from mean local variance. Stop on nonfinite output. |
| Input transforms | No complete resize/normalization recipe. | Filter at original size, clip, resize to 224×224 with antialiased bilinear interpolation, scale to [0,1], then ImageNet normalization; no crop. |
| Augmentation | No explicit policy found. | No random augmentation. |
| Backbone variants and weights | Names ResNeXt101, VGG19 and ViT without exact variants/checkpoints. | Torchvision ResNeXt101-32x8d, VGG19 without batch normalization and ViT-B/16; ImageNet-1K V1 weights. |
| Feature endpoints | Extraction layers unspecified. | ResNeXt layer4, VGG final pooling and ViT final normalized patch tokens; omit classification heads and the ViT class token. |
| Feature fusion and sequence | No fusion operator or sequence definition for static images. | Pool ViT to 7×7; concatenate spatially into 49 row-major vectors of 3,328 features. No projection or sequences across images. |
| BiGRU and classifier | Width, depth, output aggregation and dropout placement unspecified; generic ReLU setting is ambiguous. | One layer, 256 units/direction; concatenate final states, dropout 0.5, linear 29-class head. Standard sigmoid/tanh gates; no added ReLU. |
| Final training duration | Settings say 50 epochs; curve discussion says 0–25. | Complete 50 epochs. |
| Backbone fine-tuning | ViT text mentions gradual unfreezing; no schedule or equivalent CNN recipe. | CNNs always frozen; ViT frozen for five epochs, then fine-tuned for 45. |
| Module modes | Freezing and normalization-statistic handling unspecified. | Frozen backbones stay in evaluation mode; trainable modules use training mode. All modules use evaluation mode for validation/test. |
| Gradient optimizer | Gives learning rate 0.01, but no complete optimizer recipe. | SGD, momentum 0.9, zero weight decay, no learning-rate decay. |
| Loss | Exact classification loss unspecified. | Equal-weight multiclass cross-entropy, no label smoothing. |
| Search parameters and bounds | Does not identify the tuned parameters or ranges. | Search classifier learning rate in [0.001, 0.01]; keep final ViT rate fixed at 0.001. |
| Search coordinates and bounds | Representation and out-of-range handling unspecified. | One normalized coordinate x in [0,1], decoded as 10^(−3+x); clip final proposals. |
| Initial population | Requires random initialization but omits distribution and seed. | First agent at learning rate 0.01; four independent log-uniform draws using a separate seed-42 RNG. |
| Search budget | Population and iteration parameters have no numerical values. | Five agents; three iterations after initialization; at most 20 evaluations, including failures. |
| Search schedule | Numerical iteration limit absent; relation to evaluation count unclear. | Use t=0,1,2 and t/3 in the printed schedules; keep the evaluation ceiling separate. |
| Candidate training | Trial duration, resets and backbone updates unspecified. | Ten classifier-only epochs on cached fixed features, with identical seeded initialization for every trial. |
| Search fitness and ties | Defines classification error but not its partition, checkpoint or tie rule. | Validation error at candidate epoch 10; earlier-evaluated candidate wins ties. |
| Leader selection and refresh | Algorithm gives an update order but leaves implementation details open. | Use the best three evaluated candidates; hold leaders fixed while proposing a whole generation, then evaluate and refresh. |
| GWO proposals | A, C, distances and leader-derived positions are incompletely defined. | Fill these from original GWO, with independent draws per leader/agent/coordinate; retain the paper's nonlinear a schedule. |
| Leader weights, Eq. 6 | Error-proportional weights conflict with prose saying the best leader gets the greatest weight. | Keep the printed weights; use equal thirds if all three errors are zero. |
| Weighted update, Eq. 7 | Divides by three despite weights already summing to one. | Keep the extra division literally. |
| Mutation, Eqs. 8–11 | Repeated symbols obscure candidate identity and random draws. | Mutate each leader-derived position separately with Gaussian/Cauchy noise and the printed mixing coefficients; average the three results. |
| Greedy selection, Eq. 12 | Comparison and outcomes use indistinguishable symbols. | Use strict-improvement selection only when both exact candidates already have cached fitness; keep the pre-mutation candidate on ties. Otherwise keep the mutated output, without extra evaluations. |
| Crow starting position | X_N(t+1) is not clearly defined. | Use the preceding mutation/selection output; add no intermediate fitness evaluation. |
| Crow flight length and draws | Flight length and random-draw granularity unspecified. | Flight length 1; one uniform movement scalar per agent. |
| Crow branch probability | Algorithm P and equation AP use different names; printed AP can become negative. | Treat P as AP; use raw Eq. 15 without clipping. A separate draw selects Eq. 13 when below AP, otherwise Eq. 14. |
| Final initialization and checkpoint | Unclear whether to continue the winning trial or how to select the final checkpoint. | Train afresh; select lowest validation error over all 50 epochs, earliest on ties. Test only after selection. |
| Reporting metrics | Per-class results and averages lack an exact implementation. | One-vs-rest metrics, unweighted mean over 29 classes; also report ordinary accuracy. Set undefined ratios to zero and record them. |
| Precision and software | Python 3.6.5 reported; full dependency and precision settings unavailable. | Pinned PyTorch/CUDA image; float32, no mixed precision or TF32. |
| Hardware | Reports i5-8600K, GTX 1050 Ti 4 GB and 16 GB RAM. | Agreed substitute: one T4, six CPU cores and 16 GiB RAM. Runtime is not directly comparable. |

We initially planned to fine-tune all three backbones in every trial and search both classifier rate and backbone-rate multiplier. After reviewing the cost and paper wording, we changed to cached classifier-only search and ViT-only final fine-tuning. The full decision history remains in [reproduction.json](reproduction.json).

## Execution and evidence

The run used one T4, six CPU cores and 16 GiB RAM on Modal. It finished in about 49 hours, including interruptions, and resumed from checkpoints after preemptions. Conservative budget accounting corresponds to roughly **USD 51**, excluding storage and taxes; actual billed cost was not retrieved.

The confusion matrix and all reported metrics were independently checked. Eleven local tests and the reproduction-record validator passed. [reproduction.json](reproduction.json) contains the full protocol, run history, source hashes and artifact locations. [Completed cloud run](https://modal.com/apps/repro-sign/main/ap-XD6f11GQndHKMZo0M6JP99).

Implementation, execution and reporting: GPT-6 using Codex. Exact model and application versions were unavailable.

## How to run

The entry point is [scripts/modal_app.py](scripts/modal_app.py). It uses the existing ASL Alphabet data in the `repro-sign` workspace. Full runs use `--mode cloud`, a fresh run name, an approved GPU-hour budget and a future UTC deadline. The executed commands are recorded in `reproduction.json`.

To check the report locally:

```bash
python3 .agents/skills/reproduce-paper/scripts/validate_reproduction.py papers/alabduallah-2025-islrhphmoadl
```
