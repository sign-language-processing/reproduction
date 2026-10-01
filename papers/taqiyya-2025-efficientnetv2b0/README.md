# Taqiyya et al. (2025): EfficientNetV2B0 ASL classification

**Paper ID:** `c56bc98c78421f35c08f8cdf9657b22ce66dfc77`

Thifa Ziada Taqiyya, Khadijah, and Retno Kusumaningrum. *Classification of American Sign Language Using EfficientNetV2B0 Architecture.* ICITACEE 2025, pp. 196–202. [Paper](https://doi.org/10.1109/ICITACEE66165.2025.11233171).

This attempt reconstructs all four accuracy rows in Table X. The tracker export names Table X without limiting the model rows. These are the paper's own comparison experiments, not copied baselines. The Table X parameter-count column is used to verify architecture identity, rather than as a separate requested metric.

## Assignment and agent

The current tracker export is a reproduction object containing `expand.paper`, not the older confirmed-candidate array. Its paper record is `final`; no `confirmation` property exists, and its database ID differs from the immutable paper ID. The direct user-authorized batch assignment is preserved honestly in `reproduction.json`, alongside the redacted expanded record and SHA-256 of the original export. No queue confirmation was invented.

The `reproduction-agent` is GPT-6 using Codex and performs source discovery, implementation, execution and reporting. This attribution is exposed by the executing session; its exact model variant/build and Codex version are unavailable. Every run records that agent ID. A second GPT-6/Codex agent performed independent read-only paper/code/runtime review, identified an augmentation-order discrepancy and timeout mismatch, and reviewed the corrections. Its exact model and harness versions are likewise unavailable; it did not execute experiments. The master GPT-6/Codex agent independently recomputed terminal test scores from saved predictions, checked unique test indices, probability argmax and split hashes, and reviewed the source-based correction. Its exact model and app versions are unavailable; it is also review-only and absent from execution attribution. Session attestations support these identities. Attribution does not identify the operator.

## Resolved recipe and source search

The publisher PDF, references, author institution publication page, first-author public GitHub repositories, exact-title/method searches and the [author's thesis](https://eprints2.undip.ac.id/id/eprint/40526/) supplied no usable author code. The thesis exposes front matter, abstract and introduction, but not implementation chapters. Preference level **3** is therefore used: a small reconstruction using pinned Keras application backbones. There are no upstream patches or copied author configurations.

The thesis abstract resolves that Table X's **0.9957** EfficientNetV2B0 accuracy is the augmented-data model, with learning rate `1e-5`, batch size `16`, dropout `0.2` and an additional dense layer. Each Table X parameter count exactly equals a standard Keras backbone without its classifier, global average pooling, dense(256), dropout, and dense(27):

| Model | Table X accuracy | Parameters |
|---|---:|---:|
| EfficientNetV2B0 | 0.9957 | 6,254,187 |
| MobileNetV2 | 0.9906 | 2,592,859 |
| ResNet50V2 | 0.9932 | 24,096,283 |
| ConvNeXt-Tiny | 0.9940 | 28,023,931 |

All four architectures are trained with the same selected configuration. This is an explicit reconstruction of the intended comparison; the paper does not repeat the baseline training configuration. Tables IV–IX's 48 hyperparameter-search experiments are outside the requested Table X scope.

Accuracy is the number of correct top-1 predictions divided by all test examples. Predictions, true labels, indices, logits/probabilities and raw metric records are retained externally. Numerical agreement is assessed at Table X's four-decimal reporting precision; it is separate from pipeline completion and does not establish scientific success or failure.

## Data and permissions

[27 Class Sign Language Dataset](https://www.kaggle.com/datasets/ardamavi/27-class-sign-language-dataset), Kaggle version 1, contains `X.npy` of shape `(22801,128,128,3)` and float32 values in `[0,1]`, and `Y.npy` of shape `(22801,1)`. There are 27 classes; NULL has 314 examples. The arrays are used as stored without a channel-order conversion.

The exact release was absent from the shared Volume and was downloaded through a controlled CPU population step into `/datasets/mavi-27-class`. The [dataset paper](https://arxiv.org/abs/2203.03859v1) states IRB approval and consent from 173 volunteers, explains resizing to reduce identifying hand detail, and permits research reuse. The Kaggle card declares CC BY-NC-SA 4.0 while its prose also allows research/commercial use with citation. This attempt uses the common permitted case: noncommercial research, with attribution and no image redistribution. Processing in the project's cloud storage is within that research use. No new participants or human evaluation were involved.

| File | SHA-256 |
|---|---|
| Original version-1 ZIP | `1f5bee3ee04209e2337d1cd7eeab73a0cc47809621558f4c14a7bf725a819704` |
| X.npy | `20871064f141869153f198d0067116dcc98df76cc74041ecfab165f3ee320eae` |
| Y.npy | `3ec56c06e74c42663e060e514960c8cbc855524d1832f8c1bcdbe40882261554` |

The source archive, arrays and manifest remain on the `datasets` Volume; no samples are committed. The paper gives 23,352 augmented examples, exactly 551 more than the original set. Following its stated aim of balancing NULL, 551 randomly transformed NULL examples bring that class to 865, within the other classes' 863–866 range. Figure 1 and Section II.B specify bilinear resizing from 128×128 to 224×224 before augmentation. Transformations then follow section II.B: rotations up to 20°, width/height shifts 0.2, zoom 0.2, and horizontal flipping. Keras ImageDataGenerator defaults supply interpolation/fill semantics not specified by the paper.

Stratified splitting uses the exact Table III augmented counts: **18,681 train / 2,335 validation / 2,336 test**. Seed 42 is declared because the historical seed/membership is unavailable. Augmentation precedes splitting as the paper's methodology and changing test counts indicate. Thus a transformed NULL image and its source can fall in different splits; we preserve this property rather than silently substitute a different protocol. Every run preserves its split indices.

## Assumptions and interpretation

The paper omits the optimizer, loss, hidden activation, framework version, freezing policy, initialization dataset, early-stopping monitor and random seed. The reconstruction uses ImageNet weights, Adam at `1e-5`, categorical cross-entropy, ReLU dense256, softmax27, all layers trainable, maximum validation accuracy checkpoint selection, and patience 5 within a maximum of 50 epochs. Standard backbone preprocessing receives the appropriate input range after bilinear resizing to 224×224. These are declared implementation choices, not claims to have recovered the authors' code.

The random split and augmentation realization are newly sampled, not the authors' exact arrays. The inference that all 551 added images are NULL is supported by the counts and methodology but not by released augmentation metadata. The resulting measurements should be interpreted as attempts under this declared reconstruction. Numerical proximity cannot establish which unpublished implementation choices the authors used.

No author contact occurred. The queue ethics flag was investigated against the dataset's consent, IRB, processing and research-use statements; it did not independently prohibit this reuse.

## Repeating the attempt

From the repository root, after the repository's one-time setup and valid `repro-sign` authentication:

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/taqiyya-2025-efficientnetv2b0/scripts/modal_app.py --mode data
.agents/skills/reproduce-paper/scripts/check_modal_dataset.sh mavi-27-class manifest.json
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/taqiyya-2025-efficientnetv2b0/scripts/modal_app.py --mode preflight --model efficientnetv2b0
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/taqiyya-2025-efficientnetv2b0/scripts/modal_app.py --mode full --model efficientnetv2b0
```

Repeat the last two commands with `mobilenetv2` and `resnet50v2`. For `convnexttiny`, append `--gpu A100` to both commands. Use a separate terminal for each model to run them concurrently. Independent detached invocations run the four models in parallel, each with a single GPU and an isolated output directory. The `--gpu` option selects standard A10G or A100 hardware without changing the model, batch size or precision. Each training invocation saves and evaluates its best checkpoint, so evaluation is included. Existing final metrics make a rerun idempotent. Interrupted runs reload `last.keras`, including optimizer state, and resume from `state.json`; a resumed process resets its shuffle generator, so stochastic ordering across an interruption is not bitwise identical. Preserve output directories to retain checkpoints.

All training mounts `datasets` read-only. `huggingface-cache` is mounted read-write at `/cache/huggingface`, with `HF_HOME` and `HF_HUB_CACHE` set; Keras assets use its `keras/` subdirectory. Outputs are under the separate v2 `taqiyya-2025-efficientnetv2b0-results` Volume. Cached pretrained files are individually hashed, not treated as provenance merely because they were present.

The Dockerfile uses the study's GPU base image, pinned to AMD64 digest `sha256:679f06c1abe55eda74213fe883e24db8802e8aeeac1b6d94619055b08029d2ea`, and an isolated TensorFlow environment. The pins are TensorFlow 2.18.0 with its CUDA dependencies, Keras 3.8.0, NumPy 1.26.4, scikit-learn 1.5.2, SciPy 1.14.1 and Pillow 10.4.0. Resolved dependency freezes and GPU/driver descriptions are retained with every run. The input is static image arrays; no video decoding is introduced.

## Independent review and corrected attempts

The initial EfficientNet preflight passed using remote array access and the framework’s default compilation setting. A second preflight made local hash-verified array staging and disabled JIT compilation explicit; it also passed. The initial MobileNet, ResNet and ConvNeXt preflights passed before the subsequent order review. Their tiny metrics establish execution only. The earliest raw metric’s provisional `conditional` flag predates source resolution and is retained unchanged; it is not a target-comparability decision.

The first full attempts applied the 551 generated NULL transformations at 128×128 before resizing. Independent review of Figure 1 and Section II.B identified the incorrect order while the jobs were still early. All four were stopped, their available checkpoints and logs retained, and their scores excluded from target results. The correction resizes the same sampled NULL source images to 224×224 before applying the same seeded transformations. Split indices, seed, labels and other settings are unchanged; the correction was driven by source evidence, not metric proximity. Fresh `v3` directories prevent reuse of invalid `v2` checkpoints.

Corrected preflights explicitly exercise both original and generated examples in each tiny partition and verify checkpoint reload and an actual resumed optimizer step. Each corrected full run is the second and final allowed attempt. The Modal function and its subprocess enforce per-run ceilings; the subprocess finishes early enough to commit evidence. The early interrupted runs are classified `stopped/invalid_run`, not as target results. Their source script is retained by SHA256 alongside sanitized logs and native artifacts.

The corrected ConvNeXt run uses `--gpu A100`. Its standard-A100 preflight measured 0.755 seconds for a warm tiny epoch versus 1.298 seconds on A10G with identical float32 and batch size 16 settings, a 1.72× throughput improvement. Hardware was selected on timing and cost, not accuracy. The other three corrected models use A10G; each runs as an independent single-GPU experiment.

## Results and resources

**Pipeline status:** complete

**Numerical agreement:** does_not_agree

**Preference level:** 3

| Table X model | Published accuracy | Reproduced accuracy | Difference | Epochs | GPU | Runtime |
|---|---:|---:|---:|---:|---|---:|
| efficientnetv2b0 | 0.9957 | 0.994007 (2322/2336) | -0.001693 | 16 | A10G | 29.3 min |
| mobilenetv2 | 0.9906 | 0.993579 (2321/2336) | +0.002979 | 32 | A10G | 53.9 min |
| resnet50v2 | 0.9932 | 0.992295 (2318/2336) | -0.000905 | 18 | A10G | 43.9 min |
| convnexttiny | 0.9940 | 0.993579 (2321/2336) | -0.000421 | 13 | A100 | 35.7 min |

All four corrected target runs completed. Their saved split artifacts have identical SHA256, and all test denominators are 2,336. Each result comes from the highest-validation-accuracy checkpoint under the declared patience-five, maximum-50-epoch rule; checkpoint reload was verified. Differences are reproduced minus published accuracy. The agreement bound is ±0.00005 in fraction units (0.005 percentage points), chosen before full scores. These are measurements for human scientific review, not claims of scientific success or failure.

The four corrected runs consumed 2.712 measured GPU-hours in parallel, with a planning cost estimate of CHF 7.85; actual provider-billed cost is unavailable. Script runtimes exclude startup. The interrupted incorrect-order jobs used at most 0.752 submission-to-stop GPU-hours, an upper bound including startup and recording delay, and are excluded from the metrics above. Preflight timings, hardware, peak memory, package freezes and all app/function IDs appear in reproduction.json.

| Model | Final Modal app | Function call |
|---|---|---|
| efficientnetv2b0 | [ap-DjyZgxXkErXjkkX1pnK3bT](https://modal.com/apps/repro-sign/main/ap-DjyZgxXkErXjkkX1pnK3bT) | `fc-01M3VJ8A7VXX98DT6XW3GGMPHA` |
| mobilenetv2 | [ap-06KFspvRcHM80FvWds91Lh](https://modal.com/apps/repro-sign/main/ap-06KFspvRcHM80FvWds91Lh) | `fc-01M3VJ8BDKEJNCH2AAEK7WBZ0F` |
| resnet50v2 | [ap-YoUHonLuYX2mC4rZEyqV5W](https://modal.com/apps/repro-sign/main/ap-YoUHonLuYX2mC4rZEyqV5W) | `fc-01M3VJA2JNCG363QAP740NM7BY` |
| convnexttiny | [ap-CmWwWdvHzlrjaW2MLinyoZ](https://modal.com/apps/repro-sign/main/ap-CmWwWdvHzlrjaW2MLinyoZ) | `fc-01M3VJK9H8GCXTB18BNFM5E6NE` |

Every native checkpoint, probability/prediction array, split, model definition, hardware record, freeze and raw metric is retained under the corresponding `modal://taqiyya-2025-efficientnetv2b0-results/` Volume, in the model-specific full-run directories listed in reproduction.json, with its checksum and run linkage in reproduction.json. Sanitized logs and the invalid historical source are retained separately. No checkpoint or dataset is committed or republished.

No remaining question requires human action. Exact historical seed, augmentation draws, optimizer/defaults, freezing and stopping monitor are disclosed reconstruction assumptions; numerical proximity cannot resolve those unpublished choices. No original author was contacted.
