# Kenshimov et al. 2021, Kazakh fingerspelling CNN comparison: reproduction

**Paper ID:** `f912577bf219940db498d1ba9cc73acdd028049d`

**Citation:** Kenshimov, C., Mukhanov, S., Merembayev, T., Yedilkhan, D. (2021). A comparison of convolutional neural networks for Kazakh sign language recognition. Eastern-European Journal of Enterprise Technologies, 5 (2 (113)), 44-54. doi:10.15587/1729-4061.2021.241535

**Paper:** https://doi.org/10.15587/1729-4061.2021.241535 · **Code/artifacts:** none published; found after search: none. The authors supplied the dataset and two notebooks privately.

**Preference level:** 3

**Pipeline status:** `partial`

**Numerical agreement:** `does_not_agree` — the four parameter counts match Table 1; none of the four accuracies or four losses does; the four Penalty values were not produced

**Attempt date:** 2026-11-20 to 2026-11-23

## Reproduction agents

| Agent ID | Model and version | Agent application | Contribution | Attribution evidence / unknowns |
| --- | --- | --- | --- | --- |
| `reproduction-agent` | Fable 5.2 (`claude-fable-5-2`) | Claude Code 2.1.329 | Assignment ingestion, target resolution, source search, data gating, implementation, execution of every run, evaluation and report. | Session metadata on 2026-11-20: the session's system context names the model 'Fable 5.2' with model ID 'claude-fable-5-2'. Session output on 2026-11-20: the harness reported itself as Claude Code, updated to version 2.1.329. |

Every run records `agent_ids: ["reproduction-agent"]`. No other agent contributed.

## Scope and target contract

The queue record asks for **Table 1** (p. 49): four models, each with total parameters, accuracy, loss and penalty, on the paper's own Kazakh fingerspelling dataset. That is 16 numbers, all in `reproduction.json.targets` and all in scope. The paper reports one run per model and no seed.

- **Total params:** model size in millions. Reproduced with Keras `count_params()`.
- **Accuracy and loss:** measured on the 210-image test split after 20 epochs. Fig. 9 confirms that accuracy is correct / total: LeNet's matrix has 208 of 210 on the diagonal, 0.990.
- **Penalty:** a custom score from a 42x42 penalty matrix. The paper prints only a 12x12 excerpt (Fig. 4, 'Example of penalty matrix values'), and its Eq. (2) gives values at or below zero while Table 1 reports about 3.0. It cannot be computed, and the four Penalty targets are closed as not produced (gate `ksl-penalty-metric`).

Three decisions were taken at human gates before any training run, and are recorded in `reproduction.json.gates`:

| Gate | Type | Decision |
| --- | --- | --- |
| `ksl-data-cloud-use` | data | A study team member confirmed that the images may be uploaded to and processed in the repro-sign Modal workspace, and authorized populating /datasets/ksl-fingerspelling-kenshimov-2021 on the shared Volume. The copy stays private to the workspace; no image is redistributed or committed. |
| `ksl-subset-and-split` | protocol | Decided by a study team member before any training run. Subset: the first 50 files of each class folder in lexicographic file-name order. Split: random 80/10/10, not stratified, as in Read_data.ipynb, with seed 42 for both train_test_split calls and for weight initialization. Seeds 1, 2, 3 and 4 are run as supplementary evidence only and never replace the seed-42 values. Values obtained under these choices count as produced targets. |
| `ksl-penalty-metric` | target | A study team member chose alternative 'not-produced': the four Penalty targets are closed as not produced, reason metric_ambiguous, without contacting the authors for now. |

## Source provenance

| Artifact | Canonical source | Pinned revision / SHA-256 | Role |
| --- | --- | --- | --- |
| `paper-pdf` | https://journals.uran.ua/eejet/article/download/241535/241471 | `3d62ae2fc8fecea3055cf9e743cb45d200098e8580ea3942f133bcb01e3e7bf5` | Targets (Table 1) and the disclosed protocol. The same file is on Zenodo record 5642519 (MD5 52f12824024eb2bd031497b98cf4265c), which holds only this PDF. |
| `authors-read-data-notebook` | Read_data.ipynb, supplied privately by the authors to the study; not published | `b9d2b118448e2944b0dd9d57839977cfe4debd1444b1893d2d6c589d29482cf0` | The authors' original data loading, with saved outputs: 14,247 images in 42 classes, PIL grayscale resize to 320x120, division by 255, unseeded 80/10/10 train_test_split. |
| `authors-compare-nets-notebook` | Compare_nets.ipynb, supplied privately by the authors to the study; not published | `b0c5aa2b4df995a5631122c85e740418ce174ff4f545eaf925a680e295b75afd` | A later, revised protocol by the authors, not the code behind Table 1 (Adam, early stopping, augmentation, seeded stratified split, EfficientNetB0). Used only for its LeNet builder, whose parameter count equals Table 1's, and for its log of the earlier runs. |
| `authors-dataset-archive` | resized_gray_dataset.7z, supplied privately by the authors to the study; not published | `baf1f1436f2d827b03d88c6b42db783d4e9ca0b84b7bb82fc8ee2f133e494ec2` | The image collection: 14,249 grayscale 1280x720 JPEGs in 42 class folders. |

**Search for published code and data (2026-11-20).** No published code and no published dataset were found. Preference level 3.

- Paper PDF, all 11 pages: no code, data or supplement link; no data-availability statement.
- Zenodo record 5642519 (API): one file, the article PDF, identical to the pinned PDF by MD5.
- Journal landing page journals.uran.ua/eejet/article/view/241535: HTTP 403 to automated access; not inspected.
- Web search for the exact title with 'code dataset github', and for the dataset with 'github OR kaggle OR zenodo' and the folder name 'resized_gray_dataset': no code or dataset release.
- GitHub repository search for 'kazakh sign language', 'kazakh fingerspelling', 'kazakh dactyl' and 'KSL sign language recognition lenet efficientnet': eight repositories, none by the authors.
- GitHub user search by author surname; public repositories of the three matching accounts listed: none concerns this paper.

`Compare_nets.ipynb` is the authors' later, revised protocol and not the code behind Table 1: it uses Adam, early stopping, augmentation, a seeded stratified split and EfficientNetB0. Its opening note logs the earlier runs: LeNet 93.3%, AlexNet 99.0%, ResNet50 64.3%, and no score for EfficientNetB7, which 'stopped during epoch 1'. Table 1 reports 0.990, 0.990, 0.642 and 0.967. This attempt used the notebook only for its LeNet builder and for that log.

## Results

Seed 42, the one pre-declared run per model. Difference is reproduced minus original.

| Target ID | Paper location | System | Dataset/split | Metric + version | Original | Reproduced | Difference | Terminal reason / evidence |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `table1-lenet-total-params` | Table 1 (p. 49), row 'LeNet', column 'Total params (M)' | lenet | not applicable: a property of the model | Total parameters, TensorFlow 2.8.0 | 1.07 | 1.0712 | +0.0012 | produced; `full-lenet-seed42`; Model.count_params() = 1,071,162. |
| `table1-lenet-accuracy` | Table 1 (p. 49), row 'LeNet', column 'Accuracy' | lenet | test: 210 images, 10% of the 2,100-image subset, random and unstratified | Accuracy, TensorFlow 2.8.0 | 0.99 | 1.0000 | +0.0100 | produced; `full-lenet-seed42`; 210 of 210 test images correct after epoch 20, seed 42. |
| `table1-lenet-loss` | Table 1 (p. 49), row 'LeNet', column 'Loss' | lenet | test: 210 images, 10% of the 2,100-image subset, random and unstratified | Loss, TensorFlow 2.8.0 | 0.0453 | 0.0038 | -0.0415 | produced; `full-lenet-seed42`; Categorical cross-entropy on the 210 test images after epoch 20, seed 42. |
| `table1-lenet-penalty` | Table 1 (p. 49), row 'LeNet', column 'Penalty' | lenet | test: 210 images, 10% of the 2,100-image subset, random and unstratified | Penalty, none | 3.017 | not produced |  | `metric_ambiguous`: The 42x42 penalty matrix is not published (Fig. 4 shows 12 of 42 classes), and Eq. (2) yields values at or below zero while Table 1 reports about 3.0. The score cannot be computed. |
| `table1-alexnet-total-params` | Table 1 (p. 49), row 'AlexNet', column 'Total params (M)' | alexnet | not applicable: a property of the model | Total parameters, TensorFlow 2.8.0 | 35.5 | 35.4981 | -0.0019 | produced; `full-alexnet-seed42`; Model.count_params() = 35,498,090. |
| `table1-alexnet-accuracy` | Table 1 (p. 49), row 'AlexNet', column 'Accuracy' | alexnet | test: 210 images, 10% of the 2,100-image subset, random and unstratified | Accuracy, TensorFlow 2.8.0 | 0.99 | 0.9143 | -0.0757 | produced; `full-alexnet-seed42`; 192 of 210 test images correct after epoch 20, seed 42. |
| `table1-alexnet-loss` | Table 1 (p. 49), row 'AlexNet', column 'Loss' | alexnet | test: 210 images, 10% of the 2,100-image subset, random and unstratified | Loss, TensorFlow 2.8.0 | 0.1343 | 0.8167 | +0.6824 | produced; `full-alexnet-seed42`; Categorical cross-entropy on the 210 test images after epoch 20, seed 42. |
| `table1-alexnet-penalty` | Table 1 (p. 49), row 'AlexNet', column 'Penalty' | alexnet | test: 210 images, 10% of the 2,100-image subset, random and unstratified | Penalty, none | 3.021 | not produced |  | `metric_ambiguous`: The 42x42 penalty matrix is not published (Fig. 4 shows 12 of 42 classes), and Eq. (2) yields values at or below zero while Table 1 reports about 3.0. The score cannot be computed. |
| `table1-resnet50-total-params` | Table 1 (p. 49), row 'ResNet50', column 'Total params (M)' | resnet50 | not applicable: a property of the model | Total parameters, TensorFlow 2.8.0 | 23.7 | 23.6675 | -0.0325 | produced; `full-resnet50-seed42`; Model.count_params() = 23,667,498. |
| `table1-resnet50-accuracy` | Table 1 (p. 49), row 'ResNet50', column 'Accuracy' | resnet50 | test: 210 images, 10% of the 2,100-image subset, random and unstratified | Accuracy, TensorFlow 2.8.0 | 0.642 | 0.5095 | -0.1325 | produced; `full-resnet50-seed42`; 107 of 210 test images correct after epoch 20, seed 42. |
| `table1-resnet50-loss` | Table 1 (p. 49), row 'ResNet50', column 'Loss' | resnet50 | test: 210 images, 10% of the 2,100-image subset, random and unstratified | Loss, TensorFlow 2.8.0 | 1.8544 | 2.9777 | +1.1233 | produced; `full-resnet50-seed42`; Categorical cross-entropy on the 210 test images after epoch 20, seed 42. |
| `table1-resnet50-penalty` | Table 1 (p. 49), row 'ResNet50', column 'Penalty' | resnet50 | test: 210 images, 10% of the 2,100-image subset, random and unstratified | Penalty, none | 3.13 | not produced |  | `metric_ambiguous`: The 42x42 penalty matrix is not published (Fig. 4 shows 12 of 42 classes), and Eq. (2) yields values at or below zero while Table 1 reports about 3.0. The score cannot be computed. |
| `table1-efficientnetb7-total-params` | Table 1 (p. 49), row 'EffectiveNetB7', column 'Total params (M)' | efficientnetb7 | not applicable: a property of the model | Total parameters, TensorFlow 2.8.0 | 64.2 | 64.2041 | +0.0041 | produced; `full-efficientnetb7-seed42`; Model.count_params() = 64,204,093. |
| `table1-efficientnetb7-accuracy` | Table 1 (p. 49), row 'EffectiveNetB7', column 'Accuracy' | efficientnetb7 | test: 210 images, 10% of the 2,100-image subset, random and unstratified | Accuracy, TensorFlow 2.8.0 | 0.967 | 0.0238 | -0.9432 | produced; `full-efficientnetb7-seed42`; 5 of 210 test images correct after epoch 20, seed 42. |
| `table1-efficientnetb7-loss` | Table 1 (p. 49), row 'EffectiveNetB7', column 'Loss' | efficientnetb7 | test: 210 images, 10% of the 2,100-image subset, random and unstratified | Loss, TensorFlow 2.8.0 | 0.1355 | 12.7797 | +12.6442 | produced; `full-efficientnetb7-seed42`; Categorical cross-entropy on the 210 test images after epoch 20, seed 42. |
| `table1-efficientnetb7-penalty` | Table 1 (p. 49), row 'EffectiveNetB7', column 'Penalty' | efficientnetb7 | test: 210 images, 10% of the 2,100-image subset, random and unstratified | Penalty, none | 3.017 | not produced |  | `metric_ambiguous`: The 42x42 penalty matrix is not published (Fig. 4 shows 12 of 42 classes), and Eq. (2) yields values at or below zero while Table 1 reports about 3.0. The score cannot be computed. |

No score in Table 1 is copied from earlier work; all four rows are the paper's own experiments.

### What the training curves show

| Model (seed 42) | Train accuracy, epoch 20 | Validation accuracy at epochs 1, 5, 10, 15, 20 | Distinct classes predicted on test |
| --- | ---: | --- | ---: |
| lenet | 1.000 | 0.02, 1.00, 1.00, 1.00, 1.00 | 42 of 42 |
| alexnet | 0.987 | 0.03, 0.03, 0.03, 0.15, 0.96 | 42 of 42 |
| resnet50 | 1.000 | 0.01, 0.01, 0.01, 0.11, 0.57 | 31 of 42 |
| efficientnetb7 | 0.966 | 0.03, 0.04, 0.03, 0.03, 0.03 | 2 of 42 |

AlexNet, ResNet50 and EfficientNetB7 fit the training set, but their validation accuracy stays near chance (1/42 = 0.024) for 10 to 15 epochs and then rises late, or not at all for EfficientNetB7. Evaluating after exactly 20 epochs therefore catches these models at an unstable point. The paper says of its own runs that 'AlexNet and ResNet50 were overfitting, as accuracy and loss trends for the validation dataset showed small values'.

### Supplementary seeds (not targets)

Agreed at gate `ksl-subset-and-split`: seeds 1 to 4 repeat the protocol to show how much the unrecoverable split and initialization move the result. They never replace the seed-42 values.

| Model | Seed 42 (target) | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Paper |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lenet accuracy | 1.000 | 0.995 | 0.010 | 0.995 | 1.000 | 0.99 |
| alexnet accuracy | 0.914 | 0.257 | 0.962 | 0.305 | 0.748 | 0.99 |
| resnet50 accuracy | 0.510 | 0.110 | 0.514 | 0.262 | 0.419 | 0.642 |
| efficientnetb7 accuracy | 0.024 | 0.024 | 0.029 | 0.029 | 0.038 | 0.967 |

Test accuracy after 20 epochs depends strongly on the seed. LeNet seed 2 never left chance level (training accuracy 0.02 to 0.03 in every epoch). A single run, which is what the paper reports, is a weak estimate under this protocol.

### Diagnostic: EfficientNetB7 input scaling (not a target)

Keras' EfficientNet rescales pixels by 1/255 internally. The protocol divides by 255 first, as the authors' `Read_data.ipynb` does for all data, so EfficientNetB7 sees near-zero inputs; the authors' later notebook warns about exactly this. One diagnostic run fed 0-255 pixels instead: test accuracy 0.376 (79/210) against 0.024 (5/210) under the protocol. The input scaling matters, but does not bring the model near the published 0.967. The target value is the protocol run's.

Pipeline completeness and numerical agreement are separate axes. This report records the differences and does not judge the paper; that is for scientific review.

## How to repeat this

```bash
# From the repository root, with the Modal CLI on PATH and the repro-sign profile set up.
# 1. Data: needs the authors' archive (SHA-256 baf1f143...), which is not public.
papers/kenshimov-2021-kazakh-sl-cnn/data.sh /path/to/resized_gray_dataset.7z

# 2. One full run per model (Table 1), seed 42:
for model in lenet alexnet resnet50 efficientnetb7; do
  KSL_GPU=A100-40GB .agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run \
    papers/kenshimov-2021-kazakh-sl-cnn/modal_app.py::train \
    --models "$model" --run-id "full-$model-seed42" --seeds 42
done
```

`data.sh` is idempotent: it verifies the archive, uploads it once, and `populate` skips extraction when the manifest exists. Each training run mounts `/datasets` read-only, trains 20 epochs, saves and reloads the final weights, and writes `metrics.json` (test accuracy, loss, confusion matrix, test file list, history) to the Volume `ksl-f912577b-results` under the run ID. The run prints a `RESULT` line with the same numbers. Runs are short (under 15 minutes), so there is no mid-run resume; a failed run is repeated.

## Data provenance and permissions

| Dataset | Version/subset/splits | Source and access date | License/permission and cloud-use basis | Path in Volume `datasets` | Counts / manifest / checksum | Deviations |
| --- | --- | --- | --- | --- | --- | --- |
| Kazakh Sign Language fingerspelling images (Kenshimov et al. 2021), authors' archive resized_gray_dataset.7z | Archive SHA-256 baf1f1436f2d827b03d88c6b42db783d4e9ca0b84b7bb82fc8ee2f133e494ec2: 14,249 grayscale 1280x720 JPEGs, 42 classes. Experiments use the first 50 files per class in lexicographic file-name order (2,100 images). Split: Random and unstratified: sklearn train_test_split(test_size=0.2) then train_test_split(test_size=0.5) on the remainder, both with the run's seed. Each run's metrics.json lists its test files. | Supplied privately by the authors to the study on request; not published. Queue dataset record lc1a3wrizj82i1e. Received 2026-11-20. | No license. The authors granted the study permission to reproduce (queue record: permission_to_reproduce 'yes') and supplied the files. Gate ksl-data-cloud-use records the confirmation that this covers the study's Modal workspace. | `ksl-fingerspelling-kenshimov-2021` | 14,249 images, 42 classes, 142 to 446 per class; `MANIFEST.sha256` SHA-256 `7df5ce57ceb636d9f431fd19e06f531073d550b066e95445a1f163f019fd5857`; one file (`H/17.jpg`) fails PIL verification and lies outside the 2,100-image subset | `subset-first-50-per-class`, `split-seed-42` |

The dataset is not public and is not redistributed: nothing from it is committed, and the copy on Modal is private to the study workspace. The images are 1280x720 grayscale frames, numbered consecutively per class, so neighbouring files are near-duplicates and a random split places near-identical frames in train and test.

## Environment and patches

- **Image:** `nvidia/cuda@sha256:081a31f56decc6c460df1808ed1c4867fb30b0fbfa8929258b10e3e5d6dc1a2e` (tag 11.2.2-cudnn8-runtime-ubuntu20.04 when pinned) with Python 3.10, built by Modal from `modal_app.py`. No Dockerfile: the image is fully defined there.
- **Dependencies:** `tensorflow==2.8.0`, `keras==2.8.0`, `numpy==1.23.5`, `protobuf==3.20.3`, `scikit-learn==1.3.2`, `pillow==10.4.0`, `h5py==3.10.0`. Each run stores its resolved `pip freeze` (SHA-256 `8e022257e7c316b88e49aa0a8d80189e4d21629a2af96fc37e5fc3c5d4f25a0b`).
- **Hardware:** NVIDIA A100-SXM4-40GB for all full, supplementary and diagnostic runs; a Tesla T4 for the first preflight.
- **Shared cache:** every Modal function mounts `huggingface-cache` at `/cache/huggingface` with `HF_HOME=/cache/huggingface` and `HF_HUB_CACHE=/cache/huggingface/hub`. No Hub artifact is used.
- **Code:** `modal_app.py` was committed before every run; each run records the SHA-256 of the file it executed. The full runs used `4716b2cca5fa02a7d9b106224da9c70b46ddb18cc3a132039e49765ce4a0e62c`, the supplementary and diagnostic runs `d238876cf61f0c703c9b90dfeefa66da731ed54086e35e2c9de80fd00cc0237e` (adds the diagnostic's pixel-scale option only).

There is no upstream code, so there are no patches.

## Execution evidence

| Run ID / agent IDs | Attempt / max | Kind / targets | Platform / hardware | Seed/config | Start/end | Exit / terminal state / reason | Failure class | Stop ceilings | Usage (billed) | Logs/artifacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `populate-dataset-1` / `reproduction-agent` | 1 / 2 | data_population | local, CPU |  | 2026-11-20T16:19:13Z to 2026-11-23T10:20:30Z | 144 / `interrupted` / `external_interruption` | `transient_infrastructure` | 3600 s, USD 2.0 | none | log only |
| `populate-dataset-2` / `reproduction-agent` | 2 / 2 | data_population | modal, app `ap-rV6QE9aNOR2CzzF9eg5BJi`, CPU |  | 2026-11-23T10:26:53Z to 2026-11-23T10:31:22Z | 0 / `succeeded` / `completed` |  | 3600 s, USD 2.0 | USD 0.0109 (CPU) | log only |
| `preflight-1` / `reproduction-agent` | 1 / 3 | preflight | modal, app `ap-ZPl1Wi4yy2IzhZo19TJHYy`, Tesla T4, 15360 MiB, driver 620.70.06 |  | 2026-11-23T10:32:12Z to 2026-11-23T10:34:38Z | 1 / `failed` / `command_failed` | `oom_resource` | 3600 s, 1.0 GPU-h, USD 2.0 | 0.0386 GPU-h, USD 0.035 | log only |
| `preflight-2` / `reproduction-agent` | 2 / 3 | preflight | modal, app `ap-H78CJWswYM4oF9rT8cjAFj`, NVIDIA A100-SXM4-40GB, 40960 MiB, driver 610.57.04 |  | 2026-11-23T10:35:32Z to 2026-11-23T10:37:15Z | 0 / `succeeded` / `completed` |  | 3600 s, 1.0 GPU-h, USD 4.0 | 0.027 GPU-h, USD 0.0654 | log only |
| `full-lenet-seed42` / `reproduction-agent` | 1 / 3 | full | modal, app `ap-VhUPUamp1nWvu0zkXH0YUY`, NVIDIA A100-SXM4-40GB, 610.57.04, 40960 MiB | 42 | 2026-11-23T10:38:09Z to 2026-11-23T10:38:57Z | 0 / `succeeded` / `completed` |  | 3600 s, 1.0 GPU-h, USD 5.0 | 0.0115 GPU-h, USD 0.0277 | `full-lenet-seed42-metrics`, `full-lenet-seed42-pip-freeze` |
| `full-alexnet-seed42` / `reproduction-agent` | 1 / 3 | full | modal, app `ap-0NMbj7dEz11AhvqaYuZdiR`, NVIDIA A100-SXM4-40GB, 610.57.04, 40960 MiB | 42 | 2026-11-23T10:38:09Z to 2026-11-23T10:39:03Z | 0 / `succeeded` / `completed` |  | 3600 s, 1.0 GPU-h, USD 5.0 | 0.013 GPU-h, USD 0.0313 | `full-alexnet-seed42-metrics`, `full-alexnet-seed42-pip-freeze` |
| `full-resnet50-seed42` / `reproduction-agent` | 1 / 3 | full | modal, app `ap-fRoSxMpHiLhqIIm5J0Hw82`, NVIDIA A100-SXM4-40GB, 620.70.06, 40960 MiB | 42 | 2026-11-23T10:38:09Z to 2026-11-23T10:40:02Z | 0 / `succeeded` / `completed` |  | 3600 s, 1.0 GPU-h, USD 5.0 | 0.0286 GPU-h, USD 0.0691 | `full-resnet50-seed42-metrics`, `full-resnet50-seed42-pip-freeze` |
| `full-efficientnetb7-seed42` / `reproduction-agent` | 1 / 3 | full | modal, app `ap-vAFZf91UpTGsYaKLbIB6NM`, NVIDIA A100-SXM4-40GB, 620.70.06, 40960 MiB | 42 | 2026-11-23T10:38:09Z to 2026-11-23T10:50:40Z | 0 / `succeeded` / `completed` |  | 3600 s, 1.0 GPU-h, USD 5.0 | 0.2065 GPU-h, USD 0.4991 | `full-efficientnetb7-seed42-metrics`, `full-efficientnetb7-seed42-pip-freeze` |
| `supplementary-lenet-seeds1-4` / `reproduction-agent` | 1 / 3 | supplementary | modal, app `ap-tDExmdQxcQkz8AQN7Y5FKw`, NVIDIA A100-SXM4-40GB, 610.57.04, 40960 MiB | 1, 2, 3, 4 | 2026-11-23T10:52:38Z to 2026-11-23T10:53:50Z | 0 / `succeeded` / `completed` |  | 3600 s, 1.0 GPU-h, USD 5.0 | 0.0178 GPU-h, USD 0.0431 | `supplementary-lenet-seeds1-4-lenet-seed1-metrics`, `supplementary-lenet-seeds1-4-lenet-seed2-metrics`, `supplementary-lenet-seeds1-4-lenet-seed3-metrics`, `supplementary-lenet-seeds1-4-lenet-seed4-metrics`, `supplementary-lenet-seeds1-4-pip-freeze` |
| `supplementary-alexnet-seeds1-4` / `reproduction-agent` | 1 / 3 | supplementary | modal, app `ap-CDCTmEatOTYW0oeCXLTtHt`, NVIDIA A100-SXM4-40GB, 610.57.04, 40960 MiB | 1, 2, 3, 4 | 2026-11-23T10:52:38Z to 2026-11-23T10:54:17Z | 0 / `succeeded` / `completed` |  | 3600 s, 1.0 GPU-h, USD 5.0 | 0.0253 GPU-h, USD 0.0611 | `supplementary-alexnet-seeds1-4-alexnet-seed1-metrics`, `supplementary-alexnet-seeds1-4-alexnet-seed2-metrics`, `supplementary-alexnet-seeds1-4-alexnet-seed3-metrics`, `supplementary-alexnet-seeds1-4-alexnet-seed4-metrics`, `supplementary-alexnet-seeds1-4-pip-freeze` |
| `supplementary-resnet50-seeds1-4` / `reproduction-agent` | 1 / 3 | supplementary | modal, app `ap-cR9glDP6xWj2rDNBoyRfzb`, NVIDIA A100-SXM4-40GB, 620.70.06, 40960 MiB | 1, 2, 3, 4 | 2026-11-23T10:52:38Z to 2026-11-23T10:57:44Z | 0 / `succeeded` / `completed` |  | 3600 s, 1.0 GPU-h, USD 5.0 | 0.0826 GPU-h, USD 0.1997 | `supplementary-resnet50-seeds1-4-resnet50-seed1-metrics`, `supplementary-resnet50-seeds1-4-resnet50-seed2-metrics`, `supplementary-resnet50-seeds1-4-resnet50-seed3-metrics`, `supplementary-resnet50-seeds1-4-resnet50-seed4-metrics`, `supplementary-resnet50-seeds1-4-pip-freeze` |
| `supplementary-efficientnetb7-seeds1-4` / `reproduction-agent` | 1 / 3 | supplementary | modal, app `ap-JTLpqrg5Mt1Po9vlwKUqqw`, NVIDIA A100-SXM4-40GB, 620.70.06, 40960 MiB | 1, 2, 3, 4 | 2026-11-23T10:52:38Z to 2026-11-23T11:39:43Z | 0 / `succeeded` / `completed` |  | 7200 s, 2.0 GPU-h, USD 8.0 | 0.7823 GPU-h, USD 1.8909 | `supplementary-efficientnetb7-seeds1-4-efficientnetb7-seed1-metrics`, `supplementary-efficientnetb7-seeds1-4-efficientnetb7-seed2-metrics`, `supplementary-efficientnetb7-seeds1-4-efficientnetb7-seed3-metrics`, `supplementary-efficientnetb7-seeds1-4-efficientnetb7-seed4-metrics`, `supplementary-efficientnetb7-seeds1-4-pip-freeze` |
| `diagnose-efficientnetb7-raw-pixels-seed42` / `reproduction-agent` | 1 / 3 | diagnostic | modal, app `ap-3RWuDMvphy2OpNAl22XLyL`, NVIDIA A100-SXM4-40GB, 610.57.04, 40960 MiB | 42 | 2026-11-23T10:52:38Z to 2026-11-23T11:04:53Z | 0 / `succeeded` / `completed` |  | 3600 s, 1.0 GPU-h, USD 5.0 | 0.2016 GPU-h, USD 0.4872 | `diagnose-efficientnetb7-raw-pixels-seed42-metrics`, `diagnose-efficientnetb7-raw-pixels-seed42-pip-freeze` |

All Modal runs used profile `repro-sign`. Totals from Modal billing: **1.43 GPU-hours** and **USD 3.42**. Metrics artifacts are `modal://ksl-f912577b-results/RUN_ID/MODEL-seedN/metrics.json`, with SHA-256 values in `reproduction.json.artifacts`.

## Guesses and deviations

| ID | Kind | Description | Paper says | This attempt used | Basis | Effect | Scope | Status | Evidence | Related IDs | Affects | Dataset IDs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `subset-first-50-per-class` | `guess` | Which 2,100 of the archive's 14,249 images the paper used. | Sec. 5.1: '2,100 images for 42 classes of gestures, 50 images for each class'. The selection is not described. | The first 50 files of each class folder in lexicographic file-name order. | Compare_nets.ipynb says the original took 'the first 50 names from os.listdir'. Decided at gate ksl-subset-and-split. | The images are consecutive frames, so the subset covers a short stretch of each class's recording. A different selection could change every accuracy and loss value. |  | adopted before the runs |  |  | `data` | `ksl-fingerspelling` |
| `split-seed-42` | `guess` | The random 80/10/10 split and the weight initialization seed. | Sec. 5.1: 'randomly split' into 1,680 / 210 / 210. Read_data.ipynb calls train_test_split twice without a seed. | Unstratified train_test_split(test_size=0.2) then train_test_split(test_size=0.5), both with random_state 42; tf.keras.utils.set_random_seed(42). | The authors' split cannot be recovered. Seed 42 was fixed at gate ksl-subset-and-split before any run. | The 210 test images differ from the authors'. Seeds 1 to 4 show the spread: accuracy ranges from 0.010 to 1.000 for LeNet, 0.257 to 0.962 for AlexNet, 0.110 to 0.514 for ResNet50. |  | adopted before the runs |  |  | `data`, `train`, `evaluate` | `ksl-fingerspelling` |
| `input-120x320-gray-unit-scale` | `guess` | Input size, color and pixel scaling. | Not stated. The paper's LeNet description mentions 32x32 inputs in general terms only. | PIL convert('L'), resize((320, 120)) with Pillow 10.4.0's default resampling, float32, divided by 255; shape (120, 320, 1). | Read_data.ipynb, the authors' original data loading. All four of Table 1's parameter counts match this input shape. | Defines every model's input. For EfficientNetB7 see efficientnet-double-rescaling. |  | adopted before the runs |  |  | `data`, `train`, `evaluate` |  |
| `lenet-architecture` | `guess` | The LeNet variant. | Sec. 4 describes the classic seven-layer LeNet on 32x32 inputs; it does not give the layers used. | Conv 6@5x5 tanh, average pool 3/2, conv 16@5x5 same tanh, average pool 2/2, conv 16@5x5 same tanh, max pool 2/2, dense 120, dense 84, softmax 42. | The LeNet builder in Compare_nets.ipynb, without its later augmentation layers. It has 1,071,162 parameters on this input, equal to Table 1's 1.07 M. | Determines the LeNet row. |  | adopted before the runs | preflight-1 and full-lenet-seed42: count_params() = 1,071,162. |  | `train` |  |
| `alexnet-architecture` | `guess` | The AlexNet variant; neither the paper nor the notebooks give its layers. | Sec. 4 describes AlexNet in general terms. | Conv 96@11x11 stride 4, BN, max pool 3/2, conv 256@5x5 same, BN, max pool 3/2, conv 384@3x3 same, BN, conv 384@1x1, BN, conv 256@1x1, BN, max pool 3/2, flatten, dense 4096, dropout 0.5, dense 4096, dropout 0.5, softmax 42. | A widely copied Keras AlexNet. Compare_nets.ipynb says 'The 2022 version used 1x1 filters' in the last convolutional block. On this input it has 35,498,090 parameters, equal to Table 1's 35.5 M. | Determines the AlexNet row. Another architecture with the same count cannot be excluded; the dropout rate has no parameters and is unverified. |  | adopted before the runs | preflight-1 and full-alexnet-seed42: count_params() = 35,498,090. |  | `train` |  |
| `resnet50-efficientnetb7-keras-applications` | `guess` | ResNet50 and EfficientNetB7 implementation, head and initialization. | Named only; no head, pretraining or library is stated. | tf.keras.applications.ResNet50 and EfficientNetB7 with weights=None, input_shape (120, 320, 1), classes=42 (global average pooling and a softmax layer). | Parameter counts 23,667,498 and 64,204,093 equal Table 1's 23.7 M and 64.2 M. Compare_nets.ipynb's log says ResNet50 'Finished from scratch with RMSprop'. ImageNet weights need three channels, and the original loading is single-channel. | Determines both rows. If the authors used ImageNet weights for EfficientNetB7, this attempt's from-scratch training differs materially. |  | adopted before the runs | preflight-1: resnet50 count_params() = 23,667,498. preflight-2: efficientnetb7 count_params() = 64,204,093. |  | `train` |  |
| `efficientnet-double-rescaling` | `guess` | EfficientNetB7 receives pixels divided by 255, although Keras' EfficientNet rescales by 1/255 internally. | Not stated. | The same divided-by-255 input as the other models, so EfficientNetB7 effectively sees values in [0, 1/255]. | Read_data.ipynb divides all data by 255 before any model. Compare_nets.ipynb, written later, warns that 'dividing first would feed it near-zero inputs', which suggests the original runs did so. | Material. With this input EfficientNetB7 reaches 0.966 training accuracy but chance-level test accuracy. The diagnostic run with 0-255 inputs reaches 0.376 test accuracy. |  | adopted before the runs | full-efficientnetb7-seed42: 5/210 correct, 2 distinct predicted classes. diagnose-efficientnetb7-raw-pixels-seed42: 79/210 correct. |  | `train`, `evaluate` |  |
| `rmsprop-defaults` | `guess` | RMSprop hyperparameters. | Sec. 5.2: 'root mean square propagation (rmsprop) optimizer'. No learning rate or other setting. | Keras optimizer='rmsprop' defaults in TensorFlow 2.8.0: learning rate 0.001, rho 0.9, epsilon 1e-7, no momentum. | The default of the library the authors used. | Affects all accuracy and loss values. |  | adopted before the runs |  |  | `train` |  |
| `batch-size-64` | `guess` | Batch size. | Not stated. | 64 for all four models. | Compare_nets.ipynb's log of the original runs gives 'batch 64' for LeNet and AlexNet. No value is recorded for ResNet50 or EfficientNetB7. | EfficientNetB7 at batch 64 needed 31.8 to 34.3 GB of GPU memory. The paper's RTX 2080 has 8 GB, so the authors' EfficientNetB7 batch size, input or model must have differed from this attempt's. |  | adopted before the runs | preflight-1: ResourceExhaustedError on a 15 GB T4. full-efficientnetb7-seed42: peak GPU memory 34.31 GB. |  | `train` |  |
| `final-epoch-evaluation` | `guess` | Which weights are evaluated. | Sec. 5.2: 'The training process chose 20 epochs'. No checkpoint selection or early stopping is described. | The weights after the 20th epoch, evaluated once on the test split. | The paper describes no selection. | Validation accuracy of AlexNet, ResNet50 and EfficientNetB7 is still unstable at epoch 20, so the reported value depends on where training stops. |  | adopted before the runs |  |  | `evaluate` |  |
| `software-tf-2-8` | `guess` | Framework versions. | Not stated. Read_data.ipynb's metadata shows Python 3.7.3 and standalone Keras ('Using TensorFlow backend'). | TensorFlow 2.8.0 and Keras 2.8.0 on Python 3.10, CUDA 11.2, cuDNN 8. | The oldest TensorFlow that this study has run on Modal's Python and GPUs; the authors' exact versions are unknown. | Initializers, RMSprop and the Keras application models may differ in detail from the authors' versions. |  | adopted before the runs |  |  | `train`, `evaluate` |  |
| `hardware-a100` | `paper_deviation` | GPU. | Sec. 4: 'NVIDIA RTX 2080 8Gb'. | NVIDIA A100-SXM4-40GB on Modal for every full, supplementary and diagnostic run. | EfficientNetB7 at batch 64 does not fit in 15 GB; one GPU type was used for all models. | None expected on the metrics beyond GPU nondeterminism. |  | adopted after preflight-1 failed, before the full runs |  |  | `train` |  |
| `lenet-without-batch-normalization` | `paper_code_conflict` | Batch normalization in LeNet. | Sec. 5.2: 'The batch normalization ... are applied for all the networks'. The authors' LeNet builder in Compare_nets.ipynb has no batch normalization. | The authors' builder, without batch normalization. | The builder reproduces Table 1's parameter count. The notebook is the authors' code but was supplied privately, not released. | Adding batch normalization would add 152 parameters, which Table 1's rounding cannot show, so the paper's actual LeNet remains uncertain. |  | adopted before the runs |  |  | `train` |  |
| `dataset-size-statements` | `paper_internal_conflict` | The dataset size. | Abstract: 'more than 10122 gesture samples'. Sec. 4, 5.1 and 7: 2,100 images, 50 per class. The supplied archive holds 14,249 images. | 2,100 images, 50 per class, as the experimental sections state. | Sec. 5.1 gives the split counts 1,680 / 210 / 210, which only fit 2,100. | None beyond the subset guess. |  | adopted before the runs |  | `subset-first-50-per-class` | `data` |  |
| `efficientnet-accuracy-statements` | `paper_internal_conflict` | EfficientNetB7's published test accuracy. | Table 1 and Sec. 7: 0.967. Fig. 10's confusion matrix sums to 210 with one error, 209/210 = 0.995, over 41 classes. Sec. 5.2 says it 'had no errors' and that the test matrix 'contains only 40 classes'. | Table 1's 0.967 as the target value. | The assignment is Table 1. | The published value this attempt compares against is itself uncertain. Fig. 9 and Fig. 10 also have different per-class counts, so LeNet and EfficientNetB7 were scored on different test splits. |  | adopted before the runs |  |  | `evaluate` |  |
| `total-params-count` | `guess` | How 'Total params' is counted. | Table 1 column 'Total params (M)', no definition. | Model.count_params(): trainable and non-trainable parameters, divided by 1e6. | The name matches Keras' model summary line 'Total params'. | None: all four values round to Table 1's. |  | adopted before the runs |  |  | `evaluate` |  |

## Noncompliance

| ID | Description | Requirement | Run IDs | Effect | Resolution |
| --- | --- | --- | --- | --- | --- |
| `populate-ceiling-not-enforced` | populate-dataset-1 declared a 3,600 s wall-time ceiling, but its hung upload client was not stopped until about 66 hours after launch. The agent session was interrupted and nothing enforced the ceiling. | modal-data-compute.md, Full-run reliability: cancel runaway or clearly invalid work within the recorded stop ceiling; AGENTS.md: monitor remote jobs to a terminal state. | `populate-dataset-1` | None on data or results: the client transferred nothing, no Modal compute ran, and the Volume was unchanged. | Stopped on discovery. The retry runs the upload under a hard time limit. |

## Attempts, failures, and dead ends

- **Dataset upload, attempt 1 (`populate-dataset-1`).** The upload client stalled and transferred nothing; it was found and killed about 66 hours later, far past its declared ceiling (see Noncompliance). Nothing reached the Volume. Attempt 2 ran under a hard time limit and finished in 4.5 minutes.
- **Preflight on a T4 (`preflight-1`).** Hypothesis: all four models train at batch 64 on a 15 GB T4. Result: three did; EfficientNetB7 ran out of memory. Kept: the unchanged batch size. Changed: the GPU.
- **Preflight on an A100-40GB (`preflight-2`).** Hypothesis: more GPU memory runs EfficientNetB7 at batch 64. Supported: peak 31.8 GB. The paper's 8 GB card could not have run this configuration, so the authors' EfficientNetB7 setup differed in some way this attempt cannot determine.
- **EfficientNetB7 at chance level.** Investigated without changing the protocol: the training curves show the model fits the training set, and the diagnostic run isolates the input scaling as one contributing factor (see Results).
- **Architectures.** All four were pinned by matching Table 1's parameter counts exactly before any training: 1,071,162; 35,498,090; 23,667,498; 64,204,093.

## Candidate flags, ethics, and human evaluation

- **`copied_scores: no`.** Confirmed: Table 1 holds only the paper's own experiments.
- **`includes_human_evaluation: no`.** Confirmed: the paper reports none, and this attempt involved no participants or raters.
- **`potential_ethical_concerns: yes`.** Investigated. The data are photographs of hands. A contact sheet of one randomly chosen image per class (42 images) shows a single hand or forearm against a plain wall in every image, no face, and hands of several different people. The paper states no consent or ethics basis. Processing rests on the authors' permission to reproduce, given to the study, and on the data gate decision recorded in `ksl-data-cloud-use`. The copy is private to the study workspace; no image is committed, published or redistributed, and every retained artifact is a metrics file or weights. No separate ethics gate was opened.
- **`compute_requirements`.** The queue quotes the paper's desktop with an RTX 2080 8 GB. See `hardware-a100` and `batch-size-64`.

## Author and team contact

Before this attempt began, the study's data-access coordination had contacted the authors (queue record: contact dates 2026-08-20 and 2026-09-22, reply received), who supplied the dataset archive and the two notebooks.

After the independent attempt was complete, a study team member emailed the authors on 2026-11-23 (`author-protocol-questions-1` in `reproduction.json.contacts`). The message asks them to confirm or correct each protocol detail this attempt had to guess, to resolve the paper's internal conflicts, and to share the full penalty matrix, its scoring code and the original Table 1 training code or logs. It discloses none of this attempt's results. No reply as of 2026-11-23, so the contact has changed nothing: no run, target result, guess or status. The four Penalty targets stay closed as not produced (gate `ksl-penalty-metric`) and could be revisited if the authors provide the matrix and scoring code.
