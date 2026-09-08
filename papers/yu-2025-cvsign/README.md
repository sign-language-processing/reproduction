---
license: cc-by-nc-sa-3.0
tags:
  - continuous-sign-language-recognition
  - reproduction
datasets:
  - rwth-phoenix-weather-2014t
---

# CVSign reproduction

Reproduction of Yu, Liu, Feng, Xu, Jin, and Yang, *Improving Continuous Sign Language Recognition via Cross-Frame Interactions in Expanded Contextual Spaces* (ICASSP 2025), doi:10.1109/ICASSP49660.2025.10890162.

The assigning user limited the reproduction to PHOENIX14-T: PHOENIX14 (Table I) and CSL-Daily (Table II) were dropped for cost.

**Preference level:** 3

**Pipeline status:** `complete` — both in-scope targets (PHOENIX14-T dev and test) were produced; PHOENIX14 and CSL-Daily are out of scope.

**Numerical agreement:** `agrees` — PHOENIX14-T dev **17.5** vs 17.4 published (+0.1), test **19.4** vs 18.6 (+0.8)

**Attempt dates:** 2026-09-27 to 2026-09-29. An initial attempt (2026-09-05 to 2026-09-17) was replaced; see *Attempts, failures and dead ends*.

CVSign's new contribution is two modules, Contextual Correspondence Awareness (CCA) and Contextual Variability Awareness (CVA). The authors have not published code for them, so we implemented both from the paper; this is why the preference level is 3. Everything else in CVSign comes from earlier work that does have published code. We run that code at pinned commits, with only small patches:

- CorrNet: training loop, backbone, data loading and evaluation;
- TLP: the temporal decoder;
- VAC and SMKD: the training losses and the shared classifier (reached through CorrNet's code);
- ctcdecode: beam-search decoding;
- sclite: WER scoring.

## Scope and target contract

The paper's main results are Table I (PHOENIX14 and PHOENIX14-T Dev/Test WER, "CVSign (ours)") and Table II (CSL-Daily Dev/Test WER). All six are in the target ledger; the ablations (Tables III–VII) only fix the default configuration (modules after layers 2/3/4, CCA L=[3,5,9], CVA L=[9,9,9], full-image query area)..

| Target ID | Table | Dataset / split | Published | In scope |
| --- | --- | --- | ---: | --- |
| table1-phoenix14-dev-wer | I | PHOENIX14 / dev | 17.8 | no |
| table1-phoenix14-test-wer | I | PHOENIX14 / test | 18.0 | no |
| table1-phoenix14t-dev-wer | I | PHOENIX14-T / dev | 17.4 | yes |
| table1-phoenix14t-test-wer | I | PHOENIX14-T / test | 18.6 | yes |
| table2-cslday-dev-wer | II | CSL-Daily / dev | 25.7 | no |
| table2-cslday-test-wer | II | CSL-Daily / test | 24.7 | no |

Metric: gloss WER (Eq. 11) under the PHOENIX sign-recognition protocol as CorrNet runs it: beam search (ctcdecode, width 10, no LM), the PHOENIX `preprocess.sh` simplifications, `mergectmstm.py`, and sclite against ground truth equal gloss for gloss to the official RWTH `PHOENIX-2014-T-groundtruth-{dev,test}.stm`. Checkpoint selection: lowest dev WER over the completed epochs (earliest on ties), scored once on dev and test.

## Results

PHOENIX14-T gloss WER in %, lower is better. All rows except the last are copied from Table I of the paper; only the last row was produced here.

| System | Dev WER | Test WER |
| --- | ---: | ---: |
| VAC | 21.4 | 23.9 |
| SMKD | 20.8 | 22.4 |
| TLP | 19.4 | 21.2 |
| SEN | 19.3 | 20.7 |
| CorrNet | 18.9 | 20.5 |
| SignGraph | 17.8 | 19.1 |
| TCNet | 18.3 | 19.4 |
| STMC* | 19.6 | 21.0 |
| C²SLR* | 20.2 | 20.4 |
| TwoStream-SLR* | 17.7 | 19.3 |
| CVSign (paper) | 17.4 | 18.6 |
| **CVSign (reproduced)** | **17.5** | **19.4** |

\* uses additional visual cues such as keypoints.

The reproduced dev WER is 0.1 above the paper and still below every baseline. The reproduced test WER is 0.8 above the paper: it ties TCNet instead of beating it and trails SignGraph (19.1) and TwoStream-SLR (19.3). Evidence: runs `full-phoenix14t-002` and `eval-phoenix14t-002`, checkpoint of epoch 42 (lowest dev WER; the test split was scored once, after selection); sclite `.sys` sha256 `9bc1a2d8…` (dev) and `fc861e22…` (test).

### Training trajectory (`full-phoenix14t-002`)

| Epoch | 0 | 3 | 9 | 15 | 24 | 25 | 28 | 42 | 51 | 62 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev WER | 83.4 | 30.4 | 22.6 | 20.6 | 20.0 | 18.7 | 17.8 | **17.5** | 17.5 | 17.6 |

The learning rate drops ×0.2 at epochs 25 and 40. Dev WER stayed within 17.5–18.2 from epoch 35 onward. Training was stopped after epoch 62 of 70 on the assigning user's instruction, because it appeared to have converged (gate `phoenix14t-rerun-stopped-epoch-63`). Epochs 63–69 would have run at the same final learning rate; whether they would have lowered dev WER is unknown. Mean training loss went 82.2 → 5.0.

### Validating the evaluation implementation

To check our data preparation, decoding and scoring independently of CVSign, we ran CorrNet's own released PHOENIX14-T model, with unmodified CorrNet code, through the same pipeline (`corrnet-checkpoint-check-001`). It scores exactly what CorrNet reports: dev 18.9, test 20.5. So the gap between our CVSign numbers and the paper cannot come from data or evaluation. It must come from our CCA/CVA implementation, the ResNet34 backbone, the TLP decoder inside CorrNet's code, or training.

## Source provenance

| Artifact | Source | Pin / SHA-256 | Role |
| --- | --- | --- | --- |
| Paper PDF | https://doi.org/10.1109/ICASSP49660.2025.10890162 | `d9f9946bbae293bf26e998d5b34eaf6628d5566738bf8cd6dfc47f68b69a228f` | Architecture, recipe, targets |
| CVSign code | searched 2026-09-27 | none found | See *Code search* below |
| CorrNet | https://github.com/hulianyuyy/CorrNet | `5a361cc709ae8f697f4c9e23760e64c2daa8f744` | Harness: main.py, 3D-kernel ResNet, loader, losses, evaluation |
| TLP | https://github.com/hulianyuyy/Temporal-Lift-Pooling | `21be215c6ba749d98ce6a51f4b84cc9d5029bfe6` | `modules/tconv.py`, the "Temporal Decoder from TLP" |
| VAC | https://github.com/ycmin95/VAC_CSLR | `71f3e0334fbc8cecc7ce9816ec69781068abaac0` | Considered; its losses reach us unchanged through CorrNet/TLP |
| ctcdecode | https://github.com/parlance/ctcdecode | `c90ad94a0b19554f80804fb7812f2a1447a34a70` | Beam search |
| SCTK (sclite) | https://github.com/usnistgov/SCTK | `9688a26882a688132a5e414cadcb4c19b6fffaba` | Scoring |
| ResNet34 ImageNet | https://download.pytorch.org/models/resnet34-333f7ec4.pth | `333f7ec4c6338da2cbed37f1fc0445f9624f1355633fa1d7eab79a91084c6cef` | Backbone initialisation |
| CorrNet PHOENIX14-T checkpoint | CorrNet README, Google Drive `1c_wNHYMqCbqRE5KqrQL1P6chOw5VBS6Q` | `e0e7e5678b4dec791a1d5681110666f91c88bb6dccf620d383ad8029a7af0162` | Harness validation only |

Why CorrNet: CVSign's encoder has CorrNet's structure. Modules sit after ResNet layers 2, 3 and 4 behind zero-initialised residual gates, affinities are weighted by `sigmoid(·) − 0.5` (Eq. 4 and CorrNet's `Get_Correlation`), and the paper writes kernels as 1×3×3 and 1×1×1, which is CorrNet's 3D-kernel ResNet. CorrNet's `configs/baseline.yaml` already has CVSign's batch size, optimiser, VAC loss weights and SMKD shared weight-normalised classifier. CorrNet is pinned to its last commit touching code (2023-11-23, before CVSign); HEAD `a812814` differs only in demos. Neither CorrNet nor TLP ships a licence file; both are run unmodified apart from `patches/` and are not redistributed.

## Implementing CCA and CVA: unstated details

`patches/cvsign-2-cca-cva-modules.patch` puts CCA and CVA, written from Sec. II, into CorrNet's `modules/resnet.py`. The patch marks each unstated value inline:

| Detail | Paper | Our choice | Why |
| --- | --- | --- | --- |
| C_hid (CCA and CVA) | "downsample the channels into C_hid", no value | 64 | Unstated |
| CCA heads | "n heads" in Fig. 2 | 8 | Unstated |
| CCA projections | a CNN to C_hid, then "two distinct CNNs" for current and context frames, one CNN back to C_in | 1×1×1 convs, no bias, CorrNet's initialisation | Kernel sizes unstated; CorrNet's 1×1×1 convs have no bias |
| Context frames | window of 2n+1 frames | The 2n frames other than the current one | Fig. 2, "Adjacent 4 Frames" for a window of 5 |
| Window boundary | not stated | Edge frames repeated | As CorrNet's `Get_Correlation` does |
| Eq. 5 scaling | Σ over m, i′, j′, no normalisation | Literal sum | See the note below |
| CVA Conv_Block | "primarily consists of three CNNs with a kernel size of 1×3×3" | conv-BN-ReLU, conv-BN-ReLU, conv; one Conv_Block shared by every m | Norm and activation unstated |
| Backbone | ResNet34 pretrained on ImageNet | CorrNet's 3D-kernel ResNet34 loading torchvision's weights as its `resnet18()` does | Upstream `resnet34()` loads no weights |
| TLP regularisers | "adopt the Temporal Decoder from TLP" | TLP's Cu/Cp at 0.001, as TLP trains its decoder | Paper names only the VE/VA losses |
| Checkpoint selection | not stated | Best dev WER over 70 epochs | Rule of the code CVSign builds on |
| Mixed precision | not stated | CorrNet's AMP | Upstream default |
| Seed | not stated | 0 (CorrNet config) | Single run, as the paper appears to report |

On Eq. 5: taken literally, CCA's output at initialisation is about 290×, 94× and 324× its input at layers 2, 3 and 4 (measured on CPU). CorrNet's own `Get_Correlation` gives 7.8×, 0.2× and 0.6×. This is harmless at the start because α = 0, and in `diagnose-loss-005` Adam moves α only by about 1e-3 in 200 steps. It is still the least-constrained guess in the model, and a reviewer should weigh it.

With the gates at zero, the patched encoder reproduces torchvision's ImageNet ResNet34 per frame to 1.4e-6, which confirms the weights load. The checkpointed CCA gives bit-identical outputs and gradients to the uncheckpointed one.

## Environment and patches

- `Dockerfile`: `ghcr.io/sign-language-processing/reproduction:latest` (NGC PyTorch 26.02: PyTorch 2.11.0a0, CUDA 13.1, cuDNN 9.19, Python 3.12). On top: SCTK, ctcdecode, `opencv-python-headless==4.10.0.84`, `scipy==1.13.1`, `gdown==5.2.0`, pristine CorrNet at `/opt/CorrNet`, and the patched CVSign tree at `/workspace/CVSign`. CorrNet's own pins (opencv 4.5.5, scipy 1.7, numpy 1.20) have no Python 3.12 wheels; SciPy 1.13 is the newest that still ships `scipy.misc`, which CorrNet imports.
- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`: upstream `torch.load` of its own checkpoints, which carry numpy RNG state, fails under PyTorch ≥ 2.6 defaults.
- Upstream is run as published through `main.py --config ./configs/cvsign.yaml`. `main.py`'s "remove existing work dir?" prompt is answered `n` on stdin, so resumed runs keep their files.

| Patch | Applies to | Behaviour-changing | Why |
| --- | --- | --- | --- |
| `ctcdecode-cxx17.patch` | ctcdecode `c90ad94` | no | The unpatched build fails with "C++17 or later compatible compiler is required to use PyTorch"; under C++17, five KenLM `throw(SpecialWordMissingException)` specifications are errors and are removed |
| `corrnet-ctc-without-cudnn.patch` | CorrNet | no | cuDNN's CTC kernel returns NaN gradients in this stack; PyTorch's native kernel computes the same loss |
| `cvsign-1-tlp-decoder-losses.patch` | CorrNet + TLP `tconv.py` | no | Forwards TLP's lift-pool regularisers to the loss, as TLP's criterion does |
| `cvsign-2-cca-cva-modules.patch` | CorrNet | yes: the reimplemented contribution | CCA/CVA replace `Get_Correlation`; ResNet34 ImageNet loading |
| `cvsign-3-config.patch` | CorrNet | yes: the paper's recipe | `configs/cvsign.yaml`: ResNet34, 70 epochs, ×0.2 at epochs 25/40, PHOENIX14-T |

SHA-256 of each patch, of the Dockerfile and of `modal_app.py` are in `reproduction.json` (`patches`, `environment`).

The initial upstream preflight (`preflight-phoenix14t-003`) barely trained because cuDNN's CTC kernel returns NaN gradients in this software stack, also for unmodified CorrNet; the `diagnose-*` runs in `reproduction.json` isolated the cause, and `corrnet-ctc-without-cudnn.patch` computes the same loss with PyTorch's native kernel instead.

## Data provenance and permissions

PHOENIX14-T is the only dataset used.

| Field | Value |
| --- | --- |
| Version | `phoenix-2014-T.v3` |
| Frames | official `features/fullFrame-210x260px/{split}/{id}/*.png` |
| Labels | `annotations/manual/PHOENIX-2014-T.{split}.corpus.csv` |
| Source | RWTH release page (2026-09-05); archive placed on the Volume on 2026-09-15 by the SLTUNET reproduction (`raw/PROVENANCE.txt`) |
| Licence and cloud basis | CC BY-NC-SA, non-commercial research; project-cloud basis as recorded for this Volume path in `papers/camgoz-2018-nslt` |
| Path | `datasets:rwth-phoenix-2014-t/raw/PHOENIX-2014-T-release-v3/PHOENIX-2014-T` |
| Counts | 7096 / 519 / 642 clips; 827354 / 55775 / 64627 frames (train / dev / test) |
| Resized archive | 256×256, sha256 `18fe6809bf9b613e18c20887e1fceffcd0735ac544e7de06f3e13ab5de1693cd`, 68.1 GB, on the results Volume, not redistributed |

`prepare-frames-001` resized the frames with CorrNet's own `resize_dataset` (`cv2.INTER_LANCZOS4`). CorrNet's `preprocess/dataset_preprocess-T.py` cannot run end to end on this release: its CSVs list frames as `{id}/1/*.png`, while the frames are at `{id}/*.png`, so the script's globs match nothing. CorrNet's committed split files use `{id}/*.png` and otherwise equal the CSVs row for row (checked, 0 mismatches), so the script's `resize_dataset()` is driven with them, exactly as its `__main__` does. Resized frame counts equal the original PNG counts and CorrNet's recorded `num_frames` in every split. The release contains one non-frame file, `dev/31May_2011_Tuesday_tagesschau-4295/createDnnTrainingLabels-profile.py.lprof`, which the `*.png` glob skips. OpenCV 4.10 replaces CorrNet's pinned 4.5.5; resize output was not compared across versions.

## How to repeat this

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/yu-2025-cvsign/modal_app.py::launch_prepare_frames
```

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/yu-2025-cvsign/modal_app.py::launch_check_corrnet
```

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/yu-2025-cvsign/modal_app.py::launch_preflight --run-id preflight-phoenix14t-004
```

Train (it resumes itself from the newest epoch checkpoint, also across Modal's 24 h limit) and then evaluate the lowest-dev-WER checkpoint:

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/yu-2025-cvsign/modal_app.py::launch_train --run-id full-phoenix14t-002
```

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/yu-2025-cvsign/modal_app.py::launch_evaluate --train-run-id full-phoenix14t-002 --eval-run-id eval-phoenix14t-002
```

The `diagnose_*` entry points behind the cuDNN CTC diagnosis are no longer in the working tree; they are preserved at commit `2bcb12c` (`papers/yu-2025-cvsign/modal_app.py`).

## Execution evidence and compute

All runs are in `reproduction.json.runs` with Modal app IDs, timestamps, ceilings and artifact hashes. Outputs live on the Modal Volume `yu-2025-cvsign-results` under `runs/{run_id}/`: upstream `log.txt`, `dev.txt`, `test.txt`, sclite outputs, checkpoints and `dirty.patch`, the exact source diff each run used.

`preflight-phoenix14t-004` (one L40S, 46068 MiB):

| Measurement | Value |
| --- | --- |
| Train throughput | 100 steps / 27451 frames in 89 s: 3.24 ms per frame |
| Dev evaluation | ~0.26 s per clip (incl. beam search) |
| Peak GPU memory | 44172 MiB (nvidia-smi, incl. allocator cache), longest clips included |
| Checkpoint | 829 MB per epoch |
| Resume | epoch 1 continued from the epoch-0 checkpoint; mean loss 182.6 → 111.0 |

**Full run (`full-phoenix14t-002`), measured:** 63 epochs at about 33 min each (32 min training, 1.5 min dev evaluation), on one L40S. That is **36.8 GPU-hours** of container time, including three recomputed epochs and staging, or about **USD 118** at USD 3.22/h. The approved ceilings were 70 GPU-hours, CHF 200 and 4 days. `eval-phoenix14t-002` took about 10 minutes more. The preflight projected 55 GPU-hours because its subset was weighted toward long clips.

About 2.1 L40S GPU-hours of diagnostics and preflights plus about 1.5 CPU-hours of preprocessing were spent on 2026-09-27.

## Attempts, failures and dead ends

- **Initial attempt (Claude Opus 5, 2026-09-05 to 2026-09-17, replaced).** Besides CCA and CVA, it reimplemented from scratch everything CVSign borrows (CorrNet's training code, TLP's decoder, the losses, data loading, decoding and scoring) instead of running the published code, and got much of it wrong. Examples: a random crop drawn separately for every frame, colour channels reversed, max pooling instead of TLP's pooling, greedy decoding, and scoring without the PHOENIX gloss normalisation. Its PHOENIX14-T run (`full-phoenix14t-001`, about 47 A10 GPU-hours, stopped at 53 of 70 epochs) reached 31.48 dev / 31.64 test WER. Its code is at commit `57dd7fe`. The current attempt (Claude Opus 5.5) replaced it with the published dependencies.
- **Environment and glue failures on 2026-09-27, not retained as runs:**
  - The unpatched ctcdecode build failed, which motivated the C++17 patch.
  - An import check ran inside the source tree and was shadowed by it.
  - Modal rejected an `ephemeral_disk` below 512 GiB.
  - Two frame-preparation attempts were stopped by our own checks: running CorrNet's script on the CSVs found no frames because of the `{id}/1/` path, and counting all files instead of PNGs tripped on the `.lprof` file.
- **Out-of-memory restarts in `full-phoenix14t-002`.** Three CUDA OOMs occurred during upstream's fp32 dev evaluation (2026-09-27T22:31Z, 2026-09-28T02:57Z and 05:43Z). Each was a single 10.41 GiB CCA affinity allocation, failing with about 11 GiB reserved but unallocated, i.e. fragmentation. Modal retries resumed each time from the previous epoch's checkpoint, costing about 35 minutes each and using all three retries. The run was stopped after the complete epoch-22 checkpoint and relaunched with `PYTORCH_ALLOC_CONF=expandable_segments:True`. That setting changes only the allocator, not the computation, and there was no OOM in the remaining 40 evaluations.
- **`preflight-phoenix14t-003`** is classified `invalid_run`: all commands exited 0, but almost every update was skipped (cuDNN CTC NaN).
