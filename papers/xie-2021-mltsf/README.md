# Xie et al. (2021): mLTSF-Net, Table 3 — conditional reproduction

**Pipeline status:** `insufficient_information`
**Numerical agreement:** `not_assessed`
**Preference level:** 3

This attempt targets Table 3 of Pan Xie, Zhi Cui, Yao Du, Mengyi Zhao, Jianwei Cui, Bin Wang and Xiaohui Hu, *Multi-Scale Local-Temporal Similarity Fusion for Continuous Sign Language Recognition*, [arXiv:2107.12762v1](https://arxiv.org/abs/2107.12762v1) (2021). We deliberately stuck to this arXiv version rather than the later Pattern Recognition article (2023, DOI 10.1016/j.patcog.2022.109233), because it is the version that was originally selected and annotated for reproduction. No code or weights were released, so we reimplemented the model from the paper.

The paper reports two systems: mLTSF-Net and mLTSF-Net + GFE. We trained mLTSF-Net for the full 60 epochs and reached **45.3% dev / 43.0% test WER**, against the paper's 23.8% / 23.5%. Because we could not reproduce the first system, we chose not to train the second one, for computational cost reasons: completing it was estimated at 32–36 GPU-hours (about USD 96–108). An mLTSF-Net + GFE run started in parallel was stopped during its first epoch and not resumed; its checkpoint is retained.

The paper omits several settings, including batch size, decoder and normalization behavior. We filled them in with documented choices, so these numbers are conditional evidence, not a claim of faithful reproduction.

## Results

WER, deletion and insertion rates are percentages; lower is better. Differences are percentage points and descriptive, not a scientific comparability judgment.

| Split / metric | Paper | Ours | Ours minus paper |
| --- | ---: | ---: | ---: |
| Dev deletion | 8.5 | 17.0 | +8.5 |
| Dev insertion | 3.2 | 6.6 | +3.4 |
| Dev WER | 23.8 | 45.3 | +21.5 |
| Test deletion | 8.3 | 15.4 | +7.1 |
| Test insertion | 3.1 | 6.7 | +3.6 |
| Test WER | 23.5 | 43.0 | +19.5 |

Our numbers come from the epoch 52 checkpoint (lowest dev WER), greedy CTC decoding and BatchNorm statistics computed per video at inference; see [Normalization](#normalization). The same checkpoint evaluated with stored running statistics gave 56.7% dev / 54.6% test WER. That test score was seen before per-video normalization was chosen, so our test result is not an untouched first-use benchmark. mLTSF-Net + GFE (paper: 22.9% dev / 23.0% test) has no result.

Evidence: run `full-mltsf-segment-5` (training, checkpoint selection) and `adapted-normalization-evaluation-1` (reported scores), under `modal://xie-a514cc97-results/`. Predictions, raw sclite reports, environment and hashes are registered in `reproduction.json`. The adapted evaluation covers all 540 dev and 629 test videos and leaves every parameter, buffer and the checkpoint hash unchanged.

## Table 3 targets

All values below are the paper's published percentages. `reproduction.json` has one target per numeric cell: 12 for the two mLTSF systems and 46 copied prior-work values, kept as comparison context and not re-run. Dashes denote cells absent from the paper.

| System | Dev del | Dev ins | Dev WER | Test del | Test ins | Test WER | Attempt outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DeepHand | 16.3 | 4.6 | 47.1 | 15.2 | 4.6 | 45.1 | Copied comparison |
| Staged-Opt | 13.7 | 7.3 | 39.4 | 12.2 | 7.5 | 38.7 | Copied comparison |
| Re-Sign | — | — | 27.1 | — | — | 26.8 | Copied comparison |
| CNN-LSTM-HMM | — | — | 26.0 | — | — | 26.0 | Copied comparison |
| SBD-RL | 9.9 | 5.6 | 28.6 | 8.9 | 5.1 | 28.6 | Copied comparison |
| DNF | 7.8 | 3.5 | 23.8 | 7.8 | 3.4 | 24.4 | Copied comparison |
| SubUNets | 14.6 | 4.0 | 40.8 | 14.3 | 4.0 | 40.7 | Copied comparison |
| DeepSign | 12.6 | 5.1 | 38.3 | 11.1 | 5.7 | 38.8 | Copied comparison |
| SFNet | — | — | 38.0 | — | — | 38.1 | Copied comparison |
| FCN | — | — | 26.1 | — | — | 26.0 | Copied comparison |
| FCN + GFE | — | — | 23.7 | — | — | 23.9 | Copied comparison |
| mLTSF-Net | 8.5 | 3.2 | 23.8 | 8.3 | 3.1 | 23.5 | Conditional result above |
| mLTSF-Net + GFE | 7.9 | 2.7 | 22.9 | 7.6 | 3.0 | 23.0 | Not trained (cost) |

WER is corpus `(substitutions + deletions + insertions) / reference gloss tokens`, multiplied by 100. The paper does not identify its evaluator or decoding protocol. The [official dataset README](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/README-multisigner) refers to NIST sclite; this does not establish that the authors used an unchanged evaluator.

## Normalization

We trained with one video per optimizer step. Every BatchNorm layer therefore normalizes by that video's own statistics during training, while standard evaluation uses stored running averages. A read-only diagnostic (`bn-generalization-diagnostic-2`) compared both on 128 hash-selected training and 128 dev videos, with identical center-crop inputs and greedy decoding:

| Checkpoint | BatchNorm statistics | Train WER (%) | Dev WER (%) |
| --- | --- | ---: | ---: |
| Epoch 52 | Stored running statistics | 45.0 | 54.8 |
| Epoch 52 | Per-video, all layers | 6.6 | 45.1 |
| Epoch 52 | Per-video, spatial layers only | 21.5 | 47.2 |
| Epoch 52 | Per-video, temporal layers only | 11.7 | 48.0 |
| Epoch 60 | Stored running statistics | 45.2 | 55.0 |
| Epoch 60 | Per-video, all layers | 6.8 | 44.0 |

Stored statistics cost about 10 dev WER points and badly degrade even training-set predictions, so we report per-video normalization. Each BatchNorm layer uses the current video's statistics (spatial layers over its frames and positions, temporal layers over its time steps). Stored buffers stay unchanged; no labels or other videos are used. We do not claim the authors used this behavior. Six dev clips contain seven tokens absent from the training vocabulary; they count in every WER.

## Training curve and generalization

Per-epoch record from the registered artifact `full-mltsf-segment-5-epochs-jsonl` (SHA-256 `3f4075ef85a0b7b57f7ed5067902ac9cf7e1cc04966e8b56a9cb93f055204cef`), which contains all 60 epochs; selected rows are shown. Loss is the epoch mean of the per-video summed CTC loss on augmented training inputs. Dev scores use stored running statistics, as during training, and drove checkpoint selection: whenever an epoch set a new lowest dev WER it overwrote `best.pt`, and the last such epoch, 52 (bold), is the selected checkpoint.

| Epoch | Train loss | Dev WER | Dev del | Dev ins |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 58.22 | 88.2 | 79.1 | 0.3 |
| 5 | 40.60 | 79.4 | 62.2 | 0.5 |
| 10 | 29.03 | 69.5 | 44.1 | 2.0 |
| 15 | 21.59 | 61.8 | 32.2 | 3.2 |
| 20 | 15.80 | 61.6 | 29.8 | 3.4 |
| 23 | 12.97 | 60.9 | 29.6 | 4.1 |
| 31 | 7.51 | 59.8 | 23.8 | 6.1 |
| 40 | 4.49 | 60.1 | 30.8 | 4.2 |
| 44 | 3.75 | 59.5 | 23.9 | 6.4 |
| 45 | 2.34 | 60.5 | 32.2 | 3.6 |
| 50 | 1.32 | 57.9 | 28.1 | 4.9 |
| 51 | 1.22 | 61.9 | 36.0 | 3.5 |
| **52** | **1.10** | **56.7** | **26.8** | **4.8** |
| 55 | 0.88 | 57.8 | 31.0 | 4.0 |
| 60 | 0.72 | 57.2 | 29.9 | 3.9 |

Observations:

- From about epoch 20, dev WER stays between 56.7% and 64.5% while training loss falls from 15.8 to 0.72, roughly 22-fold: the model memorizes the training set without improving on dev. With per-video normalization the gap remains large (6.6% train versus 45.1% dev on the diagnostic sample), and mean CTC loss is 0.36 on training versus 32.2 on dev videos at epoch 60.
- Dev performance is already weak at epochs 13–17 (61.8–66.1% WER), while training loss is still 19–24 per video, roughly 1.6–2.1 nats per gloss. So poor dev performance is not only late memorization; stored statistics account for about 10 points of it.
- After epoch 40, dev deletion swings between 22.9% and 36.0% from one epoch to the next, with insertions moving the other way. This is consistent with noisy running statistics from one-video BatchNorm updates (not verified). Epoch 52 beat the next-best epochs 57 and 60 by only 0.2 and 0.5 points, so the selection is weakly determined.

Reconstruction choices that plausibly contribute (review notes from code and log inspection; none was tested, and none should be tuned toward the published score):

1. **One video per optimizer step.** Besides the normalization mismatch above, each epoch has about 5,671 updates, twice as many as with two videos per batch. The paper's 60-epoch schedule may assume a different batch size.
2. **Summed CTC loss changes the relative strength of the paper's only explicit regularizer.** Equation 15 is `L = Lctc + λ‖W‖²` with λ = 1e-4, and the paper describes no dropout. We sum CTC over each sequence rather than using PyTorch's default reduction, which divides by target length. The paper's figures (about 80,000 glosses in 6,841 sentences) imply about 11.7 glosses per sentence, so the data gradient is roughly 12 times larger relative to L2. Coupled `weight_decay=1e-4` also adds λW, whereas the gradient of λ‖W‖² is 2λW. The authors' reduction is unknown.
3. **Learning-rate milestones.** The paper says "After 40 epochs, we decay the learning rate by 0.5 every 5 epochs". We halve at epochs 45, 50 and 55; halving at 40/45/50/55 or 41/46/51/56 are also possible readings. The loss drop at epoch 45 (3.75 to 2.34) confirms where the first halving took effect.

Two choices follow the cited FCN paper, whose backbone mLTSF-Net adopts, and are therefore unlikely to be reconstruction errors: training from random initialization (FCN states its network "is end-to-end trainable without any pre-training") and BatchNorm after the temporal convolutions (FCN adds it "after every convolutional layer").

**Note for anyone completing mLTSF-Net + GFE.** GFE trains an auxiliary classifier on frame-level gloss labels. These labels come from force-aligning each training video with the current model, refreshed before epochs 16, 26, 36, 46 and 56. Our code computes these alignments in evaluation mode, that is, with the stored BatchNorm running statistics. The [Normalization](#normalization) table shows those statistics are badly mismatched (45.0% training WER versus 6.6% with per-video statistics), so the alignments, and therefore the GFE training labels, would likely be poor. No reported result is affected: the GFE run stopped during epoch 1, before the first alignment. Anyone resuming it should first decide which normalization to use for alignment.

## Guesses and deviations

The paper specifies the architecture (Tables 1–2), input size, augmentation, optimizer, learning rate, L2 coefficient and 60-epoch schedule. The executed reconstruction, `conditional-paper-reimplementation-v1`, was authorized in gate `reconstruction-protocol` on 2026-11-02. Choices for everything else, and the later user-approved adjustments, are listed below.

| ID | Kind | Description | Paper says | This attempt used | Basis | Effect | Scope | Status | Evidence | Related IDs | Affects | Dataset IDs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `batch-size` | `guess` | Batch size and sequence padding. | Not stated | One full video per optimizer step, no temporal truncation |  | Every BatchNorm layer normalizes by that video's own statistics during training (see per-video-batchnorm-evaluation) |  |  |  | `per-video-batchnorm-evaluation` |  |  |
| `seed-and-precision` | `guess` | Seed, initialization and precision. | Not stated | Random initialization, FP32, seed 42 |  |  |  |  |  |  |  |  |
| `backbone-channels` | `guess` | Spatial backbone channel sequence. | Table 2 backbone begins 3->32->32->64; cited FCN text says 3->32->64->64. Follow the assigned Table 2 rather than silently importing a different backbone. | The assigned Table 2, which differs from the channel sequence in the cited FCN paper |  |  |  |  |  |  |  |  |
| `optimizer-details` | `guess` | Adam settings, weight decay and loss reduction. | "We use the Adam optimizer [34] to optimize our model, where the initial learning rate is set to 10−4"; L2 regularization coefficient 10−4 (§5) | Adam at 1e-4 (default betas/epsilon), coupled weight decay 1e-4, per-sequence summed CTC loss, 60 epochs |  |  |  |  |  |  |  |  |
| `lr-milestones` | `guess` | Learning-rate decay epochs. | "After 40 epochs, we decay the learning rate by 0.5 every 5 epochs" (§5) | Halving at epochs 45, 50 and 55 |  |  |  |  |  |  |  |  |
| `spatial-crop` | `guess` | Random crop granularity. | "randomly crop the video frames to 224× 224"; center crop for testing (§5) | Frames resized to 256 × 256 (RGB in [0, 1]); one shared random 224 × 224 crop per video. Center crop for evaluation. |  |  |  |  |  |  |  |  |
| `temporal-augmentation` | `guess` | Temporal augmentation. | "we also scale up the training video sequence by +20% and then scale down by - 20%" (§5); Wording of temporal +20% then -20% augmentation does not specify sampling/interpolation or mapping cached GFE alignments through augmentation. | Nearest-neighbor resampling at a uniform factor in [0.8, 1.2] |  |  |  |  |  | `ctc-valid-augmentation` |  |  |
| `decoding` | `guess` | CTC decoding. | CTC decoder (greedy versus beam), beam width/LM, scoring normalization and deletion/insertion tie-breaking. | Greedy CTC decoding without a language model |  |  |  |  |  |  |  |  |
| `checkpoint-selection` | `guess` | Checkpoint selection. | Checkpoint selection and seeds are not documented. | Earliest lowest-dev-WER checkpoint; test evaluated only after selection |  |  |  |  |  |  |  |  |
| `scoring` | `guess` | WER scoring implementation. | Not stated | The pinned CorrNet mirror of the PHOENIX evaluator (preprocess.sh, mergectmstm.py) with NIST sclite |  |  |  |  | https://github.com/hulianyuyy/CorrNet/tree/a812814f4deefda0b75ab1dcc90b106c62912263 |  |  |  |
| `vocabulary` | `guess` | Gloss vocabulary. | Not stated | 1,231 training glosses plus blank. Dev/test tokens are not added. |  | Six dev clips contain seven tokens absent from the training vocabulary; they count in every WER. |  |  |  |  |  | `phoenix2014-multisigner-v3` |
| `gfe-settings` | `guess` | GFE auxiliary-loss settings. | GFE loss weight, enable epoch, alignment refresh schedule and handling of temporal augmentation are not specified for mLTSF. FCN uses weight 0.05, activation after epoch 15 and refresh every 10 epochs, but trains RWTH for 80 rather than 60 epochs. | FCN auxiliary head with loss weight 0.05, blank balancing, activation at epoch 16, canonical-input alignments refreshed before epochs 16, 26, 36, 46 and 56. Mapping cached alignments through temporal augmentation uses an inferred nearest-timestep formula recorded in reproduction.json. | The proposed cached-alignment temporal mapping is a substantive conditional inference, not a recovered fact. A strict reproduction requires authoritative augmentation/alignment-handling instructions. |  | mLTSF-Net + GFE (not completed) |  |  |  |  |  |
| `broken-training-clip` | `paper_deviation` | User-approved exclusion of one known broken PHOENIX training clip:5671 effective examples from5672 source CSV rows, matching pinned benchmark preprocessing. Dev/test unchanged. | 5,672 training videos | Training clip 13April_2011_Wednesday_tagesschau_default-14 (16 frames, 16 glosses) cannot fit a valid CTC path and is excluded, leaving 5,671 training videos | The pinned CorrNet preprocessing drops the same clip as broken |  |  | Gate `known-broken-training-clip` resolved | https://github.com/hulianyuyy/CorrNet/blob/a812814f4deefda0b75ab1dcc90b106c62912263/preprocess/dataset_preprocess.py#L15 |  |  | `phoenix2014-multisigner-v3` |
| `ctc-valid-augmentation` | `guess` | User-approved rejection sampling of temporal augmentation scales within[0.8,1.2] when a drawn length cannot support the CTC target. Full audit finds one remaining clip affected; no additional exclusions. |  | When a drawn temporal scale makes a clip too short for its CTC target, the scale is redrawn within [0.8, 1.2] |  | Only one 74-frame clip is affected |  | Gate `ctc-valid-augmentation` resolved |  | `temporal-augmentation` |  |  |
| `per-video-batchnorm-evaluation` | `guess` | Authorized post-training evaluation variant uses per-video statistics in every BatchNorm layer instead of stored running statistics; fixed original epoch52 selection and all other inference settings retained. Not asserted to recover the original authors' protocol; original scores retained. | BatchNorm after every convolution layer; inference behavior not stated | Each BatchNorm layer uses the current video's statistics (spatial layers over its frames and positions, temporal layers over its time steps). Stored buffers stay unchanged; no labels or other videos are used. | Stored statistics cost about 10 dev WER points and badly degrade even training-set predictions (diagnostic bn-generalization-diagnostic-2) | Per-video normalization was chosen after training and dev diagnostics and after the stored-statistics test score was seen. |  | Gate `adapted-normalization-evaluation` resolved |  | `batch-size` |  |  |

## Noncompliance

None.

## Data

PHOENIX-2014 multisigner, official v3 release, at `modal://datasets/rwth-phoenix-2014/raw/phoenix2014-release/phoenix-2014-multisigner` (frames from `features/fullFrame-210x260px`). Counts are 5,672 train, 540 dev and 629 test videos (799,006 / 75,186 / 89,472 frames), matching the paper. Per-split annotation SHA-256 values and a frame-path inventory hash are in `reproduction.json.datasets`. Dev/test scorer reference IDs match the annotations exactly. Permission basis: existing project dataset for non-commercial research under the RWTH CC BY-NC-SA 4.0 terms; no data was copied into Git or redistributed.

## Source search

No mLTSF implementation or weights were found. Searches covered the exact title, method name, authors, GitHub repository metadata, and Zenodo/OSF/GitHub; queries, endpoints and response hashes are in `source_search` and `sources`. The 33-page assigned PDF (SHA-256 `5ad3606c20b6ace2b14af559fc3ce287d0e03f3d977fdbc882fbc1a3d42d31ec`) has no code link or appendix; the publisher page returned HTTP 403, limiting supplement inspection. The cited [FCN paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690681.pdf) (SHA-256 `60ade4b2031f7875c854f708d42b8569f40452509cca4a039bbd6b031622a35d`) supplies GFE details but no executable code. No authors were contacted.

## Commands and environment

All Modal commands go through the `repro-sign` workspace wrapper, from the repository root. Use a new run ID for every call; `train` (attempt ID), `evaluate_adapted` and `diagnose_bn` reject IDs that were already used.

```bash
python3 papers/xie-2021-mltsf/scripts/test_model.py
python3 papers/xie-2021-mltsf/scripts/test_pipeline.py
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/xie-2021-mltsf/scripts/modal_app.py::audit --run-id data-audit-new
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/xie-2021-mltsf/scripts/modal_app.py::preflight --run-id preflight-new
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/xie-2021-mltsf/scripts/modal_app.py::train --run-id full-mltsf-seed42 --attempt-id full-mltsf-segment-new
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/xie-2021-mltsf/scripts/modal_app.py::evaluate_adapted --run-id adapted-normalization-evaluation-new
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/xie-2021-mltsf/scripts/modal_app.py::diagnose_bn --run-id bn-generalization-diagnostic-new
```

- `model.py` implements the network, CTC decoding, forced alignment and GFE; `experiment.py` loads frames, trains with resumable checkpoints, selects on dev, evaluates test and runs the scorer; `modal_app.py` runs everything in the container; `evaluate_adapted.py` and `diagnose_bn.py` implement the per-video normalization evaluation and diagnostic. The tests cover CTC path optimality, neighbor selection and alignment mapping, checkpoint/BatchNorm equivalence, loader pixels and RNG, and interrupted-training resume.
- Training resumes from the run directory's latest checkpoint, including mid-epoch position, optimizer and RNG state. It checkpoints every 100 videos or five minutes.
- The Dockerfile pins the shared base image by digest. Runs used an NVIDIA A100-SXM4-40GB (driver 610.57.04) with PyTorch 2.11.0a0+eb65b36914.nv26.02 and CUDA 13.1, eight CPU cores and 16 GiB RAM. Datasets are mounted read-only and the shared Hugging Face cache at `/cache/huggingface`.
- mLTSF-Net training took about 31 A100 GPU-hours across four segments. The whole attempt was conservatively accounted at USD 114 of a USD 160 user-authorized ceiling.
- CUDA CTC gradients are nondeterministic, so repeated GPU runs do not reproduce bitwise-identical trajectories.

## Limitations

- One seed and one independent reconstruction. Batch size, decoder, normalization behavior and other details remain unresolved, so the results are not comparable targets.
- Per-video normalization was chosen after training and dev diagnostics and after the stored-statistics test score was seen.
- mLTSF-Net + GFE was not trained.

## Attribution

Codex (GPT-6 Astra) implemented and ran the reproduction; session metadata identified GPT-6.1 Sol for part of the early execution and monitoring. Claude Code (Opus 5.5) then reviewed it and wrote up this report. Per-agent details are in `reproduction.json.agents`.
