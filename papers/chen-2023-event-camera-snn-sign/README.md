# Sign Language Gesture Recognition and Classification Based on Event Camera with Spiking Neural Networks

**Paper ID:** `d4acd1f57be13675331b5da0fec400b8f794215d`
**Citation:** Chen, X.; Su, L.; Zhao, J.; Qiu, K.; Jiang, N.; Zhai, G. *Electronics* 2023, 12, 786. https://doi.org/10.3390/electronics12040786
**Preference level:** 3

No code was published for this paper; the runs below are a reconstruction from the paper on top of a pinned third-party STBP implementation.

**Pipeline status:** `insufficient_information` — no Table 2 number could be produced by a sufficiently specified pipeline.
**Numerical agreement:** `not_assessed` — no in-scope target produced a comparable value; the conditional runs are evidence only.

## Targets

The assignment asks for Table 2. Its two "Ours" rows hold the paper's own six numbers; the other six rows are copied from prior work.

| Target | Paper | Result | Reason |
| --- | ---: | --- | --- |
| DVS_Sign_v2e, Acc | 77.00% | not produced | `protocol_ambiguous`: model and training recipe are underspecified |
| DVS_Sign_v2e, Acc1 | 79.00% | not produced | `metric_ambiguous`: "first part of the test set" is never defined |
| DVS_Sign_v2e, Acc2 | 76.00% | not produced | `metric_ambiguous`: "second part of the test set" is never defined |
| DVS_Sign (DAVIS346), Acc | 68.00% | not produced | `data_unavailable`: the recordings were not published |
| DVS_Sign (DAVIS346), Acc1 | 71.00% | not produced | `data_unavailable` |
| DVS_Sign (DAVIS346), Acc2 | 70.00% | not produced | `data_unavailable` |

These classifications were fixed before the full runs were launched. Whether the conditional numbers below say anything about the paper's claim is for a human reviewer to decide.

### Copied baselines (out of scope, not rerun)

| Row | Dataset | Values | Provenance |
| --- | --- | --- | --- |
| Ye et al. [42] | ASL | Acc 69.20 | cited prior work, not checked against its source |
| Zhang et al. [43] | EgoGesture | Acc 68.90 | cited prior work, not checked |
| Xu et al. [24] | ASL-DVS | Acc 51.4 | cited prior work, not checked |
| Monti et al. [44] | ASL-DVS | Acc 86.7 | cited prior work, not checked |
| Martinez et al. [18] | DVS-Lip | 55.60 / 75.46 / 65.51 | identical to Tan et al., CVPR 2022, Table 1 (where the input is "video", not "event") |
| Liu et al. [45] | DVS-Lip | 58.36 / 79.17 / 68.74 | identical to the TANet row of Tan et al., Table 1 |

## Conditional full runs

Four runs of the Table 3 configuration for the 77% row (step 80, dt 40 ms, Vth 0.3, lr 1e-3, batch 20, 200 epochs, Eq. 8 decay, seed 1), differing only in two choices the paper leaves open. Accuracy is on the published 150-file test split at the final epoch.

| Run | Optimizer | Input pooling | Final-epoch Acc | Diff. vs 77.00 | Files 0-4 | Files 5-9 | Best epoch (diagnostic) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `full-sgd-avg` | plain SGD | average | 6.67% | -70.33 | 6.67% | 6.67% | 6.67% (epoch 1) |
| `full-sgd-max` | plain SGD | max | 6.67% | -70.33 | 6.67% | 6.67% | 6.67% (epoch 1) |
| `full-adam-avg` | Adam | average | 74.67% | -2.33 | 72.00% | 77.33% | 77.33% (epoch 46) |
| `full-adam-max` | Adam | max | 63.33% | -13.67 | 61.33% | 65.33% | 74.00% (epoch 51) |

- **Plain SGD is silent.** In both SGD runs the network never emitted an output spike: training loss was exactly 0.5 and test accuracy 10/150 (chance) in all 200 epochs. "SGD using standard settings" is the paper's own wording, read as PyTorch defaults.
- **Adam learns.** Adam is the optimizer of the third-party base code, not what the paper states.
- **Single seed, small test set.** One file is 0.67 points. In the Adam runs test accuracy moved by several points between epochs (72.0-76.0% for `adam-avg` and 61.3-65.3% for `adam-max` over epochs 121-200).
- **"Files 0-4 / 5-9"** is one unverified reading of Acc1/Acc2 (paper: 79.00 / 76.00), reported only because the paper defines none.
- **Best-epoch values** are selected on the test set and are diagnostics, not results.
- **Best epoch of `full-adam-avg` vs the paper.** Its best epoch (77.33% = 116/150, epoch 46) is within one test file of the paper's 77.00%. That is consistent with the authors having used the base code's Adam and reported the best test epoch, but it does not establish it: the value is a maximum over 200 noisy epochs of a single seed, picked on the test set after seeing four runs. It does not change any target status.
- **The paper's 77.00% cannot be exact on 150 files.** Accuracy moves in steps of 0.67 points (115/150 = 76.67, 116/150 = 77.33). All of the paper's own accuracies are whole numbers printed with ".00", which suggests rounding or a test set of another size (the paper mentions an unexplained "validation set" of 100).

## Why no target is produced

**Protocol.** The paper gives no code and leaves these open (each choice made here is listed under "Reconstruction decisions"):

- Figure 3 is the only description of the network: block labels and the numbers 34, 34, 64, 64, 128, 128, 256, 15, with no kernel sizes, strides or pooling type.
- The event-to-frame encoding, LIF decay constant, surrogate gradient, SGD settings, seed and checkpoint rule are not stated.
- Section 4.3 says the initial learning rate is 1e-4; Table 3 gives 1e-3 for the 77% row.
- The paper says "the test set has 150 and the validation set has 100"; no validation split exists in the released data.

**Acc1 / Acc2.** The Table 2 caption ("accuracy of the first part / second part of the test set") is the wording Tan et al. use for DVS-Lip's two vocabulary parts. No such partition is defined for DVS_Sign. The DVS_Sign row also reports Acc1 71.00 and Acc2 70.00 with overall Acc 68.00, which no partition of one test set can give.

**DAVIS346 data.** The paper points to branch `main` of `najie1314/DVS` for the DAVIS346 recordings. That branch holds 150 files for classes 0-2 only, and every file is byte-identical to the same path in the v2e data on branch `master`. The real recordings are "available on request from corresponding authors"; no request was made.

## Sources

| Artifact | Pin | Use |
| --- | --- | --- |
| [Paper PDF](https://mdpi-res.com/d_attachment/electronics/electronics-12-00786/article_deploy/electronics-12-00786.pdf?version=1676520358) | SHA-256 `f0c3010dcb17082256239abdb6cd5cf2b1d86661d077cc9bc953137989ee5567`, accessed 2026-10-02 | targets and protocol |
| [najie1314/DVS `master`](https://github.com/najie1314/DVS/tree/master) | `439682135dc126337cd1d7f60e50764ec771ad2f` | DVS_Sign_v2e data (used) |
| [najie1314/DVS `main`](https://github.com/najie1314/DVS/tree/main) | `56a26f8f330540f62e09610704872981d2c66816` | inspected, rejected (3-class copy of v2e) |
| [thiswinex/STBP-simple](https://github.com/thiswinex/STBP-simple) | `dca9590b465c5cb37b3828bf6888c6857ad4238b` | LIF neuron and surrogate gradient, imported unchanged |
| [yjwu17/STBP-for-training-SpikingNN](https://github.com/yjwu17/STBP-for-training-SpikingNN) | `e3c0c93283cd76ac8af76f25cf52cb215aab7be8` | considered, rejected |
| [He et al. 2020](https://arxiv.org/pdf/2005.02183) | arXiv:2005.02183 | supporting evidence for the network layout |
| [Tan et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Tan_Multi-Grained_Spatio-Temporal_Features_Perceived_Network_for_Event-Based_Lip-Reading_CVPR_2022_paper.pdf) | SHA-256 `d1473d5c21c53d73f6c6c9e2d0b17af7865f0e83a3b81d5b5ec59447fbe2cfc3` | origin of the DVS-Lip rows and Acc1/Acc2 wording |

STBP-simple is not cited by the paper and is not the paper's artifact. It was chosen because its `state_update` is the paper's Algorithm 1 line for line (including the `SpikeAct` name) and its `steps` / `dt` / `Vth` hyperparameters and firing-rate MSE loss match Table 3 and Eq. 6. It is pinned to the last commit before the paper's submission (2022-11-30). It has no licence; it is cloned at image build and not redistributed.

**Code search (dead end).** The PDF has no code link. Web searches on 2026-10-02 for the exact title and for `DVS_Sign` / `DVS_Sign_v2e` with STBP and the authors' affiliation returned only publisher and institutional pages. The dataset account's other repositories (`v2e`, `chenxuena`, `text`) hold no code; `DVS` has no forks, releases or tags. The MDPI landing page could not be fetched (HTTP 403), so its supplementary section was not inspected beyond what the PDF states.

## Data

| Item | Value |
| --- | --- |
| Dataset | DVS_Sign_v2e: v2e event streams derived from LSA64, 15 classes |
| Split | as published: 600 train (40 per class), 150 test (10 per class); no file content shared between splits |
| Format | CSV rows `timestamp_seconds,x,y,polarity`, 128x128 grid, 23,297,875 events |
| Location | Modal Volume `datasets`, path `dvs-sign-v2e/`, mounted read-only at `/datasets` |
| Identifier | `MANIFEST.sha256` (750 file hashes), SHA-256 `6c3e9ca5a71e5f0c9af2e1ec1e53dd0ba95a0c084648f70e4504ee6456c7f6f1` |
| Permission | The repository has no licence file. The CC BY 4.0 paper states the dataset "has been open source" there. The user confirmed private project-cloud processing on 2026-10-02. Nothing is redistributed. |

The Volume was populated and the subset preflights ran before that confirmation was requested; the full runs were launched after it.

The survey record's ethics flag concerns the three volunteers recorded with the DAVIS346 without reported consent. No DAVIS346 recording was obtained or processed here.

## Reconstruction decisions

None of these is a paper fact.

| Item | Choice | Basis |
| --- | --- | --- |
| Neuron code | STBP-simple `layers.py`, unchanged | matches Algorithm 1 |
| Threshold | fires when `u - Vth > Vth` (effective 0.6) | behaviour of the pinned commit; upstream changed it to `u > Vth` in April 2023, after publication. Algorithm 1 as printed implies `u > Vth` |
| Decay, surrogate | tau 0.25, rectangular half-width 0.5 | upstream defaults |
| Network | Pool4 - Conv(2→34) - LIF - Conv(34→64) - LIF - AvgPool2 - Conv(64→128) - LIF - AvgPool2 - FC(8192→256) - LIF - FC(256→15) - LIF; 3x3 kernels, padding 1 | Figure 3 numbers read as channels/units and 32/16/8 as map sizes; same layout as He et al.'s DVS-Gesture CNN |
| Input pooling | run factor: average (upstream convention) or max (He et al.) | "pooling" only in the paper |
| Optimizer | run factor: plain SGD (literal) or Adam (upstream) | "SGD using standard settings" |
| Encoding | binary frame `[polarity, y, x, floor(t_ms / 40)]`, first 3.2 s, later events dropped | upstream N-MNIST preprocessing with both polarities |
| Learning rate | 1e-3 | Table 3 row for 77%; Section 4.3 says 1e-4 |
| Loss | Eq. 6 literally: half the squared error summed over classes, batch mean | paper equation; upstream's element-mean MSE is 7.5 times smaller |
| Seed, init | seed 1, PyTorch default initialisation | upstream default; paper silent |
| Checkpoint | final epoch | paper silent |

The max-pooling factor was added after subset preflights showed a silent network under average pooling; the user approved the 2x2 grid before launch. All four runs are reported and none is selected as the result.

**Deviations.** One Modal A10G instead of the paper's unspecified "TITAN server"; the study base image (NGC PyTorch 26.04, torch 2.12.0a0) instead of the paper's unspecified PyTorch version.

## Runs and evidence

| Run | Modal app | Start (UTC) | End (UTC) | GPU-hours |
| --- | --- | --- | --- | ---: |
| `full-sgd-avg` | [`ap-DnSVh6V2F2RHUhEUi6tFTC`](https://modal.com/apps/repro-sign/main/ap-DnSVh6V2F2RHUhEUi6tFTC) | 2026-10-02 13:42:20 | 19:45:13 | 6.05 |
| `full-sgd-max` | [`ap-WTlc4IOF2DwCdf676nDm2Y`](https://modal.com/apps/repro-sign/main/ap-WTlc4IOF2DwCdf676nDm2Y) | 2026-10-02 13:42:22 | 19:46:48 | 6.07 |
| `full-adam-avg` | [`ap-5ccCQryAYzWKX8HCM4c5sY`](https://modal.com/apps/repro-sign/main/ap-5ccCQryAYzWKX8HCM4c5sY) | 2026-10-02 13:42:24 | 19:54:29 | 6.20 |
| `full-adam-max` | [`ap-TkDCGaAlnetQvgIltvsD2R`](https://modal.com/apps/repro-sign/main/ap-TkDCGaAlnetQvgIltvsD2R) | 2026-10-02 13:42:26 | 19:48:20 | 6.10 |

- **Compute:** Modal profile `repro-sign`, environment `main`, one A10G, 4 CPU, 16 GiB per run; peak GPU memory 6.0 GB; about 107 s per epoch. Total 24.42 GPU-hours; about USD 35 at list rates as an upper estimate (the day's billing report was not yet available).
- **Stop policy:** declared 13:42:18 UTC, before launch: 12 h wall time (the function timeout), 12 GPU-hours and CHF 15 per run, one resume allowed. No run was interrupted or resumed.
- **Estimate vs actual:** the preflight estimate was 5.5 h per run (22 GPU-hours); actual was 6.1 h per run, so the total ended slightly above 24 GPU-hours.
- **Artifacts:** `run.json` (config, counts, final metrics, per-file predictions), `metrics.jsonl` (per-epoch loss and accuracy), `freeze.txt` and `checkpoint.pt` for each run under `modal://volume/chen-2023-event-camera-snn-sign-results/{sgd-avg,sgd-max,adam-avg,adam-max}/`; SHA-256 of each is in `reproduction.json`. The two SGD `metrics.jsonl` files are identical because both runs were silent.
- **Not captured:** Modal function-call IDs and image ID (the detached clients' log streams ended early).
- **Preflights** (60 train / 60 test files, about 15 GPU-minutes): the path ran end to end; one resume failed on a checkpoint device bug that was then fixed and re-verified; every variant stayed silent except Adam with max pooling. No stop policy was declared for them, so they are documented under `preflights` in `reproduction.json` rather than as run entries.

## How to repeat

```bash
./setup.sh
papers/chen-2023-event-camera-snn-sign/data.sh
.agents/skills/reproduce-paper/scripts/check_modal_dataset.sh dvs-sign-v2e MANIFEST.sha256
for optimizer in sgd adam; do for pool in avg max; do
  .agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach \
    papers/chen-2023-event-camera-snn-sign/modal_app.py::train --optimizer "$optimizer" --input-pool "$pool"
done; done
```

`data.sh` is idempotent and populates `datasets/dvs-sign-v2e` from the pinned commit. `train` writes to `/results/{optimizer}-{pool}/`, checkpoints every epoch, resumes when re-invoked, and returns the existing result if the run is already complete.

## Open gates

| Gate | Needed |
| --- | --- |
| `protocol-unspecified-model-and-training` | the authors' code or exact configuration, or acceptance that the runs stay conditional |
| `acc1-acc2-partition` | a definition of the two test-set parts, or a decision to drop those targets |
| `davis346-recordings-unpublished` | a data request to the corresponding author through Team S, including consent and cloud-processing terms |

The data-permission gate `dvs-repo-no-license` is resolved (user confirmation, 2026-10-02).

## Author contact

Drafted 2026-10-05, after the independent attempt and at the user's request; **not yet sent**. The email to the corresponding author (li.su@cnu.edu.cn) is in `author-contact-draft.md`. It asks for the code or exact training configuration, the Acc1/Acc2 definition and the DAVIS346 recordings, and it reports the four conditional runs. The data request in it is to be coordinated with Team S. Nothing has changed as a result of the contact so far.
