# Video-Based Sign Language Recognition via ResNet and LSTM Network

**Paper ID:** `990030f8dfefb06e99c05218741e11ccf7b08fdb`
**Citation:** Huang, J.; Chouvatut, V. *Journal of Imaging* 10(6), 149 (2024). https://doi.org/10.3390/jimaging10060149
**Preference level:** 3

Clean-room implementation; no author code was found.

**Status:** `insufficient_information`
**Numerical agreement:** `not_assessed`

## Target

The requested Table 4 contribution is the paper's ResNet-LSTM row: 86.25% accuracy. Table 3 identifies the same epoch-30, batch-16 result and also reports 84.98% F1 and 87.77% precision. These are the three targets. The other Table 4 rows are cited baselines and are out of scope.

| Metric | Paper (Table 3) | Conditional frozen-run result | Difference |
| --- | ---: | ---: | ---: |
| Accuracy | 86.25% | 76.2500% | -10.0000 |
| F1 score | 84.98% | 75.6829% | -9.2971 |
| Precision | 87.77% | 82.3100% | -5.4600 |

The conditional result is not entered as a reproduced target result: the paper does not identify its 8/2 signer split, so its numerical agreement with Table 3 cannot be assessed.

## What the paper specifies

| Item | Evidence |
| --- | --- |
| Model | ImageNet-pretrained ResNet-18 frame encoder, LSTM, then classifier (§§3.2, 3.5) |
| Frozen encoder | The pretrained hidden-layer parameters are retained/frozen (§4.4) |
| Video input | 16 RGB frames, 128×128, random crop (§§4.1, 4.4) |
| Data size | 3,200 clips; 2,560 train / 640 validation (§§4.2, 4.5) |
| Split description | 8 learners train, 2 learners test (§4.2); identities are not given |
| Target row | 30 epochs, batch 16 (§4.5, Table 3) |
| Optimizer/loss | Adam, LR 1e-4, weight decay 5e-4, label-smoothed cross entropy (§4.4) |
| Schedule | ReduceLROnPlateau, threshold 1e-4, patience 5, factor 0.1 (§4.4) |

The paper is internally inconsistent: §4.4 says Adam, while Table 2 says SGD; the reconstruction follows the detailed implementation prose. The paper describes 50-epoch experiments, but Table 3 explicitly reports the requested 30-epoch row.

## Reconstruction decisions and gates

These choices are not presented as paper facts:

- The paper does not publish signer IDs or split files. `(5, 10)` is only an inferred 8/2 convention, not an author split. It is further problematic because LSA64 says subject 10 was replaced between its two disjoint recording sessions. A result on this split is a conditional reconstruction, not a verified reproduction.
- The paper does not specify frame-selection positions, resize policy, normalization constants, the modified fully-connected/LSTM width or LSTM depth, seed, or the scheduler monitor. The code uses uniform 16-frame sampling, resize-short-side 144 then one 128 crop, ImageNet normalization, a trainable 512-unit frame projection followed by one 512-unit LSTM, seed 2024, and training loss for the scheduler. Training loss prevents the held-out clips from controlling optimization.
- The official LSA64 page currently licenses the data CC BY-NC-ND 4.0 for strict academic use and prohibits derivative works. The user authorized this private project-cloud run; no dataset, predictions, or new weights are committed or distributed. An earlier invalid checkpoint was uploaded before this review; it is not reproduction evidence and must be privatized or removed pending permission to publish learned weights.

The unresolved signer identities and license/publishing question are open gates, so no target-level numerical result is claimed here. Earlier fine-tuned and purportedly frozen results are excluded: the retained code did not enforce the paper's encoder freeze, and the fine-tuned run also drove its LR schedule from the reported holdout.

## Conditional full run

The structured run policy was backfilled from the pre-migration report (whose
SHA-256 is retained in `reproduction.json`). The committed 24-hour launcher
ceiling is preserved; exact process boundaries, declaration time, retry maximum,
GPU-hour ceiling, and cost ceiling remain explicit unknowns.

The corrected frozen-backbone recipe completed successfully on Modal `repro-sign`: one A10G, seed 2024, fixed 30 epochs and batch size 16. It froze ResNet-18 parameters and BatchNorm state, kept the modified 512-unit frame projection/LSTM/classifier trainable, scheduled only from training loss, and evaluated the inferred 640-clip holdout once at epoch 30. The terminal evidence is app `ap-JrUpXgewdIOzCRYscgQhQQ`, function call `fc-01M0SQ06EG6Q7XKQWTZYBC8NR0`, and container `ta-01M0SQ076YFW7VXP9X20ZFQD7R`.

Raw metrics: `metrics.jsonl` SHA-256 `d776f1db74429596f77c676fd5d4740643ad19e021ab115e4f1a81a0692dc169`; run manifest SHA-256 `81bde9d05e7c8616a11b3a605f17073ba31b9b9c7098e0b23a1fd1376a3c8b0b`. The epoch-30 checkpoint remains only in the private results Volume and is not published.

## Excluded fine-tuning observation

An earlier 30-epoch run on the same inferred split reached 99.6875% held-out accuracy when it **unfroze the ImageNet ResNet-18 encoder**. It is retained here solely as a diagnostic observation: it is not a paper-faithful setting, not a reproduced target result, and not evidence for numerical agreement. It conflicts with §4.4's instruction to retain the pretrained hidden-layer parameters; that earlier run also used the held-out score to control its learning-rate schedule. The frozen result above is therefore the only retained conditional reconstruction.

## How to run the conditional reconstruction

The user authorized this private project-cloud run. The split remains a conditional reconstruction, and the separate trained-weight publication gate remains open. Authenticate to Modal workspace `repro-sign` and run:

```bash
./setup.sh
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run \
  papers/huang-chouvatut-2024-resnet-lstm/modal_app.py::populate_lsa64
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach \
  papers/huang-chouvatut-2024-resnet-lstm/modal_app.py::train
```

The data command verifies/populates `datasets/lsa64`. Training builds from the root study image, which supplies `simple-video-utils==0.7.4`, uses the shared `huggingface-cache`, and writes only to the paper-specific Modal results Volume. It runs a fixed 30-epoch, batch-16 recipe; the final held-out evaluation is performed once, at epoch 30, with no best-checkpoint selection.

## Provenance

| Artifact | Source | Evidence | Use |
| --- | --- | --- | --- |
| Paper | [Publisher PDF](https://mdpi-res.com/d_attachment/jimaging/jimaging-10-00149/article_deploy/jimaging-10-00149-v2.pdf?version=1719278950) | SHA-256 `01306636261ad2be13436204deab1b0bf04c7a732a654d2f03587c5f97edb4bb` | Targets and protocol |
| Dataset | [Official LSA64 release](https://midusi.github.io/lsa64/) | Original archive SHA-256 `218197acaa188583c1f06d149750af6af0d6b2bd44a627550d55773f5eefb20e` | 3,200 original RGB MP4s |

All video decoding uses `simple-video-utils==0.7.4`. The dataset manifest on the shared `datasets` Volume records the source archive, paths, and 3,200-clip count.
