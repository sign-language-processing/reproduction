---
license: cc-by-nc-sa-4.0
library_name: pytorch
tags:
  - sign-language-recognition
  - reproduction
datasets:
  - lsa64
---

# Video-Based Sign Language Recognition via ResNet and LSTM Network

**Paper ID:** `990030f8dfefb06e99c05218741e11ccf7b08fdb`

**Citation:** Huang, J. and Chouvatut, V. *Journal of Imaging* 10(6), 149 (2024). https://doi.org/10.3390/jimaging10060149

**Preference level:** `3` — a clean-room implementation, because no executable author code was found.

**Pipeline status:** `complete` for the initial protocol; a documented fine-tuning follow-up is running.

**Numerical agreement:** not fully reproduced

**Attempt date:** 2026-08-23

## Scope and target contract

The queue requests Table 4. Its only new score is the paper's ResNet-LSTM row: 86.25% accuracy on LSA64. Table 3 additionally reports that same epoch-30, batch-16 run's F1 (84.98%) and precision (87.77%), so all three are evaluated here. The other Table 4 rows are cited baselines, not results of this paper, and are not rerun.

The paper specifies 16 RGB frames at 128x128, batch 16, 30 epochs, ImageNet pre-training, Adam at 1e-4, weight decay 5e-4, label-smoothed cross entropy, and a 5-epoch `ReduceLROnPlateau` patience. It does not publish code, seed, held-out signer IDs, LSTM width/layers, or the exact resize/crop/normalization. The implementation records those unavoidable choices below.

## Current result

| Target | Paper | Initial reproduced | Difference |
| --- | ---: | ---: | ---: |
| Accuracy | 86.25 | 78.5937 | -7.6563 |
| Macro F1 | 84.98 | 76.9026 | -8.0774 |
| Macro precision | 87.77 | 80.9082 | -6.8618 |

The initial result used a frozen ImageNet encoder, an interpretation the paper does not state. The current follow-up fine-tunes that same ResNet-18 + LSTM end-to-end; it changes no architecture, data, split, frame count, or reported training duration. The PR will retain only the protocol that the evidence supports.

## How to repeat

From a checkout authenticated to Modal profile/workspace `repro-sign`:

```bash
./setup.sh
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run \
  papers/990030f8dfefb06e99c05218741e11ccf7b08fdb/modal_app.py::populate_lsa64
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run \
  papers/990030f8dfefb06e99c05218741e11ccf7b08fdb/modal_app.py::train
```

The first command idempotently verifies/populates `datasets/lsa64`; the second writes `run.json`, `metrics.jsonl`, and checkpoints to the v2 Modal Volume `huang-chouvatut-2024-results`. The data volume is mounted at `/datasets`; the shared `huggingface-cache` volume is mounted at `/cache/huggingface`.

## Provenance, data, and implementation

| Artifact | Source | Pin / evidence | Use |
| --- | --- | --- | --- |
| Paper | [Publisher PDF](https://mdpi-res.com/d_attachment/jimaging/jimaging-10-00149/article_deploy/jimaging-10-00149-v2.pdf?version=1719278950) | SHA-256 `01306636261ad2be13436204deab1b0bf04c7a732a654d2f03587c5f97edb4bb` | Targets and protocol |
| Dataset | [Official LSA64 release](https://facundoq.github.io/datasets/lsa64/) | Original archive SHA-256 `218197acaa188583c1f06d149750af6af0d6b2bd44a627550d55773f5eefb20e` | 3,200 RGB MP4s |

LSA64 is licensed CC BY-NC-SA 4.0 for academic, educational, and personal use. This non-commercial research run uses the original release in project cloud storage only; no video, frame, prediction, or participant data is committed or published. The volume manifest records 3,200 videos and source/archive identity. All decoding uses `simple-video-utils==0.7.4`.

The clean-room model is standard ImageNet-initialized ResNet-18 frame features, a one-layer 512-unit LSTM, and a 64-way classifier. The unknown signer split is held-out signers 5 and 10, an explicit inference from the cited signer-independent LSA64 comparison. Frames are sampled uniformly across each video; a temporally consistent 144-short-side resize, 128 crop, and ImageNet normalization are used. These choices, the missing author executable, and the result mismatch limit scientific interpretation.

## Execution evidence

| Run | Platform | Result | Raw evidence |
| --- | --- | --- | --- |
| `frozen-encoder-epoch30` | Modal `repro-sign`, one A10G, seed 2024, 3.69 GPU-h, exit 0 | 78.5937% accuracy at epoch 30 | `modal://volume/huang-chouvatut-2024-results/resnet-lstm/metrics.jsonl` |

The initial run is retained as a failed protocol hypothesis, not a score-selected result. No author implementation, author checkpoint, or usable supplement was found after checking the paper, publisher/PMC record, author pages, and the two user-supplied LSA64 repositories; the latter are unrelated implementations and were not used.
