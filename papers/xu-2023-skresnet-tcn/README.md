# Xu et al. (2023): Improved SKResNet-TCN

**Pipeline status:** `insufficient_information`

**Numerical agreement:** `not_assessed`

**Preference level:** 3

This attempt stops at a structural specification gate: `insufficient_information`, with numerical agreement `not_assessed`. The raw LSA64 data is available, but the paper does not specify a network whose parameter count, FLOPs, and recognition score can be faithfully measured. No replacement architecture was invented or trained.

Xu, Xuebin; Meng, Kan; Chen, Chen; Lu, Longbin. [Isolated Word Sign Language Recognition Based on Improved SKResNet-TCN Network](https://doi.org/10.1155/2023/9503961). *Journal of Sensors*, 2023, article 9503961.

The assignment is a direct user request using the current tracker export. Its original SHA256 and redacted record are preserved in `reproduction.json`. The current export has `expand.paper.status=final`, no confirmation field, and distinct paper/database IDs. GPT-6 using Codex performed the source investigation, data audit, and reporting. Session instructions establish those identities; exact model and harness versions were not exposed.

## Target scope

Table 4, p. 8 compares the proposed system with cited earlier systems. All fourteen numeric cells remain in the ledger. The proposed method’s three cells have a structural gate. The comparison cells have a provenance gate: citations alone do not establish whether each number was copied or remeasured, so none is silently exempted from scope. Earlier SPOTER Table 5 contains the same I3D 98.91 and MEMP 99.06 accuracy values, but the origins of the other parameter/FLOP/accuracy cells are not fully established.

| Table 4 system | Parameters (millions) | Computation (GFLOPs) | LSA64 accuracy (%) | Reproduced |
|---|---:|---:|---:|---|
| I3D | 12.39 | 111.60 | 98.91 | None produced; provenance unresolved |
| (2+1)D-SLR | 33.30 | Not reported | 98.20 | None produced; provenance unresolved |
| I3D + GLR + CSA | 13.40 | 111.79 | 100.00 | None produced; provenance unresolved |
| MEMP | 12.83 | 112.34 | 99.06 | None produced; provenance unresolved |
| Improved SKResNet-TCN (proposed) | 12.34 | 62.50 | 100.00 | None produced; structural specification unresolved |

## What was resolved

Sections 3–4 specify 32 keyframes, interframe difference maxima, grouped selective-kernel spatial convolutions, temporal causal convolutions, hybrid dilation, adaptive max pooling, Mish, Ranger, learning rate 0.0001, batch 128, and a 60/20/20 split. Table 1 and §4.1 specify 1,000 iterations; Figure 7 discussion identifies best test performance at 656. These details alone do not define the executable architecture or a reproducible evaluation split.

The unresolved specification consists of:

- SKResNet stage depths, widths, grouped-convolution groups, and initialization; TCN widths, depth, kernel sizes, and actual hybrid dilation sequence.
- Input image dimensions and the parameter/FLOP counting convention. These directly determine the two efficiency targets.
- Keyframe smoothing and the procedure for reducing/padding each video to 32 frames; exact 60/20/20 split and selection seed; whether “iterations” means optimizer steps or epochs; validation versus test checkpoint selection.

Selecting arbitrary values and adjusting them until 12.34M/62.50G match would be tuning toward a reported result. A conditional surrogate would need a declared specification and would not become comparable merely because its accuracy happened to be close.

## Source search

The complete publisher article, supplied PDF, captions, equations, references, and Data Availability were inspected. No implementation or supplement defining the architecture is linked. GitHub repository searches for `SKResNet-TCN` and `9503961` returned zero results; exact-title/model/author searches found the paper and citations without author code. This is evidence of the search performed, not proof that code has never existed.

The cited Li et al. 2019 SKNet paper links `https://github.com/implus/SKNet`, which returned 404. Other SKNet implementations exist but cannot establish this hybrid architecture. The cited [Bai et al. TCN source](https://github.com/locuslab/TCN/tree/2f8c2b817050206397458dfd1f5a25ce8a32fe65) accepts an arbitrary channel list and kernel size; its powers-of-two dilation recipe does not define Xu's proposed hybrid sequence. It was inspected, not executed or copied.

## Data, execution, and repeat commands

The official [LSA64 source](https://facundoq.github.io/datasets/lsa64/) identifies 3,200 videos, 64 classes, 10 nominal signer IDs, and 5 repetitions. The raw release is 1920 × 1080 at 60 fps. Its CC BY-NC-SA 4.0 terms permit academic processing and require attribution/share-alike for derived data. Only existing project-cloud data is read; this PR redistributes no video. No new participants or author contact were involved.

The canonical v2 `datasets` Volume contains `lsa64`, with source archive SHA256 `218197acaa188583c1f06d149750af6af0d6b2bd44a627550d55773f5eefb20e`. The audit checks 3,200 paths / 64 classes, samples one video per class, hashes it, and decodes three frames through `simple-video-utils 0.7.4`. It does not create an undocumented train/test split.

```bash
./setup.sh
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/xu-2023-skresnet-tcn/modal_app.py --run-id data-preflight-004
python3 .agents/skills/reproduce-paper/scripts/validate_reproduction.py papers/xu-2023-skresnet-tcn
```

The retained run ID is immutable; use a fresh ID when repeating. The CPU container uses Python 3.12, `simple-video-utils 0.7.4`, and `av 18.0.0`. Datasets are read-only; shared v2 `huggingface-cache` is mounted at `/cache/huggingface`; outputs use `2be3c68e-skresnet-results`. No GPU was requested. Each diagnostic was bounded at 900 seconds / CHF 1. Actual billed cost is unavailable.

Three cheap execution failures were preserved: newest PyAV 19 rejects the decoder's `metadata_errors` argument; trying PyAV 16 was rejected by the package's `av>=18` constraint; PyAV 18 decoded successfully but the audit serializer incorrectly assumed a dataclass. The final script pins supported PyAV 18 and reads the documented metadata attributes. These are data-audit environment/glue changes, not changes to the proposed model.

The retained terminal evidence, timestamps, Modal app/function IDs, exact manifest/counts, dependencies, and hashes are in `reproduction.json`. Setup and the data audit are repeatable. Training and evaluation cannot be made faithful until the architecture gate is resolved; there are no pretend training entry points or invented metrics.

## Open question

Which architecture/configuration and input shape produced Table 4, and which keyframe/split/checkpoint protocol produced its 100% accuracy? An author-released implementation or explicit specification can resolve this. No author was contacted. This question requires evidence rather than an optimizer default; Ranger is already specified.

Preference level: 3. No faithful reimplementation can be selected before the structural gate is resolved. The successful CPU audit took 26.15 seconds, verified all 3,200 file paths, and decoded 192 full-resolution frames across 64 class samples. Raw evidence: `modal://2be3c68e-skresnet-results/data-preflight-004/`, app `ap-Q5jTR7jRU1BbyHlNewufsy`, call `fc-01M3VGW17RQMT228VPF0C8Q3Y0`. GPU-hours: 0.

Comparison-row question: which Table 4 cells were copied from which exact source locations, and which were remeasured with what input shape, split, and checkpoint? Until resolved, these cells are explicitly not produced rather than excluded as verified copied baselines.
