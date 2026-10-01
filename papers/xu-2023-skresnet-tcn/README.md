# Xu et al. (2023): Improved SKResNet-TCN

**Pipeline status:** `insufficient_information`

**Numerical agreement:** `not_assessed`

**Preference level:** 3

The exact-target attempt has a structural specification limit: `insufficient_information`, with numerical agreement `not_assessed`. The raw LSA64 data is available, but the paper does not specify a network whose parameter count, FLOPs, and recognition score can be faithfully measured. A separately declared conditional reconstruction completed all 1,000 optimizer steps and evaluation; it cannot identify the missing author architecture.

Xu, Xuebin; Meng, Kan; Chen, Chen; Lu, Longbin. [Isolated Word Sign Language Recognition Based on Improved SKResNet-TCN Network](https://doi.org/10.1155/2023/9503961). *Journal of Sensors*, 2023, article 9503961.

The assignment is a direct user request using the current tracker export. Its original SHA256 and redacted record are preserved in `reproduction.json`. The current export has `expand.paper.status=final`, no confirmation field, and distinct paper/database IDs. GPT-6 using Codex performed the source investigation, data audit, and reporting. A separate master agent, also identified as GPT-6 using Codex, independently reviewed rendered architecture figures and comparison provenance; it did not execute the original CPU audit runs. The conditional extension below has separate implementation and execution attribution. Session instructions establish those identities; exact model and harness versions were not exposed.

## Target scope

Table 4, p. 8 compares the proposed system with cited earlier systems. All fourteen numeric cells remain in the ledger. The proposed method’s three cells cannot be attributed to an identifiable author architecture. The comparison cells have a provenance gate: citations alone do not establish whether each number was copied or remeasured, so none is silently exempted from scope. Earlier SPOTER Table 5 contains the same I3D 98.91 and MEMP 99.06 accuracy values, but the origins of the other parameter/FLOP/accuracy cells are not fully established.

| Table 4 system | Parameters (millions) | Computation (GFLOPs) | LSA64 accuracy (%) | Reproduced |
|---|---:|---:|---:|---|
| I3D | 12.39 | 111.60 | 98.91 | None produced; provenance unresolved |
| (2+1)D-SLR | 33.30 | Not reported | 98.20 | None produced; provenance unresolved |
| I3D + GLR + CSA | 13.40 | 111.79 | 100.00 | None produced; provenance unresolved |
| MEMP | 12.83 | 112.34 | 99.06 | None produced; provenance unresolved |
| Improved SKResNet-TCN (proposed) | 12.34 | 62.50 | 100.00 | None produced; structural specification unresolved |

## What was resolved

Sections 3–4 specify 32 keyframes, interframe difference maxima, grouped selective-kernel spatial convolutions, temporal causal convolutions, hybrid dilation, adaptive max pooling, Mish, Ranger, learning rate 0.0001, batch 128, and a 60/20/20 split. Table 1 and §4.1 specify 1,000 iterations; Figure 7 discussion identifies best test performance at 656. These details alone do not define the executable architecture or a reproducible evaluation split.

An independent review of rendered Figures 2–4 confirmed that they use schematic channel/shape variables and illustrative dilations, without providing actual stage widths, depths, or input resolution.

The unresolved specification consists of:

- SKResNet stage depths, widths, grouped-convolution groups, and initialization; TCN widths, depth, kernel sizes, and actual hybrid dilation sequence.
- Input image dimensions and the parameter/FLOP counting convention. These directly determine the two efficiency targets.
- Keyframe smoothing and the procedure for reducing/padding each video to 32 frames; exact 60/20/20 split and selection seed; whether “iterations” means optimizer steps or epochs; validation versus test checkpoint selection.

Selecting arbitrary values and adjusting them until 12.34M/62.50G match would be tuning toward a reported result. The conditional reconstruction below has a declared specification and does not become comparable merely because its accuracy happens to be close.

## Source search

The complete publisher article, supplied PDF, captions, equations, references, and Data Availability were inspected. No implementation or supplement defining the architecture is linked. GitHub repository searches for `SKResNet-TCN` and `9503961` returned zero results; exact-title/model/author searches found the paper and citations without author code. This is evidence of the search performed, not proof that code has never existed.

The cited Li et al. 2019 SKNet paper links `https://github.com/implus/SKNet`, which returned 404. Other SKNet implementations exist but cannot establish this hybrid architecture. The cited [Bai et al. TCN source](https://github.com/locuslab/TCN/tree/2f8c2b817050206397458dfd1f5a25ce8a32fe65) accepts an arbitrary channel list and kernel size; its powers-of-two dilation recipe does not define Xu's proposed hybrid sequence. The conditional extension reuses its pinned TemporalBlock implementation with explicitly declared widths and dilations.

## Data, execution, and repeat commands

The official [LSA64 source](https://facundoq.github.io/datasets/lsa64/) identifies 3,200 videos, 64 classes, 10 nominal signer IDs, and 5 repetitions. The raw release is 1920 × 1080 at 60 fps. Its CC BY-NC-SA 4.0 terms permit academic processing and require attribution/share-alike for derived data. Only existing project-cloud data is read; this PR redistributes no video. No new participants or author contact were involved.

The canonical v2 `datasets` Volume contains `lsa64`, with source archive SHA256 `218197acaa188583c1f06d149750af6af0d6b2bd44a627550d55773f5eefb20e`. The audit checks 3,200 paths / 64 classes, samples one video per class, hashes it, and decodes three frames through `simple-video-utils 0.7.4`. The original audit did not create a train/test split; the later extension records an explicit seeded split.

```bash
./setup.sh
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/xu-2023-skresnet-tcn/modal_app.py --run-id data-preflight-004
python3 .agents/skills/reproduce-paper/scripts/validate_reproduction.py papers/xu-2023-skresnet-tcn
```

The retained run ID is immutable; use a fresh ID when repeating. The CPU container uses Python 3.12, `simple-video-utils 0.7.4`, and `av 18.0.0`. Datasets are read-only; shared v2 `huggingface-cache` is mounted at `/cache/huggingface`; outputs use `2be3c68e-skresnet-results`. No GPU was requested for these original audits. Each diagnostic was bounded at 900 seconds / CHF 1. Actual billed cost is unavailable.

Three cheap execution failures were preserved: newest PyAV 19 rejects the decoder's `metadata_errors` argument; trying PyAV 16 was rejected by the package's `av>=18` constraint; PyAV 18 decoded successfully but the audit serializer incorrectly assumed a dataclass. The final script pins supported PyAV 18 and reads the documented metadata attributes. These are data-audit environment/glue changes, not changes to the proposed model.

The retained terminal evidence, timestamps, Modal app/function IDs, exact manifest/counts, dependencies, and hashes are in `reproduction.json`. Setup and the data audit are repeatable. The conditional training entry point below produces evidence for its declared architecture. It does not identify the missing author architecture or silently promote its measurements to exact-paper targets.

## Open question

Which architecture/configuration and input shape produced Table 4, and which keyframe/split/checkpoint protocol produced its 100% accuracy? An author-released implementation or explicit specification can resolve this. No author was contacted. This question requires evidence rather than an optimizer default; Ranger is already specified.

Preference level: 3. No exact author implementation was recovered. The original successful CPU audit took 26.15 seconds, verified all 3,200 file paths, and decoded 192 full-resolution frames across 64 class samples. Raw evidence: `modal://2be3c68e-skresnet-results/data-preflight-004/`, app `ap-Q5jTR7jRU1BbyHlNewufsy`, call `fc-01M3VGW17RQMT228VPF0C8Q3Y0`. GPU-hours: 0.

The [original (2+1)D-SLR abstract](https://link.springer.com/article/10.1007/s00521-021-06467-9) reports 98.7% LSA64 accuracy, whereas Xu Table 4 prints 98.2%. This unresolved discrepancy supports retaining the comparison provenance gate.

Comparison-row question: which Table 4 cells were copied from which exact source locations, and which were remeasured with what input shape, split, and checkpoint? Until resolved, these cells are explicitly not produced rather than excluded as verified copied baselines.

## Documented judgment calls and conditional extension

A conditional reconstruction completed without a pending human execution decision. It uses timm 1.0.22's original-SKNet-equivalent `skresnext50_32x4d`, random initialization, 224 × 224 RGB input, Mish, and max spatial pooling. Three cited TCN blocks use width 256, kernel 3, dilations 1/2/5 and dropout 0.2, followed by temporal max pooling and a 64-class classifier. These choices were fixed before scores and were not adjusted to match the reported parameter or FLOP totals.

Preprocessing uses bilinear resizing and mean absolute grayscale frame differences smoothed over three frames. It selects the 32 strongest local maxima in chronological order, fills shortages uniformly, and repeats the last frame if necessary. Seed 42 per-class 60/20/20 allocation gives 1,920 training, 640 validation and 640 test videos. All 3,200 videos were processed without exceptions; the manifest preserves source and derived hashes, frame indices, original frame counts and split membership. Its SHA256 is `d04e5c9b682f00eaa21cbfaa9df6aa62f7a7f45fcda788aca820acb3ad5b6e92`.

Training interprets 1,000 iterations as optimizer steps, with Ranger 0.3.0 at learning rate 0.0001 and other library defaults. Effective batch 128 is accumulated over 32 microbatches of four videos. BatchNorm therefore sees 128 frames per microbatch, an explicit deviation. Validation every 50 steps selects the earliest maximum-accuracy checkpoint. RGB inputs are scaled to [0,1] and normalized with ImageNet means and standard deviations, despite random initialization. Whole-video horizontal flipping has probability 0.5 during training and the primary test, reflecting the paper's test-inversion statement. Validation is unaugmented; a deterministic test is secondary. Primary test seed 314159 was declared before scores.

Atomic training checkpoints every 60 steps coincide with complete training epochs. They preserve model, Ranger optimizer, random generators and the best validation model. The seven-step preflight exercised Ranger's six-step Lookahead synchronization, checkpoint reload and an actual additional optimizer step. It measured 27.35 GB peak GPU memory and 0.265 seconds per warm microbatch on an A100 80GB. Its tiny held-out scores are diagnostics only. The constructed model has **28,530,752 parameters** and **143,947,126,784 counted operations** under fvcore's one-fused-multiply-add convention; unsupported operators are retained in the raw metric artifact. These conditional values differ from the paper's architecture totals and were not used to alter the model.

CPU preparation completed in 340.37 seconds (`ap-JEMJetvAmhcr5mOTlHkVLv`, `fc-01M3VS6NHV4N2BSFCQA353JJ4N`). The GPU preflight completed in 141.50 seconds including wrapper overhead (`ap-f0q0RR1sqr3taTgKGauuAq`, `fc-01M3VSQ25HF8PSQGFGKW8WQ02R`). The full run is `conditional-full-seed42-001`, app `ap-0zyXHBpA6HkvCxedlnkv2E`, with a conservative 3.5 GPU-hour / CHF 11 forecast and an aggregate ceiling of five GPU-hours / CHF 20. The one permitted checkpoint-only resume was consumed by provider preemption; no full restart occurred.

The GPU image is pinned to `ghcr.io/sign-language-processing/reproduction@sha256:305b6165d306192996358ca312d9a751fa409f43063a76dc7758880a8f905291`, with timm 1.0.22, torch-optimizer 0.3.0 and fvcore 0.1.5.post20221221. The shared dataset and cache use VolumeFS v2. Training mounts `datasets` read-only; only the controlled preparation step writes the derived `lsa64-skresnet-conditional-v1` directory. Outputs, source snapshots, dependency freezes and native logs remain on `2be3c68e-skresnet-results` with individual hashes in the ledger.

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/xu-2023-skresnet-tcn/scripts/conditional_modal.py --mode prepare --run-id conditional-data-001
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/xu-2023-skresnet-tcn/scripts/conditional_modal.py --mode preflight --run-id conditional-preflight-001
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/xu-2023-skresnet-tcn/scripts/conditional_modal.py --mode full --run-id conditional-full-seed42-001
```

Existing successful run IDs return their recorded execution result; use fresh IDs for a new attempt. Preparation verifies and reuses the existing derived manifest. GPT-6 using Codex implemented and executed this extension; a separate GPT-6/Codex agent reviewed the resume and evidence logic without executing its runs. Exact model and application version identifiers were not exposed. Original source-investigation and audit attributions remain separate.

All extension measurements remain conditional evidence. The original architecture and comparison provenance remain scientifically unknown, while ordinary execution choices have been resolved and documented.

A second independent GPT-6/Codex reviewer checked source/report consistency, effective-batch scaling, checkpoint selection and epoch-boundary resume without editing or executing the experiment. The actual resume was constrained to the remaining aggregate five-hour / CHF 20 budget by an external guard; the wrapper's default per-invocation timeout does not track aggregate consumption. Primary prediction artifacts contain class labels and source indices, not logits. Secondary evaluation has a raw metric but no saved per-example predictions, so no independent secondary recomputation is claimed.

## Provider preemption and recovery

Modal preempted the original full-run container at 14:45:04 UTC on 2026-10-01 and automatically replayed the same invocation. The new container restored step 480 at 14:45:12 UTC. This consumes the one declared checkpoint resume; it is not a new training run from initialization. An external guard was configured to stop a third container or stop at 18:20:18 UTC, preserving the original five-hour aggregate ceiling. The second container completed normally before that deadline, and the guard confirmed termination. Model, optimizer and RNG state were restored, but bitwise equivalence across containers is not established: replayed validation at step 500 differed.

The original wrapper overwrote its console/runtime files on replay. Retained partial native snapshots and the platform preemption log are preserved under `artifacts/`; the original native exit code and complete original console are unavailable. Final accounting includes the original container interval and the resumed interval, while process-local metric timings describe only the resumed process.

## Full conditional result

The full run completed with **353/640 = 55.15625% primary test accuracy**. The separate deterministic test was 54.68750%. The paper reports 100%, but these measurements are conditional because the exact author architecture is unavailable; numerical agreement remains `not_assessed` for the target contract. The earliest highest-validation checkpoint was step 1000, selected without inspecting the test scores. All 1,000 logical optimizer steps and 20 retained scheduled validation entries completed, with one provider-triggered checkpoint resume from step 480 and no configuration search. Lost work since that checkpoint was repeated; bitwise cross-container equivalence is not established.

The reconstructed architecture has 28,530,752 parameters and 143.947 billion counted operations under the disclosed one-FMA convention, versus the paper's 12.34M / 62.50 GFLOPs. Unsupported operations are listed in the raw metrics. These counts describe different, incompletely specified architectures and are not promoted to produced targets.

Native run: `ap-0zyXHBpA6HkvCxedlnkv2E`, call `fc-01M3VSZ2HFDKYVJ06QRGDJZSJ7`; 2026-10-01T13:20:34+00:00 to 2026-10-01T16:04:39.988034+00:00, with an eight-second interruption. Aggregate container time was 9835.16 seconds, or 2.73199 A100 80GB GPU-hours. Estimated cost is CHF 8.59, prorated from the prospective estimate rather than a billing record; build/startup are excluded. The original wrapper’s recovered finalization record spans 13:20:35.885595–14:44:56.522101 UTC; the conservative first-container charge remains 5,070 seconds from platform timestamps. Its recorded code 1 is the wrapper’s initialized value on interruption, not proof of a native child exit status. The resumed wrapper took 4765.16 seconds; native metric timing fields cover that resumed process only. Peak allocated GPU memory was 27.457 GB.

Independent SHA-256 and byte-count inspection verified every terminal artifact URI; the two checkpoints were streamed without local storage after checking framed CLI stdout against two independently downloaded small files. The CLI appends a fixed success message to stdout; the audit validated and excluded that exact 41-byte suffix before hashing payloads. All byte counts also agree with the terminal volume listing at its displayed precision. Independent local recomputation verified all 640 unique test indices against the immutable split manifest, the correct count, accuracy and prediction checksum. Checkpoint, dependency, hardware, source and metric hashes are recorded in `reproduction.json`; large artifacts stay at `modal://2be3c68e-skresnet-results/conditional-full-seed42-001/`. The final execution hash is also taken after the wrapper closes; the metrics file sampled the first-wrapper record, preserved separately in Git before overwrite. The log hash in the ledger is taken after the log closes: the metric file's embedded log hash was sampled before its own final print and describes that earlier snapshot. No training behavior changed to resolve this reporting detail. The earlier conditional-independent-reviewer (GPT-6 using Codex, exact versions unavailable) took over terminal monitoring, evidence collection and this independent numerical/report audit; the original master agent remains the training executor in run attribution.
