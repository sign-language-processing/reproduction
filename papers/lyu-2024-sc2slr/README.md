# SC2SLR (Lyu, CNIOT 2024) reproduction

**Paper ID:** `3ccdb5012eae349ef80750678cbe83dd9f51b868`

**Citation:** Silu Lyu. 2024. SC2SLR: Skeleton-based Contrast for Sign Language Recognition. In *2024 5th International Conference on Computing, Networks and Internet of Things (CNIOT 2024)*, Tokyo, Japan. ACM, pp. 404–410.

**Paper:** https://doi.org/10.1145/3670105.3670173 · **Code/artifacts:** https://github.com/christianlsl/Skeleton-based-Contrast-for-Sign-Language-Recognition at `65c95fb03fa0c170dc3f1136265f4b97ec26d7fa` (incomplete, see below)

**Preference level:** 3

**Pipeline status:** `insufficient_information` — the full pipeline ran and emitted all four numbers, but they rest on invented behaviour-changing choices that the reviewer did not accept as the protocol (gate `sc2slr-conditional-protocol-acceptance`); blocker `protocol_ambiguous` over the four SC2SLR target IDs

**Numerical agreement:** `not_assessed` — no target is produced; the conditional values are reported below

**Attempt dates:** 2026-10-26 to 2026-10-29

## Reproduction agents

| Agent ID | Model and version | Agent application | Contribution | Attribution evidence / unknowns |
| --- | --- | --- | --- | --- |
| `claude-code-opus-5-5` | Opus 5.5 (`claude-opus-5-5`) | Claude Code (version not exposed) | Every stage: ingestion, target ledger, source search, code audit, gates, patches, containers, `modal_app.py`, all Modal runs, evaluation, and this report | Session system context. `harness_version` is null because the application version is not exposed. Every run records `agent_ids: ["claude-code-opus-5-5"]`. |

## Scope and target contract

The tracker record asks for "Table 3", which compares continuous SLR WER (%) on PHOENIX-2014 and PHOENIX-2014T dev/test. The paper's own row, SC2SLR (Ours), yields four in-scope targets:

| Target ID | Dataset / split | Published WER (%) |
| --- | --- | ---: |
| `table3-sc2slr-p14-dev` | PHOENIX-2014 dev (540) | 20.8 |
| `table3-sc2slr-p14-test` | PHOENIX-2014 test (629) | 21.2 |
| `table3-sc2slr-p14t-dev` | PHOENIX-2014T dev (519) | 20.6 |
| `table3-sc2slr-p14t-test` | PHOENIX-2014T test (642) | 21.0 |

The other seven rows (SubUNet, CNN-LSTM-HMMs, CMA, STMC, SFL, VAC, C2SLR) are prior-work numbers cited by reference. They are recorded as 22 `copied_baseline` targets with `in_scope: false`, and were not rerun or checked against their source papers. References [14] and [15] in the paper are the same Koller et al. TPAMI 2019 paper.

The metric is corpus-level gloss WER, (S+I+D)/N (Eq. 9), scored by the released `evaluation/slr_eval` sclite pipeline (SCTK `9688a268`). The paper reports one number per cell, with no seeds or variance. Its PHOENIX-2014 values also equal the full-model rows of Tables 1 and 2.

## Results

All values are WER (%), lower is better. Each is from one 120-epoch run with seed 0. The checkpoint is the one with the lowest dev WER (earliest on ties), and dev and test were each scored once. This selection rule is a guess: the paper does not state one (see `checkpoint-selection` under Guesses and deviations).

| Target ID | Target | Published | Conditional value | Difference | Status | Evidence |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `table3-sc2slr-p14-dev` | PHOENIX-2014 dev | 20.8 | 22.7 | +1.9 | `not_produced` (`protocol_ambiguous`) | `eval-p14-2`, epoch 42 of `full-p14-2` |
| `table3-sc2slr-p14-test` | PHOENIX-2014 test | 21.2 | 22.8 | +1.6 | `not_produced` (`protocol_ambiguous`) | `eval-p14-2` |
| `table3-sc2slr-p14t-dev` | PHOENIX-2014T dev | 20.6 | 21.9 | +1.3 | `not_produced` (`protocol_ambiguous`) | `eval-p14t-2`, epoch 44 of `full-p14t-2` |
| `table3-sc2slr-p14t-test` | PHOENIX-2014T test | 21.0 | 23.8 | +2.8 | `not_produced` (`protocol_ambiguous`) | `eval-p14t-2` |

sclite breakdowns (sub / del / ins):

| Split | Sentences / glosses | Sub | Del | Ins |
| --- | --- | ---: | ---: | ---: |
| PHOENIX-2014 dev | 540 / 5540 | 12.7 | 6.1 | 4.0 |
| PHOENIX-2014 test | 629 / 6504 | 13.2 | 5.9 | 3.7 |
| PHOENIX-2014T dev | 519 / 3748 | 12.1 | 5.8 | 4.0 |
| PHOENIX-2014T test | 642 / 4264 | 13.6 | 5.6 | 4.6 |

**Why these are not marked produced.** The paper does not define the keypoint-heatmap rendering or the contrastive objective (temperature, which features, projection head, loss weight), and the released code contains neither. These were invented and are listed under Guesses and deviations. `status.conditional_determination` recorded before any training that such values are conditional evidence unless a human reviewer accepts the invented choices as the Table 3 protocol. The three invented choices that decide this are:

1. **Keypoint heatmap:** Gaussians with σ = 4 px (`heatmap-rendering`).
2. **Contrastive branch:** NT-Xent with τ = 0.5 and weight 1, applied to BiLSTM outputs mean-pooled over valid steps, with no projection head (`contrastive-branch`).
3. **PHOENIX-2014T recipe:** the PHOENIX-2014 recipe reused unchanged (`p14t-recipe`).

On 2026-10-29 the reviewer declined to accept them (gate `sc2slr-conditional-protocol-acceptance`, resolved). The four targets therefore stay `not_produced` / `protocol_ambiguous`, with the conditional runs, values, and differences kept above. The conditional determination was recorded before training; the reviewer's decision of 2026-10-29 confirmed it.

**Observation: dev WER after the first learning-rate drop** (`observations[0]` in `reproduction.json`). Both runs reached their dev minimum within five epochs of the drop at epoch 40, then rose:

| Run | Epoch 39 | Epoch 40 | Minimum | Epoch 119 | Mean, epochs 100–119 |
| --- | ---: | ---: | --- | ---: | ---: |
| PHOENIX-2014 (`full-p14-2`) | 25.7 | 23.0 | 22.7 (epoch 42) | 23.9 | 23.9 |
| PHOENIX-2014T (`full-p14t-2`) | 25.5 | 22.5 | 21.9 (epoch 44) | 22.5 | 22.7 |

Under this recipe, the 120-epoch schedule over-trains relative to dev. This is recorded only: nothing was relaunched or shortened, the 120 epochs follow the paper, and the final-epoch checkpoints were not scored on test. The per-epoch history is in `runs/full-p14{,t}-2/dev.txt` (artifacts `devlog-p14`, `devlog-p14t`).

The 22 copied baselines are terminal (`not_produced`, `copied_baseline`).

## Source provenance

| Artifact | Canonical source | Pinned revision / SHA-256 | Role |
| --- | --- | --- | --- |
| Paper PDF | https://doi.org/10.1145/3670105.3670173 | `2ad1879b2433ecf524bf3bb7a86be82d559a20b64aa4d244866590f1da86e62b` | Targets and protocol |
| Published code | https://github.com/christianlsl/Skeleton-based-Contrast-for-Sign-Language-Recognition | `65c95fb03fa0c170dc3f1136265f4b97ec26d7fa` (only commit, "v1.1"; Apache-2.0) | Base of the pipeline, patched |
| AlphaPose | https://github.com/MVIG-SJTU/AlphaPose | `c60106d19afb443e964df6f06ed1842962f5f1f7` | Halpe-136 keypoint extraction |
| AlphaPose weights | AlphaPose model zoo | `halpe136_fast50_regression_256x192.pth` `6c02c87b…c2`; `yolov3-spp.weights` `87a1e8c8…27b4` (full hashes in `reproduction.json`) | Pose and person detector |
| SCTK (sclite) | https://github.com/usnistgov/SCTK | `9688a26882a688132a5e414cadcb4c19b6fffaba` | WER scoring |
| ctcdecode | https://github.com/parlance/ctcdecode | `c90ad94a0b19554f80804fb7812f2a1447a34a70` | Beam search (width 10) |

The tracker said `code_repos: N/A`, but the search found the official repository. It is owned by GitHub user `christianlsl`, which matches the email handle printed in the paper, and its description is the paper title. The full search is in `reproduction.json.source_search`.

## Released-code audit

The official code is incomplete: it imports two AlphaPose modules that were never released, and it has no contrastive loss.

- **It does not run as released.** `main.py:18` imports `AlphaPose.get_image_keypoint`, and `dataset/dataloader_video.py:23` imports `AlphaPose.keypoint_to_heatmap.get_heatmap`. `AlphaPose/` is in `.gitignore`, and neither module exists in the commit, in upstream AlphaPose, or anywhere GitHub code search reaches.
- **The paper's contrastive learning is absent.** There is no NT-Xent loss, second view, projection head, or temperature. The training loss is ConvCTC + SeqCTC + 25 × KL distillation (`slr_network.py`), essentially VAC with an early-fusion heatmap.
- **Keypoint fusion differs between paper and code.** The paper adds a 2n-dim normalized coordinate vector to the image, which is dimensionally undefined. The code adds a 224×224 heatmap whose rendering is unknown.
- **The config conflicts with the paper:** 80 vs 120 epochs; batch 1 vs 4; distillation weight 25 vs 1 (implied by Eq. 8); horizontal flip used, though the paper excludes it; TemporalRescale(0.2) vs "0.5 random frame dropout"; colour jitter, grayscale, and blur defined but unused (`deep_transform`). No PHOENIX-2014T recipe is stated.
- `torch.clamp(video, ...)` in `normalize` discards its result. The best-dev counter resets on resume. The PHOENIX-2014 train list drops CSV row 2390, as in all VAC-derived code. All three are kept as released behaviour.

Gate `sc2slr-released-code-incomplete` offered three bounded options. The operator chose the released code plus the minimal glue and the paper's contrastive branch, on both datasets. The full runs were launched without the `#repro-sign-team-r` compute discussion that AGENTS.md requires for plans over 24 GPU-hours. At this gate's resolution on 2026-10-26, the operator decided not to wait for it, because earlier requests of the same kind for other papers had gone unanswered in the channel for more than three days. No request was posted for this paper. The operator stated this reason on 2026-10-29. This remains a process deviation.

## Data provenance and permissions

| Dataset | Version/subset/splits | Source | License/permission and cloud-use basis | Path in Volume `datasets` | Counts / checksum | Deviations |
| --- | --- | --- | --- | --- | --- | --- |
| PHOENIX-2014 | phoenix-2014.v3, multisigner, 210×260 PNG frames | RWTH `phoenix-2014.v3.tar.gz`, already in the Volume (checked 2026-10-26) | CC BY-NC-SA 3.0 (the queue says CC BY-SA 3.0); non-commercial research cloud processing | `rwth-phoenix-2014/raw/phoenix2014-release/phoenix-2014-multisigner` | 5672 / 540 / 629 in the CSVs (SHA-256 in `reproduction.json`); 5671 used for training, as released | train row 2390 dropped by the released split |
| PHOENIX-2014T | PHOENIX-2014-T-release-v3, 210×260 PNG frames | RWTH `phoenix-2014-T.v3.tar.gz`, populated 2026-09-15 | Same as above | `rwth-phoenix-2014-t/raw/PHOENIX-2014-T-release-v3/PHOENIX-2014-T` | 7096 / 519 / 642 | none |

Derived inputs live in the paper Volume `lyu-2024-sc2slr-results` under `data/`. Each tar has a JSON manifest beside it, and the hashes are in `reproduction.json.artifacts`. None of them are redistributed.

- **Frames:** resized to 256×256 by the released preprocess over the committed per-split `*_info.npy`. PHOENIX-2014: 798,990 / 75,186 / 89,472 frames. PHOENIX-2014T: 827,354 / 55,775 / 64,627. All checks against the corpus CSVs passed.
- **Keypoints:** AlphaPose Halpe-136 on every original frame. PHOENIX-2014: 963,664 frames. PHOENIX-2014T: 947,756 frames. No frame lacks a detection.

## Environment and patches

- **Training image:** `Dockerfile` on `ghcr.io/sign-language-processing/reproduction:latest` (PyTorch 2.11.0a0 NGC 26.02, CUDA 13.1, cuDNN 9.19), Modal image `im-YmyymOTNJSOt1gD6faZSnW`. It installs SCTK, ctcdecode (patched to C++17), and pinned pip packages, clones SC2SLR at `65c95fb03fa0c170dc3f1136265f4b97ec26d7fa`, and applies the patches below in order.
  - **Base image pin:** `ghcr.io/sign-language-processing/reproduction@sha256:305b6165d306192996358ca312d9a751fa409f43063a76dc7758880a8f905291`.
  - **How it was verified:** the amd64 manifest of that index has the image config `sha256:737a40916d40a4572d425f14d8e12f50fefdcfe4696b489f22bd0292ba7331d2`, which is the config the Modal build logged. `pu-2024-siformer` records the same index digest.
- **Keypoint image:** `alphapose.Dockerfile` (CUDA 11.7.1 devel, Ubuntu 22.04, Python 3.10, torch 1.13.1+cu117, AlphaPose `c60106d19afb443e964df6f06ed1842962f5f1f7` with pinned dependencies), Modal image `im-csO6ZMQ3cAFuADNidOTxqG`. AlphaPose's CUDA extensions do not build against PyTorch 2.x, so it gets its own image.
  - **Base image:** `nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04`. The Modal build on 2026-10-26 logged the image config `sha256:8b66bdfd0aa0366cc7137f00c9dd336bef17e493d4197747d0be3c4f9bdef2ea`.
  - **The tag has moved:** on 2026-10-29 it resolved to a different image (index `sha256:38e59267704b5d91ef63c7d8f613359c629fab0aead1283d59ca7821029e73bc`, amd64 config `sha256:1256fa5b1b7d778ae55948fe094644f5fcede88c375001a2e81fb75877dbcbc1`). So no pullable digest of the base that ran is known.
  - **What survives:** the built image is still available as the Modal image above, and the run records give both digests (`compute.base_image`).
- **Shared Hugging Face cache:** every Modal function in `modal_app.py` mounts the v2 Volume `huggingface-cache` read-write at `/cache/huggingface`.
  - The training image (`Dockerfile`) sets `HF_HOME=/cache/huggingface` and `HF_HUB_CACHE=/cache/huggingface/hub`. This covers the preflight, diagnose, train, and evaluate functions, plus `prepare_frames` and `pack_keypoints`.
  - `alphapose.Dockerfile` does not set these two variables. In `extract_keypoints` (runs `keypoints-preflight-1`, `keypoints-p14-1`, `keypoints-p14t-1`) the cache is mounted but the variables are unset. This is process deviation `hf-cache-env-not-set`.
  - No pipeline step downloads from the Hugging Face Hub: the AlphaPose weights come from the AlphaPose model zoo and are checksummed.

| Patch | SHA-256 | Why |
| --- | --- | --- |
| `patches/ctcdecode-cxx17.patch` | `783d81207ea1b14650388e59b380196e95f2dccf862d26dc82fb34ec0e455335` | PyTorch ≥ 2.1 headers need C++17; copied from `yu-2025-cvsign` |
| `patches/sc2slr-1-keypoint-glue.patch` | `06f7151d2d7bf07d454a37f491a4ad0de946394d06d497ade0c708f1da7fb6f2` | Adds the two missing AlphaPose modules: precomputed Halpe-136 keypoints, 90-point subset, heatmap rendered before augmentation as a 4th channel |
| `patches/sc2slr-2-contrastive-branch.patch` | `c48999597fc6d0227cffb84d56425380a8d993c0aa027e0f6487e509ca33838d` | Adds the paper's two-view NT-Xent branch and the missing imports for `deep_transform` |
| `patches/sc2slr-3-no-horizontal-flip.patch` | `1f722d89ac547a097bdbba0a322a290cfeafaac4beeea4bfa272c77dbcd3b8f5` | Removes the horizontal flip the paper excludes |
| `patches/sc2slr-4-ctc-without-cudnn.patch` | `c8da41525b51cfac061d7efee3634ac4ed50a19a37b02c9020759ffdbdbeef8b` | Computes CTC without cuDNN: cuDNN's CTC kernel gave non-finite gradients that collapsed the AMP scale (see Attempts) |

Patches 1–3 change behaviour and were authorised by the gate decision. Patch 4 is an execution-correctness fix and leaves the loss unchanged.

Level 3 delivered as patches: as in `yu-2025-cvsign`, the components reimplemented from the paper (patches 1 and 2) are expressed as ordered patches against the pinned upstream so a reviewer sees exactly what was added; the level reflects the reimplementation, not the diff shape.

## How to repeat this

Every command goes through the `repro-sign` wrapper. Every Modal function is idempotent: it skips work whose output and manifest already exist, and training resumes from the newest checkpoint.

```bash
W=.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh
A=papers/lyu-2024-sc2slr/modal_app.py
# keypoint preflight: AlphaPose on 4 PHOENIX-2014 clips
$W run --detach $A::launch_keypoint_preflight --dataset phoenix2014 --limit-clips 4
# data: 256x256 frames and Halpe-136 keypoints (48 A10G shards per dataset; the default is 24)
for ds in phoenix2014 phoenix2014-T; do
  $W run --detach $A::launch_prepare_frames --dataset $ds
  $W run --detach $A::launch_keypoints --dataset $ds --num-shards 48
done
# preflight: 48 real clips, train one epoch, resume for a second, then test
$W run $A::launch_preflight --run-id preflight-p14-1 --dataset phoenix2014
$W run $A::launch_preflight --run-id preflight-p14t-1 --dataset phoenix2014-T
# full training: 120 epochs in chained 20-epoch calls
$W run --detach $A::launch_train --run-id full-p14-2 --dataset phoenix2014 --epochs 120 --chunk 20
$W run --detach $A::launch_train --run-id full-p14t-2 --dataset phoenix2014-T --epochs 120 --chunk 20
# evaluation: lowest-dev-WER checkpoint, dev and test scored once
$W run $A::launch_evaluate --train-run-id full-p14-2 --eval-run-id eval-p14-2 --dataset phoenix2014
$W run $A::launch_evaluate --train-run-id full-p14t-2 --eval-run-id eval-p14t-2 --dataset phoenix2014-T
```

`launch_diagnose --variant as-released|ctc-without-cudnn` reruns the 200-step CTC diagnosis.

## Execution evidence

| Run | What | Hardware | Wall time | Outcome |
| --- | --- | --- | --- | --- |
| `keypoints-preflight-1` | AlphaPose on 4 clips | A10 | 4 min | 23.9 frames/s, no missed detections |
| `prepare-frames-p14-1`, `-p14t-1` | 256×256 frames | CPU | 33 / 24 min | checks passed |
| `keypoints-p14-1`, `-p14t-1` | Halpe-136, 48 shards each | A10/A10G | 1.8 / 1.0 h | 17.1 / 11.9 GPU-h |
| `preflight-p14-1`, `-p14t-1` | 48-clip train, resume, test | A100-80GB | 2 min each | all steps exit 0 |
| `full-p14-1`, `-p14t-1` | first full runs | A100-80GB | 30 min | stopped, diverging (`invalid_run`) |
| `diagnose-p14t-*-1` | CTC diagnosis | A100-80GB | seconds | driver import bug (agent error) |
| `diagnose-p14t-*-2` | CTC diagnosis | A100-80GB | 4 min | hypothesis confirmed |
| `full-p14-2` | 120 epochs | A100-80GB (SXM4 and PCIe) | 47.5 h | succeeded |
| `full-p14t-2` | 120 epochs | A100-SXM4-80GB | 52 h | succeeded |
| `eval-p14-2`, `eval-p14t-2` | selection + scoring | A100-80GB | 7 / 5 min | succeeded |

Modal app IDs, function-call IDs, image IDs, timestamps, and stop policies are in `reproduction.json.runs`. Training peaked at about 48 GiB of GPU memory.

**Compute and cost.** Usage was about 47.5 + 52 A100-hours for the two full runs and about 29 A10-hours for keypoints. The cost below is Modal's own billing figure. It was retrieved through the `repro-sign` wrapper on 2026-10-29 with `modal billing report --start 2026-10-26 --end 2026-10-29 --show-resources --json`, which reports in USD.

The total for all apps named `repro-lyu-2024-sc2slr` is **USD 492.28**. By resource:

| Resource | USD |
| --- | ---: |
| A100-80GB | 252.66 |
| A10G | 34.51 |
| Memory | 112.45 |
| CPU | 92.66 |

By run:

| Runs | USD |
| --- | ---: |
| `full-p14-2` | 203.44 |
| `full-p14t-2` | 222.92 |
| `keypoints-p14-1` and `keypoints-p14t-1` | 55.86 |
| everything else | 10.06 |

"Everything else" includes USD 0.74 for apps not retained as runs: image builds and failed launches before `keypoints-preflight-1`. Each run's figure is in `runs[].compute.billed_cost_usd`.

The CHF ceilings in the stop policies were declared before launch at an assumed rate of about CHF 3.4 per A100-80GB hour (GPU + 16 CPU + 128 GiB). That rate is not used for any reported cost.

## Process deviations

**Compute gate not discussed** (`guesses_and_deviations` → `compute-discussion-skipped`).
- **Plan:** the two full runs were planned at 50–90 A100-hours each. Their stop ceilings were 110 GPU-hours and CHF 400 per run. They used about 100 A100-hours in total.
- **What AGENTS.md requires:** any plan over 24 GPU-hours must be discussed in `#repro-sign-team-r` first.
- **What happened:** it was not discussed, and no request was posted for this paper. At the gate resolution of 2026-10-26, the operator decided not to wait for the discussion, because earlier requests of the same kind for other papers had gone unanswered in the channel for more than three days. The operator stated this reason on 2026-10-29. This remains a process deviation.

**Runs launched without GPU-hour or CHF ceilings** (`stop-ceilings-not-declared`). Before these runs, the Modal function limits in `modal_app.py` were written down, never a GPU-hour or CHF ceiling.

How they are recorded:
- **Ceilings:** null. Nothing was backfilled.
- **`declared_at_utc`:** the real time those limits were written.
- **`recording.mode`:** `contemporaneous`, with empty `unknown_fields`.
- **`terminal.detail`:** ends with the omission.

| Run(s) | GPU | `declared_at_utc` | `max_wall_time_seconds` (source) | `max_gpu_hours` / `max_cost_chf` | Billed USD |
| --- | --- | --- | --- | --- | ---: |
| `keypoints-preflight-1` | A10 | 2026-10-26T12:25:54Z | 21600 (`extract_keypoints` timeout) | null / null | 0.13 |
| `keypoints-p14-1`, `keypoints-p14t-1` | 48 × A10G | 2026-10-26T12:25:54Z | 21600 per shard call (`extract_keypoints`; `pack_keypoints` 4 h) | null / null | 33.04, 22.82 |
| `preflight-p14-1`, `preflight-p14t-1` | A100-80GB | 2026-10-26T12:25:54Z | 14400 (`preflight` timeout) | null / null | 0.45, 0.49 |
| `diagnose-p14t-*-1`, `diagnose-p14t-*-2` | A100-80GB | 2026-10-26T16:22:46Z | 7200 (`diagnose` timeout) | null / null | 0.29, 0.35, 0.60, 0.55 |
| `prepare-frames-p14-1`, `-p14t-1` | CPU only | 2026-10-26T12:25:54Z | 43200 (`prepare_frames` timeout) | not applicable (null) | 1.13, 0.83 |

`attempt.max_attempts` is 4 for these runs. That is the transient-retry rule in AGENTS.md, not a declaration specific to each run.

The validator rejects the nine GPU runs: 18 issues, a missing `max_gpu_hours` and a missing `max_cost_chf` for each. It accepts null GPU ceilings only in `retrospective` mode, and that mode is reserved for migrating committed records. The four full training runs (`full-p14{,t}-1`, `full-p14{,t}-2`) and the two evaluation runs did declare all three ceilings before launch.

**Hugging Face cache variables not set** (`hf-cache-env-not-set`). Affected runs: `keypoints-preflight-1`, `keypoints-p14-1`, and `keypoints-p14t-1`.
- **What happened:** these runs used the keypoint image, Modal image `im-csO6ZMQ3cAFuADNidOTxqG`. They mounted the `huggingface-cache` Volume at `/cache/huggingface`, but `HF_HOME` and `HF_HUB_CACHE` were not set, because `alphapose.Dockerfile` does not define them. Their `compute.shared_cache.environment` records both as null.
- **Effect:** no run downloads anything from the Hugging Face Hub. The AlphaPose weights come from the AlphaPose model zoo with recorded SHA-256 checksums, so no number is affected.
- **Why the file is unchanged:** `alphapose.Dockerfile` is left as is, so that the committed file matches the image that ran.

**Validator result: 21 issues.** These are the 18 above plus 3 for the invalid cache environment of these runs. All 21 are the same class of problem: a process requirement that was not met before launch, kept truthful in the record rather than backfilled.

## Guesses and deviations

| ID | Paper/evidence says | This attempt used | Rationale | Effect on interpretation |
| --- | --- | --- | --- | --- |
| `keypoints-precomputed` | Released loader calls the missing `get_heatmap` on every augmented 224×224 frame | AlphaPose once per original 210×260 frame, scaled to 256×256, carried through crop and temporal rescale as a heatmap channel | Running detector + pose model inside the loader for 120 epochs is about 10⁸ extra passes; the crop is applied identically | Keypoints come from uncropped frames |
| `pose-model` | Sec. 3.2: AlphaPose trained on Halpe | Halpe-136 FastPose ResNet50 regression 256×192 + YOLOv3-SPP; highest-scoring person per frame | The model the AlphaPose README uses for Halpe-136 inference | Another Halpe-136 model would differ somewhat |
| `keypoint-subset` | Sec. 3.2: 7 upper body, 9 nose, 12 eye, 20 mouth, 42 hand = 90 | Halpe indices 18, 5–10, 53–61, 62–73, 74–93, 94–135 | Standard 68-point face layout inside Halpe (offset 26) | Indexing convention only |
| `heatmap-rendering` | Sec. 3.2, Eq. 2: a normalized 2n-dim coordinate vector added to the image (dimensionally undefined); released code adds a (T, 224, 224) heatmap from a missing module; no σ anywhere | Max over keypoints of confidence-weighted Gaussians, σ = 4 px at 256×256, in [0, 1], added to every RGB channel of the [−1, 1] video (`sc2slr-1`) | Follows the released fusion; σ is invented | **Behaviour-changing**; one reason the values are conditional |
| `contrastive-branch` | Sec. 3.3, Eq. 3, Eq. 8, Fig. 2: NT-Xent over two augmented views, summed with the CTC and KL losses; no temperature, projection head, pooling, weight sharing, or loss weight given; released code has no contrastive loss | Two views from the released (unused) SimCLR `deep_transform` after one shared crop/rescale; weight-shared network; all released losses on both views; NT-Xent τ = 0.5 (SimCLR default), weight 1, on BiLSTM outputs mean-pooled over valid steps, no projection head (`sc2slr-2`) | Smallest addition that implements Eq. 3 and Eq. 8 with the code's existing augmentations | **Behaviour-changing**; one reason the values are conditional |
| `schedule` | Sec. 4.2: 120 epochs, batch 4; released config: 80 epochs, batch 1 | 120 epochs, batch 4; Adam 1e-4, ÷5 at 40 and 60, weight decay 1e-4 | Where the paper states a value for the Table 3 system, the paper wins | Follows the paper |
| `no-horizontal-flip` | Sec. 4.3 and Table 2: flipping did not help and was excluded from the final scheme; released transform keeps RandomHorizontalFlip(0.5) | Flip removed (`sc2slr-3`) | Table 3's PHOENIX-2014 row equals Table 2's last row, which has no flip | Follows the paper |
| `ctc-without-cudnn` | Released criterion passes CPU int32 targets, so current PyTorch uses cuDNN's CTC kernel; `full-p14-1`/`full-p14t-1` diverged | CTC computed with cuDNN disabled for that call only (`sc2slr-4`) | Execution-correctness fix, isolated by the paired diagnoses; same fix as `yu-2025-cvsign` | Same loss; not behaviour-changing |
| `kept-released` | Eq. 8 sums the losses unweighted; Sec. 4.2 mentions "0.5 random frame dropout" | Released loss weights (Dist 25, SeqKD T = 8), TemporalRescale(0.2), seed 0, AMP, beam 10 | Eq. 8 does not rule out weights, and the dropout phrase maps onto no code parameter; the released code is the more specific evidence | May differ from the authors' runs |
| `p14t-recipe` | Sec. 4.2 gives the schedule only for PHOENIX-2014, and the ablations (Tables 1–2) are PHOENIX-2014 only; released `configs/phoenix2014-T.yaml` sets only paths | Same recipe as PHOENIX-2014 | No other evidence | **Conditional** |
| `checkpoint-selection` | Not stated in the paper (Sec. 4, Table 3 caption); released `main.py` saves a best-dev checkpoint, but its counter resets on resume | Lowest dev WER over all 120 epochs, earliest on ties, fixed before training; dev and test scored once | A guess informed by the released best-dev saving; the paper does not say whether it reports best-dev or final checkpoints | **Guess**; final-epoch checkpoints not scored on test (their dev WER was 23.9 and 22.5) |
| `train-split-2390` | Paper: 5672 training samples; released preprocess drops CSV row 2390 (inherited from VAC) | Released `train_info.npy` (5671 clips) | Published recipe | One training clip fewer than the paper's count |

Execution adaptations that leave the training semantics unchanged:
- **Chunked training:** 20-epoch Modal calls stay under Modal's 24 h limit. `--num-epoch` only bounds the loop, and the LR milestones are epoch-indexed.
- **Resumption after interruptions:** Modal interrupted `full-p14-2` once (after epoch 86) and `full-p14t-2` twice (after epochs 54 and 62). Each time, Modal's retry resumed from the newest epoch checkpoint.
- **Mixed GPUs:** the A100-80GB SXM4 and PCIe variants were mixed across chunks.

## Attempts, failures, and dead ends

1. **Assignment format.** The tracker export is a single `reproductions` object with no `confirmation` field, so `ingest_candidate.py` cannot apply. It is recorded as a direct assignment, following `pu-2024-siformer`. The Paper ID is the tracker's `paper_id` rather than the directory slug: the assignment kind is `direct_user_request`, but a platform record for this paper does exist, so its ID is what traces the report back to it, as in `pu-2024-siformer`.
2. **AlphaPose image builds.** Several failed before one worked:
   - a non-existent PyTorch base tag;
   - a conda `nvcc` without `cusolverDn.h`;
   - easy_install pulling `munkres` 3.0.0, which ships no setup script;
   - Modal rejecting Python 3.8;
   - `tkinter` missing at import time.

   Each was fixed in `alphapose.Dockerfile`. One "image build terminated due to external shut-down" and one local DNS error were transient, and the retries succeeded.
3. **First full runs diverged** (`full-p14-1`, `full-p14t-1`, stopped after about 30 minutes). On PHOENIX-2014 the epoch-0 loss rose from about 280 to about 1,500. On PHOENIX-2014T, 1,608 of 1,774 batches were skipped as NaN.
4. **Diagnosis** (`diagnose-p14t-released-2` vs `diagnose-p14t-ctc-nocudnn-2`): 200 steps each on the same 800 clips. The released criterion passes CPU int32 targets, which routes `nn.CTCLoss` to cuDNN.

   | | As released | CTC without cuDNN |
   | --- | --- | --- |
   | Steps with non-finite gradients | 40 | 5 (AMP calibration) |
   | Final AMP scale | 6e-8 | 2048 |
   | Mean loss per 50 steps | 213 / 144 / 190 / 223 | 179 / 103 / 93 / 89 |

   The fix became patch 4, and both full runs restarted from scratch. The 48-clip preflights had not shown the problem, because the scale collapse needs hundreds of steps. The first diagnostic pair failed on a `sys.path` bug in the agent's driver script.

## Candidate flags, ethics, and human evaluation

The `copied_scores: yes` flag is confirmed: seven prior-work rows of Table 3 are cited. Human evaluation is `no` and ethical concerns are `no`. The work uses existing public research corpora with no new participants, so no ethics gate is needed.

## Author and team contact

None. The full runs (about 130 GPU-hours in total for the attempt) were launched without the `#repro-sign-team-r` compute discussion that AGENTS.md requires for plans over 24 GPU-hours. At the gate resolution of 2026-10-26, the operator decided not to wait for it, because earlier requests of the same kind for other papers had gone unanswered in the channel for more than three days. No request was posted for this paper. The operator stated this reason on 2026-10-29. This remains a process deviation. Gate `sc2slr-conditional-protocol-acceptance` was resolved on 2026-10-29: the reviewer did not accept the invented choices as the protocol.
