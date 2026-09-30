# Sign2Pose reproduction

**Paper ID:** `559de08af0b4679a571ac2bd26cacd15728811aa`

**Citation:** Eunice, J.; J, A.; Sei, Y.; Hemanth, D.J. Sign2Pose: A Pose-Based Approach for Gloss Prediction Using a Transformer Model. *Sensors* 2023, 23(5), 2853. https://doi.org/10.3390/s23052853

**Paper:** https://doi.org/10.3390/s23052853 (PMC10007493) · **Code/artifacts:** none from the authors after a full search; cited basis SPOTER at `maty-bohacek/spoter@0f909bf`

**Preference level:** 3

**Pipeline status:** `insufficient_information` — the paper's Table 5 numbers come from an unpublished random re-split of WLASL, so no target can be produced. At the user's decision, a conditional SPOTER-based reconstruction ran end to end on all four subsets, and its numbers are recorded as evidence only.

**Numerical agreement:** `not_assessed` — no comparable value exists. The conditional numbers (47.50 / 27.36 / 16.83 / 11.80 top-1) sit far below the published 80.9 / 64.21 / 49.46 / 38.65, but they are not a like-for-like comparison.

**Attempt dates:** 2026-09-27 to 2026-09-29

## Reproduction agents

| Agent ID | Model and version | Agent application | Contribution | Attribution evidence / unknowns |
| --- | --- | --- | --- | --- |
| `opus-5-5-claude-code` | Opus 5.5 (`claude-opus-5-5`) | Claude Code 2.1.283 | Everything in this attempt, across two interactive sessions on 2026-09-27 onward: setup, assignment provenance, target ledger, source search, gates, pose extraction, patches, every run listed below, scoring and this report. | Session environment declares the model; `claude --version` printed `2.1.283 (Claude Code)` in both sessions. The first session ended when the user's laptop crashed. Its runs were recovered from that session's transcript and the Modal results Volume. |

## Scope and target contract

The queue asks for **"Table 5, bottom line"**. That is the row labelled "Our's" in Table 5 ("Performance analysis on top 1% macro recognition accuracy…"), which gives four numbers. The abstract, contribution 4 and Section 6 repeat the same values as the authors' own results, so none of them is a copied baseline. The four rows above it (Pose-GRU, Pose-TGCN, GCN-BERT, SPOTER) are copied baselines and fall outside the assignment.

What the paper specifies about the protocol, with every gap marked:

- **Split: not the official WLASL split.** Section 5 describes a random 85:15 split and, in the same sentence, a 70/15/15 split. It publishes no split files, seed or stratification rule.
- **Pipeline:** key-frame extraction (histogram difference, threshold μ+σ, Euclidean distance; Algorithm 1 never defines what the distance compares). Then Apple Vision API poses (54 joints, 108 dimensions), rotation/squeeze/perspective/arm-joint augmentation, YOLOv3 signing-space and hand anchor-box normalisation (no weights or detector training data given), and horizontal flip with p = 0.5.
- **Model and training (Table 3 and Section 5):** 6 encoder and 6 decoder layers, 9 heads, hidden size 108, feed-forward 2048, one class query. SGD with lr 0.001, weight decay 1e-4 and momentum 0, uniform (0, 1) initialisation, cross-entropy loss, 300 epochs. The paper says it was implemented in TensorFlow.
- **Metric:** top-1 accuracy. The caption says "macro", while the text says "top 1 class accuracy". The copied baseline rows are per-instance. The paper states no checkpoint rule; Figure 7 shows a plateau after epoch 240. It gives no seeds or variance.

## Results

All four targets are **not produced** (`protocol_ambiguous`, gate `sign2pose-private-split`). The table shows the conditional reconstruction next to the published values. These are test top-1 accuracies (%) from one seed (379), with the conditions listed under Guesses and deviations.

| Target | Paper | SPOTER's pick (max over test) | Validation-selected | Macro, validation-selected | Validation selection | Run | GPU-h |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| `table5-ours-wlasl100` | 80.9 | 47.50 | 47.50 | 49.31 | `checkpoint_v_12`, val 43.80 at epoch 121 | `full-wlasl100` | ~4.9 |
| `table5-ours-wlasl300` | 64.21 | 27.36 | 23.58 | 24.09 | `checkpoint_v_9`, val 32.92 at epoch 86 | `full-wlasl300` | ~5.8 |
| `table5-ours-wlasl1000` | 49.46 | 16.83 | 15.99 | 16.59 | `checkpoint_v_14`, val 18.79 at epoch 139 | `full-wlasl1000` | ~28.8 |
| `table5-ours-wlasl2000` | 38.65 | 11.80 | 10.64 | 10.16 | `checkpoint_v_21`, val 11.64 at epoch 208 | `full-wlasl2000` | ~34.8 |

How the numbers were scored (`score_log.py SPOTER_LOG`):

- **SPOTER's pick** is what SPOTER's `train.py` reports. Every 10 epochs it saves the best-train and the best-validation model, then evaluates all of them on the test set and reports the maximum. That choice uses the test set to select the checkpoint.
- **Validation-selected** is the saved best-validation checkpoint with the highest validation accuracy, taking the earliest on ties.
- **Macro** is the mean of SPOTER's own per-label test accuracies over the labels present in the test split.

Raw differences from the paper, taking SPOTER's pick minus the published value: −33.40, −36.85, −32.63 and −26.85 points. A reviewer should weigh them against the deviations below and not read them as a verdict on the paper.

## How to repeat this

Run all commands from the repository root. Every command goes through the Modal workspace wrapper:

```bash
W=.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh
$W run papers/eunice-2023-sign2pose/modal_app.py::extract_all --shards 64      # MediaPipe poses -> results Volume /poses (resumable)
$W run papers/eunice-2023-sign2pose/modal_app.py::export_csvs                  # SPOTER CSVs -> /csv
for n in 100 300 1000 2000; do
  $W run --detach papers/eunice-2023-sign2pose/modal_app.py::main --subset $n --variant keyframes --epochs 300 --tag full
done
$W run papers/eunice-2023-sign2pose/modal_app.py::result --call-id fc-...     # return value of a detached run
$W volume get eunice-2023-sign2pose-results /runs/full-wlasl100-keyframes-seed379/full-wlasl100-keyframes-seed379_None.log .
python3 papers/eunice-2023-sign2pose/score_log.py full-wlasl100-keyframes-seed379_None.log
```

- `poses.py` holds the MediaPipe → SPOTER joint mapping, the key-frame rule and the CSV export. `python3 poses.py check` runs its self-check.
- `modal_app.py` builds two images and pins SPOTER at `0f909bf`, applying `patches/` in name order. The CPU image is `python:3.12-slim`, `mediapipe==0.10.21` and `simple-video-utils==0.9.1`. The GPU image is `ghcr.io/sign-language-processing/reproduction:latest` with SPOTER.
- Training writes to the Modal Volume `eunice-2023-sign2pose-results`. With `patches/0004`, an interrupted run resumes from its last committed epoch.

## Data provenance and permissions

| Dataset | Version/subset/splits | Source and access date | License/permission and cloud-use basis | Path in Volume `datasets` | Counts / manifest / checksum | Deviations |
| --- | --- | --- | --- | --- | --- | --- |
| WLASL | `WLASL_v0.3.json`; WLASL100/300/1000/2000 are the first K glosses; official train/validation/test labels from `index.csv` | https://dxli94.github.io/WLASL/, checked 2026-09-27 | Custom research-only terms. A project copy already exists (queue `on_modal: yes`). Missing videos require the authors' terms-of-use request form. | `WLASL/` | JSON `31ba5a5c…`, index `fba4a2c0…` (15,146 rows, 12,255 files). Local copies of all 12,255 videos matched their index MD5s. Instances used, train/val/test: 1001/242/200, 2491/650/530, 6509/1692/1432, 10089/2905/2152 | About 28% of instances have no stored video (gate `wlasl-missing-videos`, still open). Official split used instead of the paper's private split. |

The pose extraction processed all 15,146 instances with no per-instance errors (`export_csvs` output).

## Guesses and deviations

Each of these is a behaviour-changing choice. Together with the split, they make every number conditional:

1. **Split:** official WLASL split instead of the authors' unpublished random 70/15/15 re-split.
2. **Data coverage:** only instances with stored videos, about 71–72% of each subset.
3. **Poses:** MediaPipe Holistic 0.10.21 (library defaults, a fresh tracker per instance) replaces Apple Vision. Its joints are mapped to SPOTER's 54 Apple Vision joints. `neck` is the shoulder midpoint, since MediaPipe has no neck landmark. y is flipped to Vision's upward convention. Undetected joints are (0, 0), as in SPOTER's data. Body axes, left/right sides and column names were checked against SPOTER's published WLASL100 CSVs. The user chose this after the local Apple Vision extraction crashed their laptop (see Attempts).
4. **Key frames:** Algorithm 1 is read as Ed(t) = ‖H(t−1) − H(t)‖₂ over 256-bin grayscale histograms, with Th = mean(Ed) + std(Ed) and frames kept where Ed > Th. If no frame passes, all frames are kept. The preflight kept 195 of 1,852 frames (~10.5%).
5. **Frame range:** the official WLASL `start_kit/preprocess.py` rule.
6. **Normalisation:** SPOTER's signing-space and hand normalisation replaces the unspecified YOLOv3 anchor-box step.
7. **Training:** SPOTER `train.py` defaults (batch size 1, SGD lr 0.001, its augmentations, Gaussian noise std 0.001) with `--hidden_dim 108 --epochs 300 --seed 379`. The paper's weight decay 1e-4 and horizontal flip were not added, because SPOTER, the cited basis, has neither (its optimiser is plain `optim.SGD(lr)` and its augmentations are rotation, squeeze, perspective and arm-joint rotation). The uniform (0, 1) initialisation matches SPOTER's `torch.rand` embeddings.
8. **Checkpoint selection and aggregation:** unspecified in the paper, so three variants are reported (see Results).

## Environment and patches

- **GPU:** NVIDIA L4 on Modal, driver 580.95.05. The container is NGC PyTorch release 26.02 (PyTorch `2.11.0a0+eb65b36`, CUDA 13.1 in minor-version compatibility). The image digest was not recorded; the full `pip freeze` is stored per run (SHA-256 `f417f70c…`).
- **Image fix:** Hugging Face `datasets` is uninstalled from the image because it shadows SPOTER's `datasets/` directory.

| Patch | SHA-256 | Why |
| --- | --- | --- |
| `0001-torch2-decoder-layer-compat.patch` | `dcc7a567…` | torch 2 reads `layers[0].self_attn`, which SPOTER deletes. Keeps the unused module, so the computation is unchanged. |
| `0002-evaluate-stats-num-classes.patch` | `e88a5c01…` | `evaluate()` hardcodes 101 classes, so any label ≥ 101 crashed (`KeyError: 1158`). |
| `0003-torch-load-full-model.patch` | `acdfba99…` | torch ≥ 2.6 defaults `weights_only=True`, which cannot load SPOTER's whole-model checkpoints. |
| `0004-resume-from-last-epoch.patch` | see `reproduction.json` | Resume after Modal restarts or the 24 h limit. Verified exact: a resumed run's epochs 3–4 matched an uninterrupted run digit for digit (`resume-check`). |

Runs `full-wlasl100`, `full-wlasl300` and `full-wlasl1000` used patches 0001–0003. `full-wlasl2000` used 0001–0004.

## Execution evidence

All runs used Modal profile `repro-sign`. Run IDs, app IDs, function-call IDs, timestamps, ceilings and terminal states are in `reproduction.json.runs`. Raw logs are on the results Volume with SHA-256 hashes in `reproduction.json.artifacts`.

| Run | Modal app | Outcome |
| --- | --- | --- |
| `upstream-check-attempt-1..3` | `ap-FhGm…`, `ap-RhfaH…`, `ap-t2tWh…` | SPOTER's own WLASL100 CSVs. The runs failed, in order, on a `datasets` import shadow, a torch 2 `self_attn` error, and SPOTER's code/data label mismatch. |
| `training-preflight-attempt-1..3` | `ap-4esq…`, `ap-8QRG…`, `ap-7Xf8…` | Found patches 0002 and 0003. Attempt 3 passed: train, save, reload, test. |
| `resume-check` | `ap-CiDG…` + `ap-u6BG…`/`ap-3pfh…` | Resume verified exact. |
| `full-wlasl100` | `ap-2dwYmHHGVBfIpWnsnaLGLD` | Succeeded. One Modal restart (from scratch, before 0004). |
| `full-wlasl300` | `ap-4hYUEjm8RhrJoTx1d7sznu` | Succeeded. |
| `full-wlasl1000` | `ap-hh3G01AxdUKRyrVquWKqGp` | Succeeded. One restart from scratch; ceiling raised by the user (gate `wlasl1000-ceiling`). |
| `full-wlasl2000` | `ap-E4SDYny7DyusEdCWvBls9T` | Succeeded. Three restarts, resumed after epochs 60, 101 and 130 (gate `wlasl2000-compute`). |

Pose extraction (not a GPU run): app `ap-BcnURZwcvduyQVVs4PKbBF`, 64 shards × 8 CPUs, 11.4 min wall time and about 3 container-hours. CSV export: `ap-AXrhUciD7b36cIlcWc2hML`.

**Totals:** about 75 L4 GPU-hours (the four full runs plus about 0.5 h of checks), roughly USD 60 at about USD 0.80 per GPU-hour. CPU extraction added a few dollars.

## Attempts, failures, and dead ends

- **Local Apple Vision extraction** on the user's Mac (M3 Pro, 18 GB RAM) crashed the laptop, and no output survived. The cause was measured afterwards: PyObjC/Vision objects leaked about 75 KB per frame without an autorelease pool (606 → 749 MB over 2,000 frames, against a flat 213 MB with one). That leak, 4 workers and a 98%-full disk left no memory headroom. The user then ruled out local extraction.
- **Extraction launches without the shared cache.** The first extraction preflight (`ap-Z9qe4I6wWNIF3COndjFq1G`, 24 instances) and first full launch (`ap-FmKATiMCea1Ts036IxSqVd`) did not mount `huggingface-cache`. The full launch was stopped after about 2 minutes and its partial shards were deleted before the compliant relaunch. The preflight's output (`/preflight-limit24/`) was used only for the sanity checks.
- **Wrapper under `timeout`:** launching through `timeout` makes the wrapper's profile check fail before reaching Modal (seen twice). Runs were launched without `timeout`.
- **CSV export** first failed on a split-name mismatch (`validation` vs `val`), fixed in `poses.py`.
- **SPOTER's published WLASL100 CSVs** use 0-based labels, while its loader subtracts 1. That is an inconsistency between its released code and data. Our CSVs are written 1-based.
- **Modal restarts** cost 244 epochs (WLASL100) and 125 epochs (WLASL1000) before resume support existed. The restarted attempts reproduced identical early epochs.
- `ingest_candidate.py` rejected the queue export because it is a single object rather than an array. The record is preserved verbatim, not reshaped.
- The queue record has `status: final` but no `confirmation` field. The user confirmed approval on 2026-09-27. The record stays verbatim, so the validator still flags the missing field, as with the zhang-2023-sltunet precedent.
- Every attempt to get the paper as a PDF was blocked (MDPI and Europe PMC return 403). The Europe PMC JATS XML was hashed instead.

## Observations for the reviewer

These are leads only; none of them was investigated further:

- All four runs overfit strongly. For example, WLASL100 training accuracy passed 78% while validation plateaued near 40–44% from about epoch 120.
- SPOTER logged "Problematic normalization" 14,679 times in the WLASL100 log (both container attempts). That is its fallback when normalising a sample fails, likely frames without detected hands. With only ~10% of frames kept as key frames, this may matter.
- SPOTER's test-set checkpoint selection inflates WLASL300/1000/2000 by 0.8–3.8 points over validation selection.

## Source provenance

| Artifact | Canonical source | Pinned revision / SHA-256 | Role |
| --- | --- | --- | --- |
| Paper full text | Europe PMC JATS XML, PMC10007493 | `cf6fc433b7cf8584d71b4f062af25bbc598be0905df204f6a32b28dea3470c67` | Targets and protocol |
| Published code | none found | — | — |
| SPOTER code (paper ref. 48) | https://github.com/maty-bohacek/spoter | `0f909bf92690772f43f0062be41860ed85b461ad` (Apache-2.0) | Cited basis; reconstruction base |
| SPOTER WLASL100 poses | spoter release `supplementary-data` | stored as `/csv/WLASL100_*_spoter.csv` on the results Volume (CC BY-NC 4.0) | Upstream check and convention check only |
| WLASL start kit | https://github.com/dxli94/WLASL | `ac00e6be631c1a2a486621b65f202219f3964d6b` | Frame-range rule |

**Search performed (2026-09-27):** the paper's links (only the WLASL Data Availability link exists, and there are no supplements); the MDPI landing page (403); GitHub repo and code search for "Sign2Pose" and variants; the first author's apparent GitHub account; web searches on title, method and author; Zenodo, Hugging Face and OSF. Two GitHub code hits turned up, and neither is the authors' artifact. One is a thesis bibliography. The other is `Chinh-de/Sign_Language_Recognition`, a 2025 third-party MediaPipe model with a different architecture, so it was rejected. The full list is in `reproduction.json.source_search`.

## Candidate flags, ethics, and human evaluation

- `copied_scores: yes`: this applies to the four comparison rows of Table 5, not the target row.
- `code_repos: N/A`: confirmed by the independent search.
- `compute_requirements: N/A`: the paper states no hardware. The reconstruction used one L4 per subset (see Totals).
- `includes_human_evaluation: no` and `potential_ethical_concerns: no`: consistent with the paper. The work uses existing public sign videos only and adds no participants.

## Author and team contact

None. The paper names a corresponding author (D. J. Hemanth). Since this independent attempt is now complete, author contact is permitted, but it was not made. Asking for the split, key-frame and YOLOv3 details is the remaining route to a faithful run. The request for the missing WLASL videos is routed to Team S through gate `wlasl-missing-videos`.
