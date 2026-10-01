# Siformer reproduction attempt

**Paper ID:** `d4719e6c4d9e7031bba559ad7e9a0fd84082194b`

**Citation:** Muxin Pu, Mei Kuan Lim, Chun Yong Chong. *Siformer: Feature-isolated Transformer for Efficient Skeleton-based Sign Language Recognition*. ACM Multimedia 2024. DOI [10.1145/3664647.3681578](https://doi.org/10.1145/3664647.3681578).

**Paper:** [arXiv v1](https://arxiv.org/abs/2503.20436v1) · **Code:** [official repository](https://github.com/mpuu00001/Siformer)

**Preference level:** 2

**Pipeline status:** `insufficient_information`

**Numerical agreement:** `not_assessed` — neither Table 6 target produced a comparable score.

**Attempt date:** 2026-10-01

The published model executes on real author-supplied skeletal data after resolving two ordinary environment problems. A bounded GPU preflight completed three optimization steps, saved and reloaded the model/optimizer/scheduler, and evaluated eight records. Full training is gated by material differences between the paper and its released recipe. Both required author datasets were acquired, checksummed, counted, and stored in the shared `datasets` Volume; data acquisition is not the remaining blocker.

## Scope and result

The direct assignment selects the two **Siformer (Ours)** rows in Table 6, page 8. The current tracker export nests the final paper in `expand.paper`; it has no `confirmation` field and its database `id` differs from `paper_id`. `reproduction.json.assignment` preserves that actual schema, the original export hash, and a redacted record without inventing queue-review fields.

| Target | Published top-1 | Reproduced | Terminal reason |
| --- | ---: | --- | --- |
| WLASL100, Table 6 Siformer | 86.50% | Not produced | `protocol_ambiguous` |
| LSA64, Table 6 Siformer | 99.84% | Not produced | `protocol_ambiguous` |

Top-1 is sample-level correct argmax divided by the number of evaluated samples, multiplied by 100. The pinned implementation is `siformer/utils.py:evaluate`. These two numbers are the paper's own claims. Other Table 6 comparison rows include copied scores and are outside the assigned scope.

The diagnostic result was **0 correct of 8**, after only three steps on 72 records. It is explicitly **not comparable** to either requested target: it uses a tiny unrectified subset and no augmentation. No difference from the published score or numerical-agreement claim is calculated.

## Decisions still needed

The common blocker is `table6-protocol`; WLASL also has `wlasl-provenance`.

1. **Which configuration generated Table 6?** Section 4.5 specifies six decoder attention heads; `SiFormer` hardcodes nine (`[3, 3, 2, 9]`) at the official revision. The paper specifies LSA64 80% training / 20% testing; the README first creates a 20% holdout, then applies `experimental_train_split=0.8` to the remaining training subset, retaining approximately 64% of the full dataset for training. The example also calls the holdout validation and supplies no testing path. The intended paper values are explicit; an unknown random seed or split membership alone is not a gate. The disagreement concerns whether the released example and prepared features implement that specified experiment.
2. **For WLASL, which original examples were held out before SMOTE?** Section 4.1 lists 2,038 originals, eight samples per class held out (800), and then 2,400 remaining before oversampling—an inconsistent count. The author release has exactly the stated final cardinalities, 3,200 train and 800 validation, but no source-example mapping proving the claimed sampling order. Historical notebooks contain mixed exploratory WLASL/LSA64 oversampling operations, and historical jobs split a balanced CSV; this is not a definitive Table 6 preparation recipe. This is insufficient to conclude that Table 6 leaked data, or to certify the released validation set as the paper's held-out test set.

The LSA64 feature and rectification questions were resolved during independent review. A CPU comparison against [SPOTER's original release](https://github.com/maty-bohacek/spoter/releases/tag/supplementary-data) checked all 3,177 records in order, all labels after the published loader's one-based conversion, and all 349,470 coordinate columns. Every original coordinate is identical; Siformer zero-pads each 15–201-frame sequence to 204 frames. The original file SHA256 is `eafb70e22e8a41e0df0535a0fbc27ec42bd9274a41bd106641c04670052d014e`. Thus the released LSA features are unrectified, not an unidentified transformed corpus. The original skeleton release is CC BY-NC 4.0, permitting this noncommercial cloud analysis.

Historical commit `a6b3cb84c508057fbca71b19b9fcdd9d0443f6a7` explicitly applies FE at alpha 0.4, then AA at alpha 0.4. The pinned current functions and recovered motion tables executed on four real records: shape `(4, 110, 204)` preserved, every output finite, and 4,516 coordinate values changed. The whole-corpus comparison took 184.59 CPU seconds; the rectification check took 4.74 seconds, with no GPU. Evidence is retained under `spoter-whole-corpus-001` and `rectification-check-001` in `d4719e6c-siformer-results`. Early sample comparisons first exposed zero-padded labels and the original loader's label-minus-one convention; their unmatched outputs were never interpreted as proof of different coordinates.

No author or Team S contact was made. An explicit conditional experiment could choose among these alternatives, but its score would remain conditional evidence. This attempt does not silently label one alternative a reproduced target.

## Sources and discovery

| Artifact | Pin / role |
| --- | --- |
| Official code | `979a14ed15ed0f20afd77d447ad23c4f4107a2c3`; MIT; selected author training, loader, model and evaluator |
| AA/FE range tables | `09a7c1c575849edbd5e245dc6d1c80ddce94188a`, `active_motion/`; exercised in four-record CPU check |
| Historical recipe/checkpoint | `fd052122b6d4cd3814cecfa70fd1ecb5dc9fac3e`; older SPOTER-named architecture and six-encoder/six-decoder job; not accepted as a Table 6 checkpoint |
| Author feature release | [README-linked Drive folder](https://drive.google.com/drive/folders/13JyaGqX4voqC1wv3ETzjdE_wh50uLszv); exact file hashes below |

Read all ten paper pages, surrounding table text and references; inspected official source files, the 126-commit history, deleted preparation notebooks/job scripts/range tables, author profile, branches (only main), tags (none), releases (none), and issues. Exact-title/method/supplement searches found the official artifact and secondary implementations; none supplied an authoritative target config. The DOI page returned 403 and the OpenReview forum required a browser challenge; arXiv supplied the complete paper. A Monash institutional PDF was also discoverable. Paper PDF hash and original assignment-export hash are retained in `reproduction.json`; no licensed PDF is committed.

Section 3.4 explicitly says the early-exit classifiers can remain untrained. This unusual choice is supported by the paper and matches the code; it is **not** an open question or a proposed patch.

## Data and permissions

The shared `datasets` and `huggingface-cache` Volumes were readable in workspace `repro-sign`. Existing `LSA64`, `lsa64`, and `WLASL` trees contained raw videos/annotations, which do not substitute for the authors' extracted poses. The exact CSVs were added in paper-specific subdirectories of those existing dataset roots.

| File in Volume `datasets` | Count | SHA-256 |
| --- | ---: | --- |
| `lsa64/siformer/LSA64_60fps.csv` | 3,177; 64 classes; all 204 frames | `52a169473cf199cc5432eab988eaacfab5b527fcb339a24952bc1202b2e247dd` |
| `WLASL/siformer/WLASL100_train_25fps.csv` | 3,200; 32/class | `3027464fe8e53afafaf3f7d859a43df11529a146af3f15a09b90abdd3b21716b` |
| `WLASL/siformer/WLASL100_val_25fps.csv` | 800; 8/class | `19829dcb1bacef8e57fc3c85dbdd86437d495825c90bb7313c6b90bfd48a15fd` |

Each directory retains `manifest.json`, with source information, counts and hashes; its hash/URI is in `reproduction.json`. LSA64 has seven classes below 50 samples, agreeing with the paper's qualitative count description. No missing records were manufactured.

[LSA64's official page](https://facundoq.github.io/datasets/lsa64/) provides CC BY-NC-SA 4.0 and explicitly permits academic use, including derivative preprocessing under the same license. WLASL's official [C-UDA 1.0](https://github.com/dxli94/WLASL/blob/master/start_kit/C-UDA-1.0.pdf) permits computational use; its README restricts the dataset to academic/noncommercial use. Research-cloud computation on those existing skeletal features is the permission basis. No new people or raters were recruited and no identifiable samples are included in Git or logs. No data or weights were published to Hugging Face.

LSA was downloaded and hash-checked before upload. A slow redundant local WLASL download was canceled and its partial bytes discarded; a bounded CPU function acquired the full author files in [app ap-783B92JOmctJGUqXKRcJmG](https://modal.com/apps/repro-sign/main/ap-783B92JOmctJGUqXKRcJmG). An earlier acquisition invocation stopped locally because the paper output Volume was incorrectly required to be v2 even though it had just been created as v1; removing that unnecessary requirement resolved it. The canonical dataset/cache Volumes were unchanged. Acquisition is now an idempotent checksummed `data.sh` called by the Modal entry point.

## Repeat the retained attempt

Run from the repository root after the repository's one-time `./setup.sh` and authenticated `repro-sign` profile:

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh volume list --json
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/pu-2024-siformer/scripts/modal_app.py --acquire
.agents/skills/reproduce-paper/scripts/check_modal_dataset.sh lsa64 siformer/manifest.json
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh volume ls datasets WLASL/siformer --json
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/pu-2024-siformer/scripts/modal_app.py --run-id preflight-repeat-001
python3 .agents/skills/reproduce-paper/scripts/validate_reproduction.py papers/pu-2024-siformer
```

The uppercase existing `WLASL` root is checked with the wrapper directly because `check_modal_dataset.sh` accepts lowercase slugs only. Use a new run ID to preserve existing evidence. The preflight fails if that output directory already contains evidence. Training/evaluation mounts `/datasets` read-only; only controlled acquisition mounts it writable. Shared HF cache is mounted read-write at `/cache/huggingface`, with `HF_HOME` and `HF_HUB_CACHE` set. No Hub model is downloaded.

The preflight invokes the pinned author dataset, model, `train_epoch`, and `evaluate` functions; it does not reimplement their computations. Output goes to `d4719e6c-siformer-results/RUN_ID/`: execution metadata, native stdout, package freeze, GPU/driver details, diagnostic metrics and checkpoint. Source data is not copied into the output bundle. Checkpoint round-trip covers model, optimizer and scheduler state. This is a preflight, not a full resume implementation for the upstream training script.

Full training/evaluation commands are intentionally not presented as a resolved target recipe while the gates remain open. The upstream entry point, 100 epochs, AdamW (learning rate 0.0001, betas 0.9/0.999, weight decay 1e-8), and MultiStepLR (epochs 60/80, factor 0.1) are known. The unresolved inputs/configuration above must be settled before a Table 6 run.

## Environment, patch, and execution

Base: `ghcr.io/sign-language-processing/reproduction@sha256:305b6165d306192996358ca312d9a751fa409f43063a76dc7758880a8f905291`. Runs initially resolved the repository `latest` tag; the repeat entry point now fixes its observed OCI digest. Runtime: Python 3.12, PyTorch `2.11.0a0+eb65b36914.nv26.02`, CUDA 13.1, NVIDIA A10, driver 580.95.05 with CUDA minor-version compatibility. The upstream environment requests Python 3.11/PyTorch 2.1. Resolved freezes are preserved per run. Direct additional dependencies are pandas 2.2.3, scikit-learn 1.6.1, matplotlib 3.10.1 and opencv-python-headless 4.11.0.86. The upstream requirements also contain non-pip/system/Windows entries; they were not blindly installed. OpenCV is needed by augmentation imports; no video is decoded.

`upstream.patch` adds an empty `datasets/__init__.py`. This prevents an unrelated installed Hugging Face `datasets` package from shadowing the authors' namespace directory. The patch does not modify data or model behavior. Its SHA-256 is recorded in JSON. Missing OpenCV is resolved with an environment dependency, not a source change.

| Run | UTC start/end | Result | App / function call |
| --- | --- | --- | --- |
| `preflight-001` | 10:22:42.908–10:22:59.860 | Exit 1: `ModuleNotFoundError: datasets.czech_slr_dataset`; scoped package-marker hypothesis retained | [ap-bAaXG4qdduI2uJHIwEUpub](https://modal.com/apps/repro-sign/main/ap-bAaXG4qdduI2uJHIwEUpub), `fc-01M3VFR4KJ0JJN5V6E8BDWJ9RY` |
| `preflight-002` | 10:24:13.073–10:24:16.776 | Exit 1: loader resolved, missing `cv2`; installed declared dependency | [ap-l6ob1qOY1CFvTY6spwzn4O](https://modal.com/apps/repro-sign/main/ap-l6ob1qOY1CFvTY6spwzn4O), `fc-01M3VFW70NREVGYTH97GQTRY5P` |
| `preflight-003` | 10:25:41.029–10:25:58.834 | Exit 0, `succeeded/completed`: 3 optimizer steps, checkpoint reload, 8-record evaluation | [ap-3iqF1KM1ukwbwAu8XrdNEV](https://modal.com/apps/repro-sign/main/ap-3iqF1KM1ukwbwAu8XrdNEV), `fc-01M3VFYXQDJ6FGD6CCFX934VRW` |

All three are attributed to `siformer-agent`, profile `repro-sign`, environment `main`, one GPU. Attempts 1–3 had a predeclared maximum of three attempts, 1,800 seconds, 0.5 GPU-hours and CHF 3 each. The two failures are `deterministic_code`, linked to their next attempt; the successful run needs no retry. Raw evidence URIs and checksums for every file are in `reproduction.json.artifacts`, attached to its corresponding run.

Successful training took 2.186 seconds for 72 samples; peak allocated GPU memory was 1,080,499,200 bytes. Function wall time was 17.805 seconds. Across all three GPU calls, measured function time was about 38.46 seconds (0.0107 GPU-hours), excluding image builds/cold starts; this is not a billing measurement. Actual CHF cost is unavailable. A training-only linear extrapolation gives 2.70 hours for 100 epochs of 3,200 samples and 2.14 hours for 100 epochs of 2,541 samples, excluding all rectification, input loading, evaluation and checkpoint sweeps. No complete full-run estimate or expense was committed while protocol gates remain open; this is not a compute blocker.

## Attribution and limits

`GPT-6` using `Codex` performed source discovery, implementation, execution, and reporting as `siformer-agent`. Session instructions identify that model family and application; exact model ID/version and Codex application version were not exposed and are not inferred. Every retained GPU run references that agent. The orchestrating agent assigned scope but did not execute these runs. A separate GPT-6/Codex agent independently reviewed the protocol evidence, clarified the explicit paper split and the historical notebook limitations, and did not execute these runs; its exact model and harness versions were likewise unavailable.

Seed 379 and batch size 24 follow upstream defaults. The diagnostic uses consecutive author records, no rectification and no augmentation, and has no target-score interpretation. No optimizer was guessed: it is specified by the paper. No behavior-changing patch to decoder heads, splits or checkpoint selection was retained. Rectification was exercised separately on four records using unchanged pinned author functions; the GPU preflight remains explicitly unrectified. Candidate ethics/copied-score flags were investigated; no new human evaluation was introduced. Scientific interpretation of the documented discrepancies remains for review.

Repeat the additional CPU checks with fresh run IDs:

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/pu-2024-siformer/scripts/modal_app.py --compare-all --run-id spoter-whole-corpus-001
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/pu-2024-siformer/scripts/modal_app.py --check-rectification --run-id rectification-check-001
```
