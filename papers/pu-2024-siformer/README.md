# Siformer: comparative conclusion reproduced

Muxin Pu, Mei Kuan Lim and Chun Yong Chong. *Siformer: Feature-isolated Transformer for Efficient Skeleton-based Sign Language Recognition*. ACM Multimedia 2024. [DOI](https://doi.org/10.1145/3664647.3681578) · [Paper, arXiv v1](https://arxiv.org/abs/2503.20436v1) · [Official code](https://github.com/mpuu00001/Siformer).

Paper ID: `d4719e6c4d9e7031bba559ad7e9a0fd84082194b`. Attempt date: 2026-10-01. Scientific review: 2026-10-02.

**Preference level:** 2

**Pipeline status:** `complete` — under the accepted released-artifact scope (2/2 targets).

**Numerical agreement:** `does_not_agree` — with the exact published values.

**Comparative conclusion:** `reproduced`, validated by human scientific review on 2026-10-02. No further human decision is pending.

Both 100-epoch training attempts completed and emitted audited held-out accuracies. The scientific review accepts the published nine-head implementation and the already executed data reconstruction as the scope of this reproduction. The resulting Siformer scores exceed all main comparison rows in Table 6; on LSA64, 100% also ties the parenthetical original SPOTER result. No baseline was rerun here.

Before the full scores, the attempt was classified conditional because the paper specifies six decoder heads and WLASL lacks verified original-example/SMOTE lineage. That original assessment remains in `assessment_history` and the prospectively recorded `continuation_decisions`. The 2026-10-02 review accepts a released-artifact reproduction; it does not retrospectively establish the unknown historical configuration or resolve WLASL lineage. Scores, checkpoint selection, execution attribution and raw evidence are unchanged. Exact numerical differences and comparative scientific conclusions are assessed separately.

## Scope and measured results

The assignment selects the two **Siformer (Ours)** rows in Table 6, page 8. Comparison rows are outside the execution scope and are used only as paper-reported evidence for the comparative conclusion; they were not rerun or independently verified against their original sources. The tracker export nests the final paper in `expand.paper`, has no `confirmation` property, and distinguishes database `id` from `paper_id`. The direct user assignment, actual redacted record and original export hash are preserved without inventing queue-review fields.

| Dataset | Published top-1 | Reproduced top-1 | Raw difference | Selected epoch |
| --- | ---: | ---: | ---: | ---: |
| WLASL100 | 86.50% | 89.1250% (713/800) | +2.6250 pp | 99 |
| LSA64 | 99.84% | 100.0000% (636/636) | +0.1600 pp | 23 |

Top-1 is correct argmax predictions divided by evaluated samples, multiplied by 100. The metric is the pinned `siformer/utils.py:evaluate`. Selection uses the highest held-out validation accuracy across all 100 epochs, choosing the earliest tie, as prospectively declared from the author validation/checkpoint workflow. The holdout also selects the checkpoint; it is not represented as an untouched independent test.

The native evaluator emitted the exact labels and logits used for each epoch's score. Independent NumPy recomputation verified finite logits, argmax correct count, denominator, earliest maximum epoch and artifact SHA-256. The chosen checkpoint exists and loads. Evaluator arrays are shuffled and do not include original row IDs: no per-example unique-ID audit is claimed. Membership is established by the immutable input CSVs and author loader. Native logs, all epoch metrics, predictions, checkpoints, execution metadata and environment evidence remain on `modal://d4719e6c-siformer-results/`, with run-specific paths and checksums in `reproduction.json`.

## Comparative conclusion and its limits

Table 6 reports the following comparison values. Its caption identifies parenthetical values as scores from the original authors and preceding scores as the Siformer authors' reproductions using GitHub code. This attempt transcribes those values from the pinned paper; it performs no new baseline experiments.

| Dataset | Modality | Comparison method | Table 6 main top-1 | Parenthetical original-author top-1 |
| --- | --- | --- | ---: | ---: |
| WLASL100 | RGB | I3D | 65.89% | — |
| WLASL100 | RGB | TCK | 77.52% | — |
| WLASL100 | RGB | SignBERT+ | 84.11% | — |
| WLASL100 | RGB | Fusion-3 | 75.67% | — |
| WLASL100 | Skeleton | Pose-TGCN | 55.43% | — |
| WLASL100 | Skeleton | ST-GCN | 50.78% | — |
| WLASL100 | Skeleton | SignBERT+ | 79.84% | — |
| WLASL100 | Skeleton | SPOTER | 58.52% | 63.18% |
| LSA64 | RGB | LSTM + LDS | 98.09% | — |
| LSA64 | RGB | DeepSign CNN | 96.00% | — |
| LSA64 | RGB | MEMP | 99.06% | — |
| LSA64 | RGB | I3D | 98.91% | — |
| LSA64 | Skeleton | SPOTER | 99.52% | 100.00% |
| LSA64 | Skeleton | LSTM + DSC | 92.15% | — |

Observed WLASL100 accuracy exceeds the highest comparison value, 84.11%, by 5.015 percentage points. Observed LSA64 accuracy exceeds the highest main comparison value, 99.52%, by 0.48 percentage points and ties the original SPOTER 100%. Therefore the accepted comparison supports the paper's Table 6 conclusion, with the explicit qualification that Siformer does not strictly beat every parenthetical baseline. This is a comparison against published values under the accepted released-artifact scope, not evidence of statistical significance or an independently rerun common-protocol benchmark. Efficiency, robustness and ablation claims were not requested and are not assessed.

## Decisions and scientific limits

The user authorized documented protocol judgments. We retained the published implementation rather than replacing its model with a local reimplementation:

- Nine decoder heads, three encoders/two decoders, FIM enabled, encoder iterative attention enabled with patience one, decoder iterative attention disabled. Section 4.5 instead specifies six decoder heads; the exact historical configuration remains unknown, but the scientific review accepts following the released nine-head code.
- Unchanged author FE rectification followed by AA, both alpha 0.4. Historical commit `a6b3cb84c508057fbca71b19b9fcdd9d0443f6a7` establishes this order. Current functions and recovered motion tables are used.
- LSA64: one stratified seed-42 80/20 split of all 3,177 released records, giving 2,541 training and 636 held-out samples. The paper specifies 80/20. The README's additional 0.8 training reduction is omitted. Missing historical seed alone is an ordinary stochastic-replication uncertainty.
- WLASL100: preserve released 3,200/800 partitions and perform no new SMOTE. Section 4.1 lists 2,038 originals, 800 held out, then 2,400 remaining before oversampling, an arithmetic inconsistency. The release has the stated final sizes but no source-example mapping or definitive prior rectification history. Historical notebooks contain mixed exploratory oversampling operations; they neither prove leakage nor certify leakage-free Table 6 provenance.
- Seed 42, batch 24, four loader workers, float32, 100 epochs, AdamW learning rate 0.0001, betas 0.9/0.999, weight decay 1e-8, MultiStepLR at epochs 60 and 80 with gamma 0.1. Optimizer settings come from the paper/author code; seed and worker count are declared routine choices.

Section 3.4 explicitly permits untrained early-exit classifiers. That unusual behavior matches the code and was retained. No configuration was changed to approach a published score. No author or Team S contact was made.

## Sources and data provenance

| Artifact | Immutable pin / role |
| --- | --- |
| Official implementation | `979a14ed15ed0f20afd77d447ad23c4f4107a2c3`, MIT; author training, loader, model, transformations and evaluator |
| Motion range tables | `09a7c1c575849edbd5e245dc6d1c80ddce94188a`, `active_motion/` |
| Historical preparation evidence | `a6b3cb84c508057fbca71b19b9fcdd9d0443f6a7`, FE then AA at 0.4 |
| Historical checkpoint/job | `fd052122b6d4cd3814cecfa70fd1ecb5dc9fac3e`; older SPOTER-named six-encoder/six-decoder model, not accepted as a Table 6 checkpoint |
| Author feature release | [README-linked Drive folder](https://drive.google.com/drive/folders/13JyaGqX4voqC1wv3ETzjdE_wh50uLszv), individual checksums below |

Discovery covered all paper pages, references, author pages, repository files and 126-commit history, deleted notebooks/job scripts/range tables, branches, tags, releases, issues and title/method/supplement searches. No authoritative alternative Table 6 configuration was found. The DOI endpoint returned 403 and OpenReview required a browser challenge; the complete arXiv paper was used. PDF and export hashes are recorded; neither licensed paper nor data is in Git.

| Canonical file on Volume `datasets` | Records | SHA-256 |
| --- | ---: | --- |
| `lsa64/siformer/LSA64_60fps.csv` | 3,177, 64 classes, 204 frames | `52a169473cf199cc5432eab988eaacfab5b527fcb339a24952bc1202b2e247dd` |
| `WLASL/siformer/WLASL100_train_25fps.csv` | 3,200, 32/class | `3027464fe8e53afafaf3f7d859a43df11529a146af3f15a09b90abdd3b21716b` |
| `WLASL/siformer/WLASL100_val_25fps.csv` | 800, 8/class | `19829dcb1bacef8e57fc3c85dbdd86437d495825c90bb7313c6b90bfd48a15fd` |

A whole-corpus CPU audit compared all 3,177 LSA records, mapped labels and 349,470 coordinate columns with [SPOTER's original release](https://github.com/maty-bohacek/spoter/releases/tag/supplementary-data), SHA-256 `eafb70e22e8a41e0df0535a0fbc27ec42bd9274a41bd106641c04670052d014e`. Every original coordinate matches; Siformer zero-pads 15–201-frame sequences to 204 and uses zero-based labels. Seven classes contain fewer than 50 records, consistent with the paper's description. No missing records were manufactured. Initial sample checks exposed padding and label conversion and were superseded by this whole-corpus audit, not interpreted as proof of different data.

[LSA64](https://facundoq.github.io/datasets/lsa64/) permits academic use under CC BY-NC-SA 4.0; SPOTER's skeleton release specifies CC BY-NC 4.0. WLASL's [C-UDA 1.0](https://github.com/dxli94/WLASL/blob/master/start_kit/C-UDA-1.0.pdf) permits computational use, with academic/noncommercial restrictions in its README. These existing public research skeletons support the noncommercial project-cloud processing performed here. No new participants/raters were recruited; no identifiable samples were committed or included in logs. No data or weights were published to Hugging Face.

Full transformed CSVs, manifests, original hashes and compressed author diagnostics are stored under `datasets/lsa64/siformer/resolved-seed42-aafe04` and `datasets/WLASL/siformer/resolved-seed42-aafe04`. Preparation calls unchanged row-independent author functions in batches of 64 to bound memory and retain completed chunks. CPU preparation took 1,358.16 and 1,682.88 seconds respectively. The original CSVs remain unchanged.

## Environment and patches

Base OCI: `ghcr.io/sign-language-processing/reproduction@sha256:305b6165d306192996358ca312d9a751fa409f43063a76dc7758880a8f905291`. Patched Modal image: `im-ny7WiuC5WuZ7j2afO49W5G`. Python 3.12, PyTorch `2.11.0a0+eb65b36914.nv26.02`, CUDA 13.1, NVIDIA A10/A10G, driver 580.95.05 with minor-version compatibility. Direct additions are pandas 2.2.3, scikit-learn 1.6.1, matplotlib 3.10.1 and opencv-python-headless 4.11.0.86. Per-run freezes and hardware reports record the resolved environment. Upstream requests Python 3.11/PyTorch 2.1 and includes non-pip/system/Windows requirements; the repository base was used successfully. OpenCV serves augmentation imports; no video decoding was introduced.

Three ordered patches retain the author computation:

1. `001-package-marker.patch` adds empty `datasets/__init__.py`, preventing the installed Hugging Face package from shadowing the author loader.
2. `002-checkpoint-resume.patch` saves/restores epoch, model, optimizer, scheduler, Python/NumPy/PyTorch/CUDA RNG and loader-generator state, and records native epoch metrics. Resume state is atomically replaced at epoch boundaries.
3. `003-evaluation-evidence.patch` records logits/labels from the exact existing evaluator pass without changing prediction or aggregation.

Patch SHA-256 values and executed adapter hashes are in JSON. The initial LF-context build failed against upstream CRLF; `git apply --ignore-space-change` was tested against the exact original Git blob and retained. Original preflight adapter versions are preserved by content hash where later full-run count/checkpoint guards were added. No training semantics changed after preflight.

## Repeat commands

After one-time repository `./setup.sh` and authorized `repro-sign` authentication, run from the repository root. Every Modal operation uses the wrapper. Verify both canonical v2 Volumes, acquire pinned inputs, and perform a representative preflight:

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh volume list --json
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/pu-2024-siformer/scripts/modal_app.py --acquire
.agents/skills/reproduce-paper/scripts/check_modal_dataset.sh lsa64 siformer/manifest.json
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh volume ls datasets WLASL/siformer --json
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/pu-2024-siformer/scripts/modal_app.py --prepare-data --tiny --dataset lsa64 --run-id prepare-lsa64-tiny-002
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/pu-2024-siformer/scripts/modal_app.py --target --tiny --dataset lsa64 --run-id preflight-lsa64-repeat
```

The tiny preparation ID is the directory expected by the adapter; repeat preparation checks the existing manifest. Use a fresh GPU run ID to retain independent evidence. Repeat the two tiny commands for `wlasl100`, using `prepare-wlasl100-tiny-002`. Then run each dataset's full pipeline in its own detached app:

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/pu-2024-siformer/scripts/modal_app.py --prepare-data --dataset lsa64 --run-id prepare-lsa64-repeat
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/pu-2024-siformer/scripts/modal_app.py --target --dataset lsa64 --run-id full-lsa64-repeat
```

Wait for preparation to complete before launching training. Repeat with `wlasl100` and corresponding fresh IDs. Preparation is idempotent against source and derived-file hashes. Training invokes the pinned `train.py`; evaluation occurs inside its epoch loop. Completed output is recognized by its metrics artifact. Epoch resume was verified in preflight; any retry must first preserve its prior invocation metadata and follow the recorded attempt ceiling.

The `datasets` and `huggingface-cache` handles explicitly select VolumeFS v2. Training/evaluation mounts datasets read-only; only controlled acquisition/preparation writes them. Shared cache mounts read-write at `/cache/huggingface`, with `HF_HOME` and `HF_HUB_CACHE` set. No Hub model is downloaded. Outputs use the existing paper-specific Volume; datasets never use that Volume except temporary tiny preflight slices.

## Runs, failures and resources

The initial `preflight-001` failed on the loader namespace; `preflight-002` resolved that but found missing OpenCV; `preflight-003` completed three training steps and model/optimizer/scheduler checkpoint reload. Its 0/8 diagnostic evaluation used an unrectified tiny subset and is not a target result. Those calls totaled about 38.46 function seconds (0.0107 GPU-hours), excluding builds/startup.

Both `prepare-*-tiny-001` builds failed on CRLF patch matching before any remote function or GPU ran. Their scoped second attempts completed. Sanitized failed-build logs are immutable by hash. The new `preflight-resolved-*-001` runs trained two epochs, resumed to three, verified optimizer step advancement and evaluated real rectified data with unchanged author augmentation. LSA64 used 64/16 records and 27.19 script seconds; WLASL used 80/80 and 56.45 seconds. Both peaked near 1.093 GB allocated GPU memory. WLASL's zero tiny-subset validation score produced no best-validation checkpoint because upstream initializes the best score to zero; its independent resume state was valid. Full runs explicitly require a selected checkpoint.

The predeclared full plan conservatively estimated 3 and 5.5 A10G-hours. Actual full scripts completed as follows:

| Full run | Script wall time | GPU-hours | Planning-rate estimate | Modal app |
| --- | ---: | ---: | ---: | --- |
| `full-resolved-lsa64-001` | 2685.46 s | 0.7460 | CHF 1.86 | [ap-aYZf4WIHaS1xO1S7yaOUQZ](https://modal.com/apps/repro-sign/main/ap-aYZf4WIHaS1xO1S7yaOUQZ) |
| `full-resolved-wlasl100-001` | 3146.27 s | 0.8740 | CHF 2.18 | [ap-0WW5WdqVe0j5qpaxatH808](https://modal.com/apps/repro-sign/main/ap-0WW5WdqVe0j5qpaxatH808) |

These are script-duration GPU-hour estimates, excluding some cold-start/container overhead, CPU preparation, earlier diagnostics and storage. CHF uses a prospective conservative rate of CHF 2.50/GPU-hour, not an invoice. Actual billed CHF is unavailable. LSA64's enforced ceiling was 6 hours/CHF 15 and WLASL's 8 hours/CHF 20, each at most two attempts; both completed their first full attempt. No score-based restart or tuning occurred. All run commands, UTC timestamps, exit codes, function-call IDs, source hashes and controlled terminal records are in `reproduction.json`.

Acquisition also retained its dead ends: a redundant slow local WLASL download was canceled, and a CPU function acquired the checked author files; an initial local invocation unnecessarily required the newly created output Volume to be v2, then removed that requirement without changing the canonical v2 Volumes. No data access restriction was bypassed.

## Agent attribution and validation

All agents were exposed by their sessions as **GPT-6 using Codex**. Exact model IDs/builds and application versions were unavailable and are explicitly not inferred. `siformer-agent` performed initial discovery, data acquisition and diagnostics, then report-only Table 6 verification and recording of the 2026-10-02 human scientific assessment; `arabic-vit-review-agent` independently reviewed the initial protocol; `continuation-agent` prepared data, executed the conditional preflights/full runs and wrote the updated report; `codex-orchestrator-review` reviewed resume/evidence patches and the checkpoint-index formula, then independently verified both terminal prediction hashes and numerical results. Review-only agents are absent from execution `agent_ids`. Session attestations and contribution scope are recorded in JSON.

Syntax, shell parsing, patch application and repository metadata checks passed. Numerical evidence was independently recomputed from the exact evaluator outputs, subject to the no-source-ID limitation above. Run the report validator with:

```bash
python3 .agents/skills/reproduce-paper/scripts/validate_reproduction.py papers/pu-2024-siformer
```
