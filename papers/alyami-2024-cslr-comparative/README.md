# Alyami & Luqman (2024) — Comparative Study of CSLR Techniques: reproduction

**Paper ID:** `8d2b672d06ddb561f890798d7c1e434389c34202`

**Citation:** Sarah Alyami and Hamzah Luqman. *A Comparative Study of Continuous Sign Language Recognition Techniques.* arXiv:2406.12369v1, 2024. A revised version appeared in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, 2025, pp. 4923–4932.

**Paper:** https://arxiv.org/abs/2406.12369v1 · **Code/artifacts:** the study did not publish its own code. Each evaluated model's official repository is used (see Source provenance).

**Preference level:** 2

The preference level is expected rather than confirmed; see `status.preference_level_basis`.

**Pipeline status:** not yet set. It stays null until targets have terminal results. The compute gate `compute-full-table` was resolved on 2026-11-02 with an absolute ceiling of USD 2,000.

**Numerical agreement:** `not_assessed` — no target has been produced.

**Attempt dates:** 2026-10-26 (intake and gates); continued from 2026-11-02

## Reproduction agents

| Agent ID | Model and version | Agent application | Contribution | Attribution evidence / unknowns |
| --- | --- | --- | --- | --- |
| `claude-opus-5-5-claude-code` | Claude Opus 5.5 (`claude-opus-5-5`) | Claude Code 2.1.310 | Assignment intake, version resolution, source search, dataset inspection on Modal `repro-sign`, target/protocol/compute gates | Session system context names the model and ID. `claude --version` returned 2.1.310. No runs executed. |
| `claude-opus-5-5-claude-code-2026-11-02` | Claude Opus 5.5 (`claude-opus-5-5`) | Claude Code 2.1.316 | From 2026-11-02: gate updates, data population, container, preflights, estimates, launching and monitoring full runs | Session system context names the model and ID. `claude --version` returned 2.1.316. |

## Scope and target contract

The assignment is a single "reproductions" export record together with the arXiv v1 PDF. The record's `what_to_reproduce` is "Table 1".

The record has paper `status: final` but no `confirmation` field, and it is not a top-level candidate array. `ingest_candidate.py` therefore cannot accept it. It is recorded as a `direct_user_request` that preserves the export with emails redacted, following the precedent in `papers/pu-2024-siformer`.

**Target version (gate `target-table-version`, resolved).** The two paper versions have materially different Table 1s:

- **arXiv v1 Table I:** 5 models × 6 settings.
- **ICCVW 2025 Table 1:** 7 models × 8 settings. It adds Swin-MSTP, SlowFastSign and CSL-Daily, uses different epochs, and changes some values.

The operator chose the arXiv v1 Table I, which is the supplied PDF. That gives 30 targets: VAC, SMKD, TLP, SEN and CorrNet, each evaluated on six settings:

- Phoenix2014 Signer-Dep
- Phoenix2014 Signer-Indep (SI5)
- ArabSign Signer-Dep
- ArabSign Signer-Indep (mean of 6 leave-one-signer-out folds, Table II)
- GrSL Signer-Indep (signer 3)
- GrSL Unseen-Sent

The metric is test WER in percent: sclite corpus-level (S+D+I)/N, Eq. 1. None of the values is a copied baseline. The paper re-ran the Phoenix Signer-Dep column itself, and the other columns are new benchmarks. The paper states no seeds or checkpoint rule.

**Protocol (gate `protocol-lr-epochs`, resolved).** The paper says it "followed the same experimental settings described in each study". The same paragraph then states Adam, lr 1e-3, 40 epochs, ×0.2 at epochs 20 and 35, and batch 2 for all models. Every pinned upstream config instead uses lr 1e-4, and TLP/SEN train for 80 epochs (steps 40/60). The operator chose the paper-stated protocol for every model: lr 1e-3 and 40 epochs with steps 20/35. Everything else is each repo's unchanged config.

The ICCVW 2025 version (Sec. 3.3) instead says TLP and SEN were trained for 80 epochs, with steps 30/60. The decision stands because arXiv v1 is the assigned paper. Yet the TLP and SEN Table I values are the same in both versions: all six TLP cells and five of six SEN cells are identical at one decimal. So the change in stated schedule did not come with new numbers, and the evidence does not show which schedule produced them. Before the full TLP and SEN runs, the measured cost of an 80-epoch schedule will be reported to the operator for a decision.

| Target ID | Paper location | Published WER |
| --- | --- | ---: |
| vac-phoenix2014-signer-dep / -signer-indep | Table I | 21.94 / 40.40 |
| vac-arabsign-signer-dep / -signer-indep | Table I (SI = Table II avg) | 0.2787 / 6.97 |
| vac-grsl-signer-indep / -unseen-sent | Table I | 29.60 / 74.92 |
| smkd-phoenix2014-signer-dep / -signer-indep | Table I | 22.31 / 43.08 |
| smkd-arabsign-signer-dep / -signer-indep | Table I | 0.2091 / 4.24 |
| smkd-grsl-signer-indep / -unseen-sent | Table I | 30.80 / 61.10 |
| tlp-phoenix2014-signer-dep / -signer-indep | Table I | 22.00 / 35.70 |
| tlp-arabsign-signer-dep / -signer-indep | Table I | 0.2439 / 2.90 |
| tlp-grsl-signer-indep / -unseen-sent | Table I | 6.40 / 75.50 |
| sen-phoenix2014-signer-dep / -signer-indep | Table I | 20.20 / 34.40 |
| sen-arabsign-signer-dep / -signer-indep | Table I | 0.2962 / 7.22 |
| sen-grsl-signer-indep / -unseen-sent | Table I | 8.09 / 74.80 |
| corrnet-phoenix2014-signer-dep / -signer-indep | Table I | 19.90 / 33.60 |
| corrnet-arabsign-signer-dep / -signer-indep | Table I | 0.2265 / 7.65 |
| corrnet-grsl-signer-indep / -unseen-sent | Table I | 1.50 / 64.70 |

## Source provenance

| Artifact | Canonical source | Pinned revision / SHA-256 | Role |
| --- | --- | --- | --- |
| Paper PDF (target) | https://arxiv.org/pdf/2406.12369v1 | `db652a81563f9e5c7901ba37df4b69164c28c9acfae858ffff682441ae9eecf6` | Target and protocol |
| Paper PDF (ICCVW 2025) | openaccess.thecvf.com, ICCV2025W/MSLR | `922a2038faac61b3eacbefb5be417e5dbf126b3655814005189f06391dd5aa63` | Version comparison only |
| VAC + SMKD code | https://github.com/VIPL-SLP/VAC_CSLR | `71f3e0334fbc8cecc7ce9816ec69781068abaac0` | VAC; SMKD via `share_classifier`/`weight_norm` (README) |
| TLP code | https://github.com/hulianyuyy/Temporal-Lift-Pooling | `21be215c6ba749d98ce6a51f4b84cc9d5029bfe6` | TLP |
| SEN code | https://github.com/hulianyuyy/SEN_CSLR | `c9a4c7ebd0489e22496c00f9f31d930c0d968ad8` | SEN |
| CorrNet code | https://github.com/hulianyuyy/CorrNet | `a812814f4deefda0b75ab1dcc90b106c62912263` | CorrNet |
| ArabSign split CSVs | https://github.com/Hamzah-Luqman/ArabSign `ArabSign-EncoderDec.zip` | repo `8f6e127`, zip `6980aec2…eb0` | Per-signer train/test lists |

Locations searched on 2026-10-26:

- The arXiv abs page, which has only v1 and no code link.
- The ICCVW open-access page, which has no code or supplement.
- The dataset author's GitHub repositories.
- GitHub searches for SMKD, self-mutual distillation, ArabSign and GrSL CSLR.
- An exact-title web search.

None of these turned up a repository for the comparative study itself.

## Results

No target has been produced yet. All 30 targets are open, and no full run has completed.

## How to repeat this

Run from the repository root. Every Modal command goes through the `repro-sign` wrapper.

```bash
W=.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh
A=papers/alyami-2024-cslr-comparative/modal_app.py

papers/alyami-2024-cslr-comparative/data.sh       # populate/verify the datasets Volume (idempotent)
$W run $A::launch_prepare_frames --dataset phoenix2014   # 256x256 frame archives on the results Volume
$W run $A::launch_prepare_frames --dataset arabsign
$W run $A::launch_prepare_frames --dataset gsl
```

The preflight, training and evaluation commands are listed per run in `reproduction.json.runs`.

## Data provenance and permissions

| Dataset | Splits | License / cloud basis | Path in Volume `datasets` | State |
| --- | --- | --- | --- | --- |
| Phoenix2014 v3 | multisigner 5672/540/629; SI5 4376/111/180 | CC BY-NC-SA (RWTH), non-commercial research | `rwth-phoenix-2014/raw/phoenix2014-release` | Present; checked. All SI5 IDs resolve to multisigner frame folders. Corpus CSV hashes are in `reproduction.json`. |
| ArabSign (RGB) | Signer-Dep 7,489 train+val / 1,841 test (of 7,492 / 1,843 listed); 6 LOSO folds | Non-commercial; author-granted permission via Team S | `arab-sign/RGB/videos` | Extracted 2026-11-02 by `data.sh`: 9,330 mp4 with per-file SHA-256. 5 listed videos are absent from the RGB release. The 709-sample validation subset is unpublished. |
| Continuous GSL | SI (signer 3) 8821/588/881; unseen-sentence 8189/1063/1043 | CC BY 4.0 (Zenodo 3941811) | `gsl-continuous` | Populated 2026-11-02 by `data.sh`: 15 RGB zips plus `supplementary.zip`, 40.17 GB, md5-verified. The 6 split files match their recorded SHA-256. |

`data.sh` runs two population steps:

- `populate_gsl` downloads the Zenodo RGB zips and the Drive split files into `gsl-continuous/`. It writes nothing elsewhere.
- `extract_arabsign` extracts the six RGB 7z archives already on the Volume to `arab-sign/RGB/videos/`. It checks each archive against the import manifest and reconciles the videos with the dataset author's split CSVs.

Both steps are the only ones that mount `datasets` writable. Every other function mounts it read-only.

The GSL Drive folder labelled "GSL SD" actually holds the `*_unseen` files. None of its 1,043 test sentences occurs in training, and they cover 45 unique sentences, so this folder matches the paper's Unseen-Sent split. "GSL SI" holds signer 3 out with seen sentences, matching the paper's Signer-Indep split.

The five ArabSign videos missing from the RGB release are `01/test/0006`, `01/test/0044`, `01/train/0020`, `02/train/0020` and `05/train/0042`, all with `(timestamp)_c.mp4` stems. Signer-Dep test therefore has 1,841 of 1,843 clips. In addition, 35 file stems carry a signer/sentence prefix that differs from their folder. The folder path is what the author's CSVs list, so it defines signer, split and sentence.

Frames for all three datasets are resized to 256×256 PNG with the upstream `resize_img` operation (`cv2.resize`, `INTER_LANCZOS4`) and archived on the paper's results Volume `alyami-2024-cslr-results`:

- **Phoenix2014:** VAC's `preprocess/dataset_preprocess.py`, byte-identical in all four repos, runs unchanged on a local copy.
- **ArabSign:** mp4s are decoded with `simple-video-utils` 0.7.4 (rgb24, checked against PyAV), then converted to BGR and resized.
- **GSL:** jpg frames are decoded from the zips and resized.

## Environment and patches

Nothing has been built yet. The plan is to start from `ghcr.io/sign-language-processing/reproduction:latest`, reusing the patches from `papers/yu-2025-cvsign` that are needed on current PyTorch: `ctcdecode` built as C++17, and the CTC loss computed without cuDNN. Each patch will be re-justified by this paper's own preflight.

## Execution evidence

No runs.

## Guesses and deviations

| Detail | Paper/evidence says | This attempt will use | Rationale |
| --- | --- | --- | --- |
| Learning rate, TLP/SEN epochs | arXiv v1: lr 1e-3, 40 epochs (steps 20/35) for all. Upstream: lr 1e-4, TLP/SEN 80 epochs (steps 40/60). ICCVW 2025: lr 1e-3, TLP/SEN 80 epochs (steps 30/60). | arXiv v1 values for all five models | Operator decision at gate `protocol-lr-epochs`; arXiv v1 is the assigned paper |
| TLP/SEN schedule evidence across versions | Table I vs ICCVW Table 1 (checked 2026-11-02 from both PDFs). All six TLP cells and five of six SEN cells are identical at one decimal. SEN GrSL Signer-Indep is 8.09 in v1, 8.0 in ICCVW Table 1 and 8.1 in ICCVW Fig. 3. Across all 30 shared cells, 26 are identical at one decimal, 3 match only by truncation (SMKD Phoenix SI 43.08→43.0, SEN GrSL SI 8.09→8.0, CorrNet ArabSign SI 7.65→7.6), and 1 differs (SMKD Phoenix SD 22.31→22.1). | 40 epochs (arXiv v1) | The numbers did not change when the stated schedule went from 40 to 80 epochs, so the evidence does not show which schedule produced them. The 80-epoch cost will be measured and put to the operator before the TLP/SEN full runs. |
| Seed | Not stated | 0 (upstream default) | Single reported number per cell |
| Checkpoint selection | Not stated | Dev-best per upstream `main.py`; the rule for ArabSign LOSO folds (no dev set) is to be fixed before launch | Upstream behavior |
| ArabSign Signer-Dep validation | 709 samples, unpublished | To be decided before launch | Split not released |

## Attempts, failures, and dead ends

None yet. The operator chose to stop before data population and preflight until the compute plan is approved.

## Candidate flags, ethics, and human evaluation

The queue record flags no human evaluation and no ethical concerns, and `copied_scores` is `no`. The text confirms the paper re-ran every model and copied no baselines. All three datasets are existing recordings released for research, and no participants are involved.

## Author and team contact

None. The ArabSign access shown in the queue record was coordinated earlier by Team S.

## Compute gate (resolved)

`compute-full-table` covers about 55 single-GPU trainings: Phoenix 10, ArabSign Signer-Dep 5, ArabSign LOSO 30, GrSL 10.

The estimate presented in `#repro-sign-team-r` on 2026-10-30 was about 420–610 L40S GPU-hours, or USD 1,350–2,000. It included about 15 GPU-hours of preflights and a 10% retry margin, at USD 3.22/h effective for an L40S with 16 cores and 64 GiB. The 30 ArabSign LOSO runs account for about USD 550–1,050 of that.

That estimate replaces the earlier GPU-only figure of USD 800–1,600, which left out CPU and memory charges, preflights and retries.

The project lead approved the full Table I on 2026-11-02, including the 30 ArabSign LOSO runs. The absolute ceiling is **USD 2,000**: about 620 L40S GPU-hours at USD 3.22/h, or about CHF 1,650. Blocks run in this order:

1. Phoenix SD
2. Phoenix SI
3. GrSL SI
4. GrSL Unseen-Sent
5. ArabSign SD
6. ArabSign LOSO

Each block declares its own ceilings, and cumulative spend is tracked against USD 2,000.
