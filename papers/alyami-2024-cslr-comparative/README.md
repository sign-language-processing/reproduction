# Alyami & Luqman (2024) — Comparative Study of CSLR Techniques: reproduction

**Paper ID:** `8d2b672d06ddb561f890798d7c1e434389c34202`

**Citation:** Sarah Alyami and Hamzah Luqman. *A Comparative Study of Continuous Sign Language Recognition Techniques.* arXiv:2406.12369v1, 2024. A revised version appeared in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, 2025, pp. 4923–4932.

**Paper:** https://arxiv.org/abs/2406.12369v1 · **Code/artifacts:** the study did not publish its own code. Each evaluated model's official repository is used (see Source provenance).

**Preference level:** 2

Level 2 is confirmed by this paper's own build, preflights and 55 full runs: the pinned official code runs through its own `main.py` with correctness patches and environment fixes only. See `status.preference_level_basis`.

**Pipeline status:** `complete` — all 30 in-scope targets were produced in six blocks of full runs (55 runs in total), each trained for 40 epochs in one Modal attempt, with no run reaching its ceiling. The compute gate `compute-full-table` set an absolute ceiling of USD 2,000; total Modal spend is USD 1,169.16.

**Numerical agreement:** `does_not_agree` — the tolerance is ±1.0 WER point, from the paper's own "around 1% fluctuations" for its Phoenix re-runs. 7 of the 30 values are within it:

- SMKD on Phoenix SD;
- CorrNet on Phoenix SI;
- VAC, SMKD and TLP on ArabSign SD;
- VAC and SMKD on ArabSign SI.

The other 23 are outside it, by between −25.80 and +87.61 points. Two qualifications apply:

- **ArabSign SD.** The published values are all below 1% WER, so a 1-point tolerance cannot distinguish them.
- **ArabSign SI.** The VAC and SMKD values are means over six folds, and not one of the 30 per-signer folds is within ±1.0 of its Table II value.

See `status.numerical_agreement_basis`.

**Attempt dates:** 2026-10-26 (intake and gates); 2026-11-02 to 2026-11-05 (data, preflights, blocks 1–6, investigation, close-out)

## Reproduction agents

| Agent ID | Model and version | Agent application | Contribution | Attribution evidence / unknowns |
| --- | --- | --- | --- | --- |
| `claude-opus-5-5-claude-code` | Claude Opus 5.5 (`claude-opus-5-5`) | Claude Code 2.1.310 | Assignment intake, version resolution, source search, dataset inspection on Modal `repro-sign`, target/protocol/compute gates | Session system context names the model and ID. `claude --version` returned 2.1.310. No runs executed. |
| `claude-opus-5-5-claude-code-2026-11-02` | Claude Opus 5.5 (`claude-opus-5-5`) | Claude Code 2.1.316 | From 2026-11-02: gate updates, data population, container, preflights, estimates, launching and monitoring full runs. On 2026-11-04: collecting blocks 2–5, the no-training investigation, and declaring and launching block 6. On 2026-11-05: collecting block 6, the environment freeze, the LOSO targets and the final status. Executing agent of every retained run. | Session system context names the model and ID. `claude --version` returned 2.1.316 on 2026-11-02 and 2.1.318 on 2026-11-04 and 2026-11-05, in the same continued session. Whether the running process changed version is not exposed. |

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

The ICCVW 2025 version (Sec. 3.3) instead says TLP and SEN were trained for 80 epochs, with steps 30/60. The decision stands because arXiv v1 is the assigned paper. Yet the TLP and SEN Table I values are the same in both versions: all six TLP cells and five of six SEN cells are identical at one decimal. So the change in stated schedule did not come with new numbers, and the evidence does not show which schedule produced them. The measured cost of an 80-epoch schedule was reported on 2026-11-02, and the operator kept 40 epochs for TLP and SEN (see Gates).

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

All 30 targets have been produced. The WER is sclite's exact (S+D+I)/N on the test split.

- **Difference:** reproduced minus published (arXiv v1 Table I), so positive means a higher error.
- **Relative:** that difference divided by the published value.
- **Precision:** ArabSign Signer-Dep values keep the paper's four decimals.
- **ArabSign Signer-Indep (LOSO):** each value is the plain arithmetic mean of six per-signer test WERs. Each fold trains on the other five signers and scores its final epoch, 39. The paper's Table II Avg column is the plain mean of its six per-signer values, to rounding, for all five models. The per-fold values follow the main table.

| Target | Published | Reproduced | Difference | Relative | Selected epoch (dev WER) | Runs |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| VAC, Phoenix2014 Signer-Dep | 21.94 | 29.67 | +7.73 | +35.2% | 35 (29.64) | `full-vac-phoenix-sd-001`, `eval-vac-phoenix-sd-001` |
| SMKD, Phoenix2014 Signer-Dep | 22.31 | 22.14 | −0.17 | −0.8% | 36 (21.57) | `full-smkd-phoenix-sd-001`, `eval-smkd-phoenix-sd-001` |
| TLP, Phoenix2014 Signer-Dep | 22.0 | 26.58 | +4.58 | +20.8% | 39 (26.19) | `full-tlp-phoenix-sd-001`, `eval-tlp-phoenix-sd-002` |
| SEN, Phoenix2014 Signer-Dep | 20.2 | 24.48 | +4.28 | +21.2% | 36 (24.75) | `full-sen-phoenix-sd-001`, `eval-sen-phoenix-sd-001` |
| CorrNet, Phoenix2014 Signer-Dep | 19.9 | 21.54 | +1.64 | +8.2% | 38 (21.34) | `full-corrnet-phoenix-sd-001`, `eval-corrnet-phoenix-sd-002` |
| VAC, Phoenix2014 Signer-Indep (SI5) | 40.4 | 50.39 | +9.99 | +24.7% | 28 (51.41) | `full-vac-phoenix-si-001` |
| SMKD, Phoenix2014 Signer-Indep (SI5) | 43.08 | 36.88 | −6.20 | −14.4% | 39 (36.68) | `full-smkd-phoenix-si-001` |
| TLP, Phoenix2014 Signer-Indep (SI5) | 35.7 | 47.34 | +11.64 | +32.6% | 28 (47.56) | `full-tlp-phoenix-si-001` |
| SEN, Phoenix2014 Signer-Indep (SI5) | 34.4 | 40.56 | +6.16 | +17.9% | 26 (40.10) | `full-sen-phoenix-si-001` |
| CorrNet, Phoenix2014 Signer-Indep (SI5) | 33.6 | 34.40 | +0.80 | +2.4% | 39 (35.22) | `full-corrnet-phoenix-si-001` |
| VAC, GrSL Signer-Indep | 29.6 | 3.80 | −25.80 | −87.2% | 37 (4.72) | `full-vac-grsl-si-001` |
| SMKD, GrSL Signer-Indep | 30.8 | 5.06 | −25.74 | −83.6% | 33 (6.39) | `full-smkd-grsl-si-001` |
| TLP, GrSL Signer-Indep | 6.4 | 25.76 | +19.36 | +302.5% | 37 (27.73) | `full-tlp-grsl-si-001` |
| SEN, GrSL Signer-Indep | 8.09 | 38.54 | +30.45 | +376.4% | 36 (42.32) | `full-sen-grsl-si-001` |
| CorrNet, GrSL Signer-Indep | 1.5 | 89.11 | +87.61 | +5,840.7% | 8 (89.44) | `full-corrnet-grsl-si-001` |
| VAC, GrSL Unseen-Sent | 74.92 | 81.71 | +6.79 | +9.1% | 36 (72.10) | `full-vac-grsl-unseen-001` |
| SMKD, GrSL Unseen-Sent | 61.1 | 70.96 | +9.86 | +16.1% | 30 (64.30) | `full-smkd-grsl-unseen-001` |
| TLP, GrSL Unseen-Sent | 75.5 | 92.14 | +16.64 | +22.0% | 7 (89.49) | `full-tlp-grsl-unseen-001` |
| SEN, GrSL Unseen-Sent | 74.8 | 85.10 | +10.30 | +13.8% | 37 (85.06) | `full-sen-grsl-unseen-001` |
| CorrNet, GrSL Unseen-Sent | 64.7 | 92.72 | +28.02 | +43.3% | 6 (91.02) | `full-corrnet-grsl-unseen-001` |
| VAC, ArabSign Signer-Dep | 0.2787 | 0.3838 | +0.1051 | +37.7% | 27 (0.18) | `full-vac-arabsign-sd-001` |
| SMKD, ArabSign Signer-Dep | 0.2091 | 0.2442 | +0.0351 | +16.8% | 36 (0.18) | `full-smkd-arabsign-sd-001` |
| TLP, ArabSign Signer-Dep | 0.2439 | 1.1165 | +0.8726 | +357.8% | 36 (0.36) | `full-tlp-arabsign-sd-001` |
| SEN, ArabSign Signer-Dep | 0.2962 | 36.8981 | +36.6019 | +12,357.2% | 38 (33.65) | `full-sen-arabsign-sd-001` |
| CorrNet, ArabSign Signer-Dep | 0.2265 | 8.8625 | +8.6360 | +3,812.8% | 20 (6.65) | `full-corrnet-arabsign-sd-001` |
| VAC, ArabSign Signer-Indep (LOSO mean) | 6.97 | 6.16 | −0.81 | −11.6% | 39 in every fold (final epoch) | `full-vac-arabsign-loso-{01…06}-001` |
| SMKD, ArabSign Signer-Indep (LOSO mean) | 4.24 | 4.29 | +0.05 | +1.2% | 39 in every fold | `full-smkd-arabsign-loso-{01…06}-001` |
| TLP, ArabSign Signer-Indep (LOSO mean) | 2.9 | 8.03 | +5.13 | +176.9% | 39 in every fold | `full-tlp-arabsign-loso-{01…06}-001` |
| SEN, ArabSign Signer-Indep (LOSO mean) | 7.22 | 32.36 | +25.14 | +348.2% | 39 in every fold | `full-sen-arabsign-loso-{01…06}-001` |
| CorrNet, ArabSign Signer-Indep (LOSO mean) | 7.65 | 59.41 | +51.76 | +676.6% | 39 in every fold | `full-corrnet-arabsign-loso-{01…06}-001` |

Phoenix SD counts on the 629 test sentences (6,504 words): VAC 1,930 errors, SMKD 1,440, TLP 1,729, SEN 1,592, CorrNet 1,401. Each target from blocks 2–6 cites its runs' `{run_id}-sclite-test` artifacts, with exact error and word counts.

**ArabSign Signer-Indep per tested signer.** Each cell gives the reproduced test WER at epoch 39, then the arXiv v1 Table II value in parentheses. Each LOSO target's `result.per_fold` also records the sclite counts.

| Model | Signer 01 | 02 | 03 | 04 | 05 | 06 | Mean (Table II Avg) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VAC | 3.54 (7.55) | 1.86 (5.03) | 6.50 (8.67) | 15.76 (5.66) | 4.02 (8.17) | 5.25 (6.76) | 6.16 (6.97) |
| SMKD | 2.68 (5.88) | 6.95 (2.93) | 3.57 (2.11) | 2.59 (4.52) | 4.00 (1.89) | 5.93 (8.11) | 4.29 (4.24) |
| TLP | 5.43 (2.91) | 1.78 (0.53) | 6.06 (2.09) | 8.19 (1.14) | 9.58 (1.29) | 17.16 (9.46) | 8.03 (2.90) |
| SEN | 16.24 (5.84) | 25.16 (5.70) | 12.99 (6.20) | 64.39 (10.90) | 26.19 (5.40) | 49.17 (9.30) | 32.36 (7.22) |
| CorrNet | 32.99 (10.50) | 99.39 (5.50) | 26.66 (6.70) | 51.68 (11.10) | 99.35 (3.80) | 46.39 (8.30) | 59.41 (7.65) |
| Test words | 4,771 | 4,949 | 5,064 | 4,822 | 4,647 | 4,686 | |

Pooling the six folds' sclite counts instead (errors over words) gives the following. This is recorded as a note only, in each target's `result.word_weighted_mean_note`:

| Model | Pooled WER | Change from the plain mean |
| --- | ---: | ---: |
| VAC | 6.16 | 0.00 |
| SMKD | 4.29 | 0.00 |
| TLP | 7.94 | −0.09 |
| SEN | 32.15 | −0.21 |
| CorrNet | 59.18 | −0.23 |

No comparison against the ±1.0 tolerance changes.

**Agreement basis.** The ±1.0-point tolerance is kept as recorded.

- **ArabSign Signer-Dep.** The published values are all below 1% WER, so any reproduced value under about 1.2% falls inside the tolerance. VAC (+37.7%), SMKD (+16.8%) and TLP (+357.8%) are "within" on that column only in this absolute sense.
- **ArabSign Signer-Indep.** VAC and SMKD are within at the level of the six-fold mean, but none of the 30 folds is within ±1.0 of its Table II value. VAC's folds differ by −4.15 to +10.10, and SMKD's by −3.20 to +4.02.

**Results by code family.** These are facts under the stated protocol (lr 1e-3, 40 epochs, steps 20/35, batch 2, seed 0), without interpretation:

- VAC and SMKD (repository VAC_CSLR) train on every dataset. In all four settings of blocks 2–5 their lowest dev WER comes late (epochs 27–39), far below their epoch-0 value. Their test WERs are:
  - 3.80 and 5.06 on GrSL SI;
  - 81.71 and 70.96 on GrSL Unseen-Sent;
  - 0.3838 and 0.2442 on ArabSign SD;
  - 6.16 and 4.29 on ArabSign SI, the six-fold LOSO mean.

  In every LOSO fold, their WER on the 50-clip training token list is 0.0 at epoch 39.
- TLP, SEN and CorrNet (repositories Temporal-Lift-Pooling, SEN_CSLR and CorrNet) train on Phoenix SD and SI, but degrade or stall on GrSL and ArabSign:
  - **GrSL SI:** TLP reaches 25.76, SEN 38.54 and CorrNet 89.11, where VAC/SMKD reach 3.80/5.06; the paper has the opposite order (TLP 6.4, SEN 8.09, CorrNet 1.5, VAC/SMKD about 30).
  - **GrSL Unseen-Sent:** TLP's and CorrNet's dev WER stays at about 89–100% for all 40 epochs, and dev-best selection picks epochs 7 and 6.
  - **ArabSign SD:** SEN's dev WER stays at 83–95% for the first 21 epochs and ends at 36.90 on test; CorrNet ends at 8.86; TLP reaches 1.12.
  - **ArabSign SI (LOSO):** TLP reaches 8.03 (folds 1.78–17.16), SEN 32.36 (12.99–64.39) and CorrNet 59.41 (26.66–99.39). In CorrNet folds 02 and 05 every test hypothesis uses the same two glosses, and training skipped 350 and 82,755 batches for a non-finite loss.
- The two families differ in one training-loop fact at the pinned commits. VAC_CSLR's `seq_train` runs in fp32. TLP, SEN and CorrNet run the forward pass and loss under `torch.cuda.amp.autocast()` with a `GradScaler`. All four skip a batch whose loss is non-finite.
- The data path is identical across the four repositories for GrSL SI, GrSL Unseen-Sent and ArabSign SD (`adapter-check-002`, below). The LOSO folds were not part of that check; they use the same ArabSign frames, gloss dictionary and list format.

**Block 1 observations.** These are recorded as evidence, without interpretation, and are also in `compute-full-table.resolution.block_ledger`:

- The lowest dev WER came late for all five models. The best epochs, 0-based out of 40, were VAC 35, SMKD 36, TLP 39, SEN 36 and CorrNet 38.
- TLP's dev WER was still falling at its final (40th) epoch: 26.7, 26.8, 26.5 and 26.2 at epochs 36–39. The ICCVW 80-epoch schedule for TLP was not adopted (see Guesses and deviations).
- VAC is +7.73 against the paper, while SMKD is within 0.17. SMKD comes from the same repository, with the same data lists and scoring, and differs only in `share_classifier` and `weight_norm`. Both use SeqCTC 1 + ConvCTC 1, the loss weights recorded as a guess. This rules out a data or scoring fault for this block.

**Convergence curves.** Every full run's upstream per-epoch dev WER log (`work/dev.txt`) is copied verbatim to `artifacts/{run_id}-dev.txt`. Each copy is recorded as artifact `{run_id}-dev-log` with its sha256 and its source URI on the results Volume. This is done for all 55 full runs of blocks 1–6. Each run in blocks 2–6 also records three more logs by URI and sha256: its full training log (`{run_id}-train-log`), stdout (`{run_id}-stdout`) and attempt records (`{run_id}-attempt-records`). In the LOSO runs, `dev.txt` holds the WER on the fixed 50-clip training token list, which is training data and is never used for selection.

**Block 2–6 observations.** These are recorded as evidence, without interpretation, and are also in `compute-full-table.resolution.block_ledger`. Every run trained 40 epochs in one Modal attempt, and none reached its ceiling.

The table counts training batches skipped for a non-finite loss over 40 epochs, from each run's `stdout.log` (`runs[].skipped_training_batches`):

| Setting | VAC | SMKD | TLP | SEN | CorrNet | Batches per epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phoenix SI | 4 | 1 | 4 | 2 | 2 | 2,187 |
| GrSL SI | 9 | 8 | 11 | 8 | 21,566 | 4,410 |
| GrSL Unseen-Sent | 0 | 0 | 0 | 0 | 109,472 | 4,094 |
| ArabSign SD | 0 | 0 | 0 | 0 | 0 | 3,390 |
| ArabSign LOSO, 6 folds | 0 | 0 | 0 | 0 | 83,105 (fold 02: 350 from epoch 15; fold 05: 82,755 from epoch 9; others 0) | 3,845–3,915 |

**Block 6 (ArabSign LOSO) observations.**

- **Outcome.** All 30 runs finished in one attempt. Containers started between 09:54:48 and 10:13:32 UTC on 2026-11-04, and the last run finished at 19:43:36.
- **Token-list WER at epoch 39.** The 50-clip list is training data. VAC, SMKD and TLP score 0.0 in every fold. SEN scores 1.3–21.9.
- **CorrNet token-list WER.** CorrNet scores 10.3–99.4 at epoch 39 and swings widely from epoch to epoch. Fold 01, for example, moves between 3.9 and 36.8 over epochs 16–39.
- **CorrNet collapsed folds.** Folds 02 and 05 score 99.4 on the token list and on test. Their 1,597 and 1,499 test hypotheses use only two distinct glosses, recorded in each run's `failure_signature` with the sclite alignment.
- **Not checked.** The block 6 checkpoints were not scanned for non-finite values.
- **SEN empty outputs.** SEN fold 06 has 11 empty test hypotheses. No other LOSO run has any.

In GrSL Unseen-Sent, 176 of 5,187 dev and 105 of 4,490 test reference gloss tokens are glosses that never occur in its training list. GrSL SI and ArabSign SD have none.

### Investigation of 2026-11-04 (no training)

The operator held block 6 and asked for two checks before deciding on LOSO. Both ran CPU-only on Modal against the retained run directories.

**1. Data adapters across the four repositories (`adapter-check-002`, `modal_app.py::adapter_check`).** For each of GrSL SI, GrSL Unseen-Sent and ArabSign SD, the check copies each of the five full runs' own code directory (`runs/full-{model}-{setting}-001/code`). It then compares what those copies hold and runs each repo's own `BaseFeeder`, built as its `main.py` builds it, on the same 20 clips (10 train, 10 test, evenly spaced). **No difference was found:**

| Compared | GrSL SI | GrSL Unseen-Sent | ArabSign SD |
| --- | --- | --- | --- |
| Train / dev / test lists (every row: fileid, folder, signer, label, frame count) | identical (8,821 / 588 / 881) | identical (8,189 / 1,063 / 1,043) | identical (6,780 / 709 / 1,841) |
| Gloss dictionary (size, every entry) | identical, 306 | identical, 310 | identical, 93 |
| Ground-truth STMs for scoring | identical | identical | identical |
| Label ids and glosses of the 20 clips | identical, equal to the list label | identical, equal to the list label | identical, equal to the list label |
| Frames read per clip (20 clips) | identical, equal to the list count (24–183) | identical (15–226) | identical (51–152) |
| Frame size read | 256×256×3 | 256×256×3 | 256×256×3 |
| Test transform (CenterCrop 224) output | bit-identical tensors | bit-identical | bit-identical |
| Train transform at seed 0 (RandomCrop 224, flip, TemporalRescale 0.2) | bit-identical tensors | bit-identical | bit-identical |

The only difference is in the `feeder_args` keys. TLP adds `frame_interval: 1` and `image_scale: 1.0`; SEN and CorrNet add those plus `input_size: 224`; VAC_CSLR has none of them and hardcodes the same values. The identical tensors show the outputs do not differ. The preprocessing helper `preprocess/dataset_preprocess.py`, `evaluation/slr_eval/preprocess.sh`, `mergectmstm.py` and `utils/decode.py` are byte-identical across the four repos. `utils/video_augmentation.py` differs only in options these configs do not use. `evaluation/slr_eval/wer_calculation.py` differs in how it returns sclite's result and in an optional Python scorer; every run scores dev (during training) and test with sclite.

**2. Failure signatures.** These are recorded in full in `reproduction.json`, in each run's `failure_signature` (per-epoch dev WER, mean training loss and skipped batches; sclite counts; hypothesis statistics; checkpoint scan). The checkpoint scan is `checkpoint-check-002`.

| | CorrNet, GrSL SI | CorrNet, GrSL Unseen-Sent | SEN, ArabSign SD |
| --- | --- | --- | --- |
| Selected checkpoint | epoch 8 | epoch 6 | epoch 38 |
| Test S / D / I (words) | 1,579 / 1,428 / 110 (3,498) | 1,980 / 2,158 / 25 (4,490) | 1,241 / 697 / 177 (5,732) |
| Dev S / D / I (words) | 1,055 / 957 / 72 (2,330) | 1,959 / 2,714 / 48 (5,187) | 453 / 222 / 64 (2,196) |
| Empty test hypotheses | 0 of 881 | 6 of 1,043 | 1 of 1,841 |
| Distinct glosses / sequences in test hypotheses | 29 / 75 | 33 / 83 | 92 / 308 |
| Dev WER by epoch | 89.4–91.6 for epochs 0–8, then exactly 104.1 at every epoch 9–39 | 91.0–99.5 for epochs 0–12, then exactly 100.7 at every epoch 13–39 | 83.3–95.3 for epochs 0–20; 63.5 at epoch 21 (first epoch at lr 2e-4), falling to 33.7 at epoch 38; 41.0 at 39 |
| Mean training loss by epoch | 54.3 → 36.9 (epoch 9) → 16.3 (epoch 20); up to 18.7 at epoch 28; 11.5 at epoch 39 | 51.0 → 28.9 (epoch 13); then `nan` in 18 of the remaining 26 epochs, other values 13.1–30.1 | 34.9 → 14.0 (epoch 20); 19.6–24.3 in epochs 21–27; 18.8 at epoch 39 |
| Training batches skipped (non-finite loss) | 0–4 per epoch for epochs 0–9; 354 at epoch 10, peak 1,967 at epoch 18, 0–5 from epoch 31 | 0 for epochs 0–12; 3,038 at epoch 13; 4,092–4,094 of 4,094 at every epoch 14–39 | 0 in every epoch |
| Final-epoch (39) dev output | the same two glosses, `ΠΛΗΡΩΝΩ ΠΗΓΑΙΝΩ`, for all 588 dev clips | the same two glosses, `ΠΙΝΩ ΠΕΡΙΟΔΟΣ`, for all 1,063 dev clips | 196 distinct sequences; 11 empty |
| Non-finite values in checkpoints | epoch 39: 2,306 values, only in BatchNorm buffers `conv1d.temporal_conv.1` and `.5` (`running_mean`, `running_var`); no model parameter. Epoch 8: none. | epoch 39: 2,108 values in the same four buffers; no parameter. Epoch 6: none. | none (epochs 38 and 39) |

For comparison, the checkpoint scan found no non-finite value in any retained checkpoint of `full-corrnet-arabsign-sd-001`, `full-tlp-grsl-si-001`, `full-sen-grsl-si-001` or `full-vac-grsl-si-001`. Hypotheses are never empty in the sense of a missing line: upstream's `mergectmstm.py` writes `[EMPTY]` for a sentence with no gloss, and those are counted as empty above. The evaluated (dev-best) checkpoints of both CorrNet GrSL runs predate the frozen-output epochs.

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

The preflight, training and evaluation commands are listed per run in `reproduction.json.runs`. Full runs and their evaluation have this form:

```bash
$W run --detach $A::launch_train --model {vac|smkd|tlp|sen|corrnet} --setting {setting} --run-id {run_id} [--dev-rule {rule}]
$W run --detach $A::launch_evaluate --train-run-id {run_id} --eval-run-id {eval_run_id}
$W run $A::launch_arabsign_split_record   # re-derives the ArabSign split hashes recorded in the gates
$W run --detach $A::launch_run_full --model {model} --setting {setting} --run-id {run_id} --eval-run-id {eval_run_id} --max-gpu-hours {ceiling} [--dev-rule {rule}]   # blocks 2-5: train + evaluate in one detached call
$W run --detach $A::launch_adapter_check   # no training: cross-repo data-adapter comparison on the retained run directories
$W run --detach $A::launch_checkpoint_check --run-ids {run_id},{run_id},...   # no training: non-finite values per checkpoint tensor
$W run $A::launch_inspect_environment   # no training: versions, pip freeze and image ID of the training image
```

Block 6 runs once per model and fold:

```bash
$W run --detach $A::launch_run_full --model {model} --setting arabsign-loso-{01..06} --run-id full-{model}-arabsign-loso-{FF}-001 --eval-run-id eval-{model}-arabsign-loso-{FF}-001 --max-gpu-hours {ceiling} --dev-rule final-epoch
```

The exact command for each of the 30 runs is in `reproduction.json.runs`. Each LOSO target is the plain mean of the six `sclite_counts.test.wer_percent_exact` values in `runs/eval-{model}-arabsign-loso-{FF}-001/evaluation.json`.

The settings are `phoenix-sd`, `phoenix-si`, `grsl-si`, `grsl-unseen`, `arabsign-sd` (`--dev-rule stratified-709-seed42`) and `arabsign-loso-01` … `-06` (`--dev-rule final-epoch`).

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

**GrSL gloss coverage (fact; `datasets[grsl].gloss_coverage`, from `adapter-check-002`).** In GrSL Unseen-Sent, 176 of the 5,187 dev reference gloss tokens and 105 of the 4,490 test reference gloss tokens are glosses that never occur in its training list. In GrSL SI (and ArabSign SD) there are none.

The five ArabSign videos missing from the RGB release are `01/test/0006`, `01/test/0044`, `01/train/0020`, `02/train/0020` and `05/train/0042`, all with `(timestamp)_c.mp4` stems. Signer-Dep test therefore has 1,841 of 1,843 clips. In addition, 35 file stems carry a signer/sentence prefix that differs from their folder. The folder path is what the author's CSVs list, so it defines signer, split and sentence.

Frames for all three datasets are resized to 256×256 PNG with the upstream `resize_img` operation (`cv2.resize`, `INTER_LANCZOS4`) and archived on the paper's results Volume `alyami-2024-cslr-results`:

- **Phoenix2014:** VAC's `preprocess/dataset_preprocess.py`, byte-identical in all four repos, runs unchanged on a local copy.
- **ArabSign:** mp4s are decoded with `simple-video-utils` 0.7.4 (rgb24, checked against PyAV), then converted to BGR and resized.
- **GSL:** jpg frames are decoded from the zips and resized.

## Environment and patches

`Dockerfile` starts from `ghcr.io/sign-language-processing/reproduction:latest` (NGC PyTorch 26.02: torch 2.11.0a0, CUDA 13.1, cuDNN 9.19, Python 3.12, numpy 2.1). On top of that it adds:

- SCTK `sclite` (`9688a26`);
- `ctcdecode` (`c90ad94`);
- `simple-video-utils` 0.7.4 with `av` 18.1.0;
- `lmdb` 1.5.1, which TLP's and SEN's `dataset/dataloader_video.py` import at module level;
- the four repos at their pinned commits under `/opt/repos/{vac,tlp,sen,corrnet}`, each with one correctness patch (`patches/{repo}-ctc-without-cudnn.patch`, below).

**Training precision (fact, unchanged by this reproduction; `sources[].training_precision`).** VAC_CSLR (the VAC and SMKD runs) trains in fp32: its `seq_scripts.py` `seq_train` has no autocast or GradScaler. TLP, SEN and CorrNet train in fp16 mixed precision. Their `seq_train` creates a `GradScaler` at the start of every epoch and runs the forward pass and loss under `torch.cuda.amp.autocast()` (TLP lines 22/29, SEN and CorrNet lines 19/26 at the pinned commits). All four skip a batch whose loss is inf or nan.

The repos' own Python pins (numpy 1.20, scipy 1.7, opencv 4.5.5) have no Python 3.12 wheels. They are replaced by the oldest releases that do and that still ship `scipy.misc`, as in `papers/yu-2025-cvsign`.

**Image and dependency freeze.** Every full run, blocks 1–6, records Modal image `im-RRhJE2W44Meiwq3PVXQMMX` in its `train_calls.jsonl`. `inspect-environment-001` ran on 2026-11-05 from the unchanged `Dockerfile` and patches, and Modal reused that same image ID. Its output is in `artifacts/inspect-environment-001.json`:

- Python 3.12.3;
- torch 2.11.0a0+eb65b36914.nv26.02, CUDA 13.1, cuDNN 91900;
- numpy 2.1.0;
- the full `pip freeze` of 268 packages (sha256 `1f39e822fa855d25…`);
- the four repos' HEADs, which equal the pinned commits.

The `Dockerfile` is recorded as artifact `dockerfile`, with sha256 `85e8c0dd…`.

Each run works on its own copy of one repo. The harness adds only data files and configs to that copy:

- the setting's info lists, gloss dict and STMs, written with the repo's own `generate_gt_stm` and `sign_dict_update`;
- a `configs/{dataset}.yaml`;
- `configs/run.yaml`. This is the pinned `configs/baseline.yaml` with only the gate-approved keys changed, and each run records the exact delta.

Tracked upstream files stay unchanged, and each run records the hash of `git diff HEAD`.

Phoenix Signer-Dep uses each repo's own `phoenix2014` dataset config. SEN and CorrNet use their committed lists, which are identical to each other. VAC and TLP commit no lists, so they use the output of the byte-identical `preprocess/dataset_preprocess.py`.

Every other setting is written in the same format under a dataset name containing `phoenix`. In TLP, SEN and CorrNet that substring is the only thing that selects the train/dev/test loaders and the frame-folder glob. VAC always reads `./preprocess/phoenix2014/`. So no upstream code needs changing for the new datasets.

Patches and environment fixes, each justified by this paper's own build or preflight. The patch files are recorded as `patch-*` artifacts with these sha256 values:

| Patch | sha256 |
| --- | --- |
| `patches/ctcdecode-cxx17.patch` | `cb68df51440663569382513520196372d38c412da2bead8dcc7fc01c2eab24ae` |
| `patches/vac-ctc-without-cudnn.patch` | `3f82f3157ba97586fc7d2ac6e95402d3779a7aacec7b81200eb98dc68bdfb65c` |
| `patches/tlp-ctc-without-cudnn.patch` | `b2c6de79fa245b7f3a3af380238a23efd5c02e892909aa9e4a105411df06f4ca` |
| `patches/sen-ctc-without-cudnn.patch` | `54663cb575f67f2c968455bbae462f90b63f0259df910e78bccf0a74673d37e6` |
| `patches/corrnet-ctc-without-cudnn.patch` | `9bfa60a2aabc76c0ad031daf59a5b0ddc4d03205b1b672aa41aedcb1b2ba7e05` |

| Item | Why | Evidence |
| --- | --- | --- |
| `patches/ctcdecode-cxx17.patch` | Unpatched `ctcdecode` does not compile against current PyTorch: `#error C++17 or later compatible compiler is required to use PyTorch`. The patch builds as C++17 and drops five KenLM dynamic exception specifications. No decoder logic changes. It is identical to the yu-2025-cvsign patch. | Image build `im-zwwdoz4GF7RwJO4BnSPdG6` failed 2026-11-02 |
| `simple-video-utils==0.7.4` installed in the Dockerfile | The published `reproduction:latest` image lacks it, although the root Dockerfile pins it. | Build `im-L4hfuloTR54towJj5WD1nD`: `ModuleNotFoundError` |
| `av==18.1.0` | `simple-video-utils` requires `av>=18`, which resolved to 19.0.2. That version rejects the `metadata_errors` argument `simple-video-utils` passes. | App `ap-OMIRjJ8IzhzSco6yzVHx7u`: `open() got an unexpected keyword argument 'metadata_errors'` |
| `patches/{vac,tlp,sen,corrnet}-ctc-without-cudnn.patch` | On this stack `nn.CTCLoss` sends the repos' CTC call (CPU int32 targets) to cuDNN even when the two clips in a batch differ in length. PyTorch 1.x used cuDNN only when every input length equalled T. cuDNN's loss value is right, but its gradient is wrong for unequal lengths. Disabling cuDNN for the CTC call alone restores the native computation; nothing else changes. Same change as yu-2025-cvsign. | Fixed-batch gradients vs a float64 CPU reference: cosine 0.06–0.96 with cuDNN, 1.0000 without, and equal-length batches match. Under AMP, GradScaler skipped 26–50 of 60 updates with cuDNN against 5–8 without. With the patch, the as-is and cuDNN-off arms coincide (`ctc-diagnostics-*-phoenix-001`, `-patched-001`, `preflight-*-002/003`). |
| `lmdb==1.5.1` | TLP's and SEN's loaders `import lmdb`; no config uses the lmdb datatype. | `preflight-{tlp,sen}-*-001`: `ModuleNotFoundError` |
| `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` | Upstream resumes with a bare `torch.load` of checkpoints that hold numpy RNG state. PyTorch ≥ 2.6 refuses these by default. This is an environment variable, not a code change. | As in yu-2025-cvsign |

Two full-run settings change no computation:

- `PYTORCH_ALLOC_CONF=expandable_segments:True`. VAC and SMKD reached 45.2 GB of the L40S's 46 GB with the longest clips, and this setting only changes how freed memory is reused.
- 32 GiB of host memory instead of 64. Preflight host peaks were at most 19.5 GiB.

`ctc_check.py` is a diagnostic only. It builds each repo's own `Processor` from `main.py`, runs its own `seq_train` on real preflight batches, and compares CTC gradients on fixed batches.

## Execution evidence

All runs are in `reproduction.json.runs`, with Modal app and function-call IDs, image IDs, timestamps and declared ceilings. Outputs live under `runs/{run_id}/` on the Modal Volume `alyami-2024-cslr-results`.

**Data jobs, 2026-11-02:**

| Job | App | Result |
| --- | --- | --- |
| `extract_arabsign` | `ap-3EEqRJGxaxKfMN9NLO8W1O` | 9,330 mp4 |
| `populate_gsl` | `ap-Sg1meM9jpuHIbOLcF5I5bd` | 40.17 GB |
| Phoenix frames | `ap-Iew7yQSxjZSIdEXuS1iphB` | 70.2 GB, sha256 `3833a87f…` |
| ArabSign frames | `ap-0LKt5eSKIfpberp3xYdJon` | 1,104,581 frames, 91.5 GB, sha256 `27bf5694…` |
| GSL frames | `ap-RIAXU8Tv7XUNX8wBsYkvB3` | 1,036,155 frames, 91.1 GB, sha256 `09f596dd…` |

**Preflights (15, one per model × dataset).** Each uses 120 random plus the 16 longest training clips, and 40 random plus the 8 longest dev and test clips. It runs upstream `main.py` for 2 epochs, resumes from the epoch-1 checkpoint for a 3rd, then runs the test phase with sclite scoring. Every model × dataset passed. ArabSign used a provisional random 709-clip dev list, for cost measurement only.

Rates per model (train / eval ms per frame, with peak GPU memory from nvidia-smi):

| Model | Phoenix | GrSL | ArabSign | Peak GPU (GB) |
| --- | --- | --- | --- | --- |
| VAC | 0.73 / 1.01 | 0.46 / 2.13 | 0.77 / 1.01 | 45.2 |
| SMKD | 0.73 / 1.01 | 0.51 / 1.85 | 0.77 / 1.01 | 42.9 |
| TLP | 0.72 / 0.63 | 0.80 / 0.85 | 0.81 / 0.58 | 22.6 |
| SEN | 1.00 / 0.76 | 0.79 / 1.14 | 1.04 / 0.87 | 21.8 |
| CorrNet | 0.86 / 0.88 | 0.70 / 1.28 | 0.83 / 1.01 | 29.8 |

Each upstream checkpoint is about 385 MB.

**Scoring check (`score_check`).** A perfect hypothesis CTM scores 0.0% on Phoenix SI5, both GrSL splits, ArabSign SD and one LOSO fold. Dropping each sentence's first gloss gives exactly sentences ÷ reference words, so upstream's `preprocess.sh` plus sclite handles Greek and Arabic glosses unchanged. Apps are `ap-KcrIWUrzHeRYOQbuw2fxzc` and `ap-INgCesOLEX4oXS0KpfsCMZ`. Because sclite logs WER to one decimal and the paper reports ArabSign Signer-Dep WERs like 0.2787, evaluation also records sclite's raw error and word counts.

**Measured estimate (`compute-full-table.resolution.measured_estimate`).** This assumes 40 epochs for all models at USD 2.96/h (L40S, 16 cores, 32 GiB), 15 min overhead per run, and no retry margin:

| Block | Runs | USD |
| --- | ---: | ---: |
| Phoenix SD | 5 | 122 |
| Phoenix SI | 5 | 88 |
| GrSL SI | 5 | 115 |
| GrSL Unseen-Sent | 5 | 123 |
| ArabSign SD | 5 | 129–161 (by dev rule) |
| ArabSign LOSO | 30 | 803–965 (by checkpoint rule) |
| **Total** | 55 | **about 1,380** (cheapest rules) |

Per run this is 5.3–10.6 h. The paper reports about 10 h per Phoenix/GrSL run on an RTX A6000. The per-frame rates come from 136-clip epochs with worker start-up included, so they are more likely high than low.

**Spend before the full runs:** USD 16.08 on 2026-11-02, from Modal billing summed over this paper's apps, covering data jobs, image builds, preflights and diagnostics.

**Block 1, Phoenix SD (complete, 2026-11-03 01:45 UTC).**

| Model | Training time (h) | Estimate (h) |
| --- | ---: | ---: |
| VAC | 6.15 | 7.65 |
| SMKD | 6.16 | 7.65 |
| TLP | 4.61 | 7.23 |
| SEN | 8.32 | 9.85 |
| CorrNet | 6.94 | 8.75 |

Each model trained in a single attempt. Two evaluations needed a second attempt, both rerun unchanged:

- The first TLP evaluation hit a GPU hardware fault (XID 154, CUDA unknown error) before loading anything; the rerun is `eval-tlp-phoenix-sd-002`.
- The first CorrNet evaluation was stopped when the local Modal client lost its connection, before the function had written anything; the rerun is `eval-corrnet-phoenix-sd-002`. Evaluations now run with `modal run --detach`.

The block cost USD 98.15 (Modal billing per app, training and evaluation), against an estimate of USD 122. Cumulative spend is USD 114.69 of USD 2,000. The block ceiling is USD 228. That is the sum of the per-run GPU-hour ceilings (14 + 14 + 16 + 14 + 18 GPU-h) at USD 3.0/h, against an estimate of USD 122.

| Run | Launched (UTC) | App | Declared ceiling |
| --- | --- | --- | --- |
| `full-vac-phoenix-sd-001` | 14:05 | `ap-Bq2sMC5UcEDiiAY8EoOOXV` | 14 GPU-h |
| `full-smkd-phoenix-sd-001` | 14:05 | `ap-HCnhtQKUVw36UvqaRMevLa` | 14 GPU-h |
| `full-corrnet-phoenix-sd-001` | 14:05 | `ap-LpdMiHskC3BGNTMPJhaVmR` | 16 GPU-h |
| `full-tlp-phoenix-sd-001` | 15:28 | `ap-brLTkY5v3XGxA5aUMV3ioZ` | 14 GPU-h |
| `full-sen-phoenix-sd-001` | 15:28 | `ap-q7NTK645rUEFQURf7rlJGn` | 18 GPU-h |

**Blocks 2–5 (Phoenix SI, GrSL SI, GrSL Unseen-Sent, ArabSign SD): 20 unattended runs, launched 2026-11-03 08:36–08:41 UTC.**

Each run is a single detached Modal call of `run_full`, launched as `modal run --detach … ::launch_run_full`, one call per app. The call:

1. trains with the unchanged protocol (lr 1e-3, 40 epochs, steps 20/35);
2. evaluates the dev-best checkpoint on dev and test;
3. writes `runs/{eval_run_id}/evaluation.json`, sclite's raw reports and the hypothesis CTMs to the results Volume, with one record per attempt in `runs/{run_id}/run_full.jsonl`.

No step depends on this session, a local process or an open connection.

Each run's declared GPU-hour ceiling is enforced inside the function. A heartbeat records container time per Modal attempt in `gpu_seconds.json`, and upstream is stopped when the ceiling is reached. A stopped run is recorded as `stopped` and is not retried. Modal retries (up to 3) resume from the newest checkpoint that loads.

The 20 runs together are estimated at 154 GPU-h (about USD 465 at USD 3.03/h). The sum of their ceilings is 276 GPU-h, a worst case of USD 836. Every app and function-call ID is in `reproduction.json.runs`, with a dashboard link.

**Proof of detachment.** `full-vac-phoenix-si-001` was launched first, at 08:36:37. Its local client exited at 08:36:51. At 08:39:28, with no local `modal` process, its app (`ap-TExBCfqcV25PagMad0KbLc`) was `ephemeral (detached)` with one running task. Its call (`fc-01M6G8VAYT4NYXFWJNAZ9SSKDV`) was running, and it had written its own start record to the Volume. Only then were the other 19 launched.

**Harness checks.**

- `check-run-full-complete-002` ran `run_full` on the finished SMKD Phoenix SD run. It skipped training, re-evaluated epoch 36 and got 1,440 / 6,504 = 22.1402%, identical to `eval-smkd-phoenix-sd-001`.
- `check-run-full-ceiling-002` exercised the ceiling stop with a 0.05 GPU-h budget. Upstream was killed after 1.4 s, the attempt was recorded as `stopped` (`gpu_hour_ceiling_reached`), and the call returned with no retry.

The first Phoenix SI launch, at 08:18, used the train-only function. It was stopped during frame staging, before any checkpoint, and replaced by `run_full`. The stopped calls are listed under each run's `compute.superseded_calls`.

**GPU concurrency.** At 08:41:52, 5 runs had a GPU and 15 were waiting for one. The containers started between 08:36:51 and 09:10:35 UTC, so the longest wait for a GPU was about 30 minutes. All 20 then ran at once until the first finished at 12:32:15.

**Outcome (collected 2026-11-04).** All 20 calls succeeded in one Modal attempt each, with no retry and no ceiling reached. Each trained 40 epochs, evaluated its dev-best checkpoint on dev and test, and wrote its outputs to the Volume. The last finished at 19:13:58 UTC on 2026-11-03.

| Block | Container GPU-h | Estimate (GPU-h) | Ceiling (GPU-h) | Billed USD |
| --- | ---: | ---: | ---: | ---: |
| 2 Phoenix SI | 23.67 | 29.81 | 55 | 72.06 |
| 3 GrSL SI | 38.05 | 38.75 | 69 | 115.62 |
| 4 GrSL Unseen-Sent | 34.46 | 41.58 | 74 | 104.82 |
| 5 ArabSign SD | 33.30 | 43.66 | 78 | 101.25 |
| **Total** | **129.48** | **153.8** | **276** | **393.76** |

Per-run container time is in each run's `compute.gpu_hours`. The longest was SEN on GrSL SI, at 10.47 of its 16 GPU-h. Cumulative spend is **USD 509.75 of USD 2,000** (Modal billing summed over this paper's apps, 2026-11-02 to 2026-11-04, including the CPU-only checks of 2026-11-04).

**Block 6, ArabSign LOSO: 30 runs, declared 2026-11-04 before launch.** Gate `arabsign-loso-hold` was resolved by the operator. The protocol is the approved one: lr 1e-3, 40 epochs, steps 20/35, batch 2, seed 0, final-epoch checkpoint. Each run is one detached `run_full` call, as in blocks 2–5, with `--dev-rule final-epoch`. Upstream's per-epoch dev loop reads the fixed 50-clip token list, which is never used for selection; the run then scores epoch 39 on that list and on the held-out signer.

Per-run estimates take the measured ArabSign SD run of the same model (upstream training and evaluation time, container overhead) and rescale it by each fold's training frames, token dev list and test frames. Ceilings use the blocks 2–5 rule, ceil(1.6 × estimate + 1) GPU-h. USD is at 3.041/h, the blocks 2–5 billed rate.

| Model | Estimate per fold (GPU-h) | Ceiling per fold (GPU-h) |
| --- | ---: | ---: |
| VAC | 6.45–6.60 | 12 |
| SMKD | 6.47–6.62 | 12 |
| TLP | 5.05–5.17 | 10 |
| SEN | 9.05–9.26 | 16 |
| CorrNet | 6.93–7.09 | 13 |
| **30 runs** | **207.0 (USD 630)** | **378 (USD 1,150)** |

Cumulative spend before launch is USD 509.75. The projected total after block 6 is about **USD 1,139**. If every run hit its ceiling the total would be USD 1,659, under the USD 2,000 absolute ceiling. Every run's app and function-call ID is in `reproduction.json.runs`.

The declarations were pushed (`acd4c59`) before the first launch. The 30 runs were launched between 09:54:31 and 10:01:26 UTC, one `modal run --detach` app each. At 10:01:49 all 30 apps were `ephemeral (detached)`, 22 with a running task and 8 waiting for a GPU, and no local `modal` process was running. The first run had written its own start record to the Volume, confirming 40 epochs, steps 20/35 and the `final-epoch` rule.

**Block 6 outcome (collected 2026-11-05).** All 30 calls succeeded in one Modal attempt each, with no retry and no ceiling reached. The longest used 9.52 of its 16 GPU-h (`full-sen-arabsign-loso-03-001`). Containers started by 10:13:32 UTC, so no run waited more than about 12 minutes for a GPU. The first finished at 14:29:56 (TLP fold 01) and the last at 19:43:36 (SEN fold 06).

| Model | Container GPU-h, 6 folds | Range per fold | Estimate (GPU-h) | Ceiling (GPU-h) |
| --- | ---: | ---: | ---: | ---: |
| VAC | 43.18 | 7.10–7.42 | 39.34 | 72 |
| SMKD | 43.03 | 7.03–7.38 | 39.46 | 72 |
| TLP | 30.61 | 4.57–5.31 | 30.81 | 60 |
| SEN | 56.76 | 9.34–9.52 | 55.17 | 96 |
| CorrNet | 43.33 | 5.22–7.94 | 42.25 | 78 |
| **Block 6** | **216.91** | | **207.0** | **378** |

The block cost USD 659.15 in Modal billing, summed over its 30 apps. That is USD 3.04 per container GPU-h, against the USD 629.6 estimate.

**Total spend: USD 1,169.16 of the USD 2,000 ceiling** (Modal billing, 2026-11-02 to 2026-11-04, summed over this paper's apps; `compute-full-table.resolution.spend_ledger`).

| Item | USD |
| --- | ---: |
| Data jobs, image builds, preflights, diagnostics and CPU-only checks | 18.10 |
| Block 1, Phoenix SD (training and evaluation) | 98.15 |
| Blocks 2–5 | 393.76 |
| Block 6, ArabSign LOSO | 659.15 |
| **Total** | **1,169.16** |

USD 830.84 of the ceiling is unused. Billing showed no rows for 2026-11-05 when queried, so the 2026-11-05 status query and `inspect-environment-001` are not included; both are CPU-only and cost cents. Container time is 346.39 GPU-h over the 50 runs of blocks 2–6. Block 1 recorded training time per model (32.18 h in total) rather than container GPU-h.

## Guesses and deviations

| Detail | Paper/evidence says | This attempt uses | Rationale |
| --- | --- | --- | --- |
| Learning rate, TLP/SEN epochs | arXiv v1: lr 1e-3, 40 epochs (steps 20/35) for all. Upstream: lr 1e-4, TLP/SEN 80 epochs (steps 40/60). ICCVW 2025: lr 1e-3, TLP/SEN 80 epochs (steps 30/60). | arXiv v1 values for all five models, in every block including ArabSign LOSO | Operator decision at gate `protocol-lr-epochs`, confirmed on 2026-11-02 for TLP/SEN after the measured 80-epoch cost was reported (about USD +578 over all blocks). arXiv v1 is the assigned paper. ICCVW's 80 epochs (steps 30/60) is a known alternative that was not adopted. |
| TLP/SEN schedule evidence across versions | Table I vs ICCVW Table 1 (checked 2026-11-02 from both PDFs). All six TLP cells and five of six SEN cells are identical at one decimal. SEN GrSL Signer-Indep is 8.09 in v1, 8.0 in ICCVW Table 1 and 8.1 in ICCVW Fig. 3. Across all 30 shared cells, 26 are identical at one decimal, 3 match only by truncation (SMKD Phoenix SI 43.08→43.0, SEN GrSL SI 8.09→8.0, CorrNet ArabSign SI 7.65→7.6), and 1 differs (SMKD Phoenix SD 22.31→22.1). | 40 epochs (arXiv v1) | The numbers did not change when the stated schedule went from 40 to 80 epochs, so the evidence does not show which schedule produced them. Recorded as a known alternative, not adopted. |
| VAC / SMKD loss (guess) | The paper says each study's settings were followed and names no loss weights. VAC_CSLR `README.md` line 36 at `71f3e03` states "The VAC result is corresponding to the setting of `loss_weights: SeqCTC: 1.0, ConvCTC: 1.0`", and VAC+SMKD adds `share_classifier`/`weight_norm: True`. `baseline.yaml` comments out ConvCTC and a `Dist: 10.0`. | SeqCTC 1 + ConvCTC 1; SMKD adds share_classifier and weight_norm | A guess from the README's own mapping of config to result. Line 33 also labels the VAC row "ResNet18+VAC (CTC only)". Dist is not part of either. |
| Seed | Not stated | 0 (upstream default) | Single reported number per cell |
| Checkpoint selection | Not stated | Lowest dev WER, earliest epoch on ties. This is TLP/SEN/CorrNet's `_best_model.pt` rule. VAC_CSLR has no best-model rule, so per-epoch checkpoints (`--save-interval 1`) let the same rule apply. | Upstream behavior; uniform across models |
| GPUs | v1: "RTX A6000 with two 48GB GPUs". ICCVW: batch 2 "on a single GPU". | One L40S (48 GB) | Matches ICCVW and avoids VAC's documented multi-GPU BatchNorm issue |
| Phoenix SI5 lists | SI5 corpus files. Upstream drops pandas row 2390 of the multisigner train CSV, the broken clip `13April_2011_Wednesday_tagesschau_default-14`, which has 16 frames and is not resized. | SI5 lists drop that clip by ID | In the SI5 file row 2390 is a different, valid clip, so dropping by index would drop the wrong clip |
| ArabSign gloss labels | ArabSignGroundTruth.txt "Sign Language Sentence". The author's CSVs carry it as `caption`, plus a processed `procCaption` that strips the definite article from some words. The ArabSign paper (Luqman, FG 2023; arXiv 2210.03951v1, sha256 `7be6d45d…`) states in Sec. III-B "The dataset's sentences consist of 155 signs, and the dataset's vocabulary consists of 95 signs" and "The average sentence length is 3.1 signs" (Table II: "Vocab. size 95"). | The published "Sign Language Sentence" (= `caption`) | Neither version of the paper says which labels were used. The chosen labels give 50 sentences, 155 sign tokens and 3.1 signs per sentence, matching the ArabSign paper exactly, and 93 distinct glosses against its 95. `procCaption` gives 90, further away. The two-gloss gap is not explained by either column, so this supports the choice but does not prove it. Details: `reproduction.json.datasets[arabsign].vocabulary_check`. |
| ArabSign videos | 9,335 samples | 9,330 (5 absent from the RGB release; test 1,841 of 1,843) | Data as released; listed in `reproduction.json.datasets` |
| ArabSign Signer-Dep validation | 6,783 train / 709 validation (ICCVW only). The subset itself is unpublished. The dataset author's code draws validation with `train_test_split(random_state=42, stratify=sentence ID)`. | 709 of the 7,489 usable training rows, from `train_test_split(test_size=709, random_state=42, stratify=sentence ID)` with scikit-learn 1.8.0 on rows in sorted archive-path order. That gives 6,780 train / 709 dev / 1,841 test, all 50 sentences in each. Dev-best selection. | Operator decision at gate `arabsign-sd-validation`, 2026-11-02. Split hashes are in the gate's resolution. |
| ArabSign LOSO checkpoint | No dev set. arXiv v1 Sec. V, on CorrNet: "Better results are likely to be obtained by … adopting early stopping". | Train 40 epochs on all clips of the five training signers and score the final epoch (39). Upstream's per-epoch dev loop reads a fixed 50-clip token list (the first training clip of each sentence), which is never used for selection. | Operator decision at gate `arabsign-loso-checkpoint`, 2026-11-02, based on the early-stopping remark. Per-fold hashes are in the gate's resolution. |
| ArabSign Signer-Indep aggregation | Table I gives one number per model. Table II gives WER per "Tested Signer" 1–6 and an "Avg" column; the text says "average WER amongst all signers". | Plain arithmetic mean of the six per-signer test WERs. Tested signer *k* is the ArabSign signer folder `0k`. The word-weighted (pooled) mean is recorded as a note only. | For all five models, the mean of Table II's six per-signer values equals its Avg to rounding (within 0.0033). Operator decision of 2026-11-05. The pooled mean differs by at most 0.23 points. |
| Frame resizing for ArabSign/GrSL | "Resized to 256x256" | Upstream `resize_img` (`cv2.resize`, `INTER_LANCZOS4`) to 256×256 PNG, the Phoenix pipeline's own operation | Same operation the repos apply to Phoenix |
| Checkpoint retention | Not applicable | Full runs keep only the newest epoch checkpoint (for resume) and the current dev-best (for selection), and delete superseded ones during training | Declared before launch. About 385 MB × 40 epochs × 55 runs would otherwise cost about USD 75 per month in Volume storage. |

## Attempts, failures, and dead ends

The 2026-10-26 session stopped before data work at the compute gate. On 2026-11-02 the following happened, in order:

- **Image build: `ctcdecode`.** The first build failed in `ctcdecode`, which justified reusing the C++17 patch.
- **Image build: `simple-video-utils`.** The second build failed because `reproduction:latest` lacks `simple-video-utils`.
- **ArabSign frames, first two failures.** The first failed on PyAV 19 (`metadata_errors`), which led to the pin `av==18.1.0`. The second failed on 35 file stems whose prefixes differ from their folders, so sample IDs now come from the path.
- **First preflights:**
  - `preflight-smkd-gsl-001` lost its GPU mid-run (`CUDA error: unknown error`, and `nvidia-smi` stopped reporting the device), and was retried unchanged.
  - TLP and SEN failed on the missing `lmdb` module.
  - The CorrNet runs passed, but the helper `ctc_check.py` itself raised, because CorrNet's `seq_train` returns `None`.
- **CTC diagnostics.** These found the cuDNN gradient error, which led to the four `ctc-without-cudnn` patches.
- **ArabSign labels.** `preflight-*-arabsign-001` was stopped after `score_check` showed that the harness compared the wrong CSV columns with the ground-truth file. The CSV columns are `caption` and `procCaption`, not an Arabic sentence and a sign sentence.
- **SEN log parsing.** SEN's preflights then failed twice in the harness's log parser, because SEN logs WER without a percent sign.
- **Unattended launch, 2026-11-03.** The first two `run_full` checks were launched without `--detach`. `launch_run_full` only spawns and returns, so their apps stopped 3 s later and the calls never ran. They were relaunched with `--detach` as `-002`. The five train-only Phoenix SI calls from 08:18 were stopped during staging and replaced by `run_full`. Full-run containers have 40 GiB RAM from block 2 on, because block 1 peaked at 26.8 of 32 GiB.
- **Block 1 evaluations.** `eval-tlp-phoenix-sd-001` hit a GPU hardware fault (XID 154). `eval-corrnet-phoenix-sd-001` lost its client connection. Both were reruns of the same command, and `evaluate()` now raises on a non-zero upstream exit instead of failing in the log parser.
- **Investigation checks, 2026-11-04.** The adapter and checkpoint checks first ran at 08:48–08:58 UTC (apps `ap-bHQ3B5ERWHTpK2odgSqckY` and `ap-yqdhpwvsiOWMsfi0pFWHdc`) before their ceilings were written into `reproduction.json`. They were re-run unchanged as `adapter-check-002` and `checkpoint-check-002` with the ceilings declared first. Their outputs match the first invocations' (the adapter files differ only in timing fields); the first invocations are described under each run's `prior_invocation`.
- **Block 6, 2026-11-04.** No failure. All 30 LOSO runs finished in their first attempt. CorrNet folds 02 and 05 completed but produced two-gloss outputs, recorded as results and not retried.
- **Environment freeze, 2026-11-05.** No pip freeze had been recorded during the full runs. `inspect-environment-001`, a CPU-only call of the existing `inspect_environment` function, recorded it afterwards. Its stop policy was declared before launch, and it confirmed the same image ID as the full runs.

Every model × dataset preflight passed within its declared attempt limit. Each of the 55 full runs trained in one Modal attempt.

## Candidate flags, ethics, and human evaluation

The queue record flags no human evaluation and no ethical concerns, and `copied_scores` is `no`. The text confirms the paper re-ran every model and copied no baselines. All three datasets are existing recordings released for research, and no participants are involved.

## Author and team contact

None. The ArabSign access shown in the queue record was coordinated earlier by Team S.

## Gates

### Compute gate (resolved)

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

Each block declares its own ceilings, and cumulative spend is tracked against USD 2,000. All six blocks are complete, at a total of USD 1,169.16 (Execution evidence).

### Protocol decisions of 2026-11-02 (resolved)

- **TLP/SEN epochs** (`protocol-lr-epochs` follow-up). The measured cost of 80 epochs for TLP and SEN was reported: about USD +578 over all blocks, or +238 outside LOSO. TLP and SEN stay at 40 epochs with steps 20/35 in every block, as arXiv v1 states. ICCVW's 80 epochs (steps 30/60) is recorded as a known alternative that was not adopted.
- **`arabsign-sd-validation`: `stratified-709-seed42`.** 709 dev clips, sentence-stratified, seed 42, as in the guesses table. Split record: app `ap-4HwGUsgGtScHCYmacWBOAP`.

  | Split | Clips | fileid-list sha256 |
  | --- | ---: | --- |
  | train | 6,780 | `bd960085…` |
  | dev | 709 | `6407bcd9…` |
  | test | 1,841 | `d19c5fdc…` |

  The lists hash is `1d36ba96…`.
- **`arabsign-loso-checkpoint`: `final-epoch`.** No dev-based selection, based on v1's early-stopping remark. The per-fold train/test/token-dev hashes are in the gate's resolution.

With these decisions the projected total was about USD 1,396 before retries, including the USD 16 already spent (`compute-full-table.resolution.projection_after_decisions`).

### ArabSign LOSO hold (resolved)

`arabsign-loso-hold`. On 2026-11-04, after blocks 2–5, the operator held block 6 (ArabSign LOSO: 30 runs, 5 targets) and asked for the no-training investigation in Results first. After reviewing it, the operator resolved the gate the same day: launch the 30 runs under the approved protocol, with lr 1e-3, 40 epochs, steps 20/35 and the final-epoch checkpoint. No run uses a different learning rate or schedule. The operator gave four reasons:

- the adapters are identical across the repos;
- the observed failures are optimization instabilities under the stated protocol;
- ArabSign SD trained for all five models;
- the table must stay internally consistent.

Two facts were recorded with the decision: training precision per repository (`sources[].training_precision`) and GrSL Unseen-Sent gloss coverage (`datasets[grsl].gloss_coverage`).
