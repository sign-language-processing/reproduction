# XmDA (Ye et al., 2023) reproduction

**Paper ID:** `b32c4a0a5e401cb565d12d5b30485b21aeb38570`

**Citation:** Jinhui Ye, Wenxiang Jiao, Xing Wang, Zhaopeng Tu, Hui Xiong. 2023. Cross-modality Data Augmentation for End-to-End Sign Language Translation. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 13558–13571, Singapore. doi:10.18653/v1/2023.findings-emnlp.904

**Paper:** https://aclanthology.org/2023.findings-emnlp.904/ · **Code/artifacts:** https://github.com/Atrewin/SignXmDA (pinned), https://github.com/Atrewin/PGen (pinned), SLTUNet SMKD checkpoints from https://data.statmt.org/bzhang/iclr2023_sltunet/

**Preference level:** 2

**Pipeline status:** `partial` — the 12 baseline targets are produced; the 36 mix-up/KD/XmDA targets are not produced (`protocol_ambiguous`). Their conditional runs are in progress on both datasets (gate `compute-full-plan` resolved 2026-10-30)

**Numerical agreement:** `does_not_agree` — all 12 produced values are above the published ones: PHOENIX-2014T by 1.74 to 3.87 points, CSL-Daily by 5.04 to 9.93 points (one seed each; see Results)

**Attempt dates:** 2026-10-28 to 2026-10-29

## Where this attempt stands

- **Done.** Both Sign Language Transformers baselines (Table 1 and Table 2, "SMKD features" rows) were trained for the full 150 epochs on T4 at seed 56 and evaluated with the published code. These runs produce the 12 comparable targets.
- **CSL-Daily gap.** CSL-Daily text is character-level, as in the authors' own pickle builder (Atrewin/slt `video2tensor/concat_data.py`), so tokenization is not the main candidate explanation. There are two candidate explanations:
  - **Hyperparameters (leading).** The paper's CSL-Daily baseline may not have been trained with Table 10. One CSL-Daily baseline with the released config as is was run as conditional evidence (gate `hyperparameters`). It is closer to the paper but still above it (dev BLEU-4 14.30 against 11.55).
  - **Sign features (second candidate).** The authors' development fork built its CSL-Daily pickles from VAC_CSLR `checkpoints/cslt/gloss30check2` (Atrewin/slt `4b47598`, `video2tensor/concat_data.py` L168–170), a VAC_CSLR CSLR model. This attempt uses SLTUNet's released SMKD checkpoint. This attempt did not locate those weights, so this was not tested. The Table 10 run stays the produced target. Greedy (beam 1) scores and the full dev beam/alpha grid are recorded next to the grid-selected values for both baselines.
- **Gate `teacher-gloss-decoding` (resolved 2026-10-29 by the assignee as `patch-mirror-training`).** The conditional teachers are signjoey `forward_type=gloss` models, which the pinned code cannot validate or decode. `patches/0001-gloss-decoding-mirror-training.patch` builds the validation/test gloss input exactly as `Batch_Sign` does in training. This is an **invented behavior choice**: the authors' fork (Atrewin/slt `9ecb427`, "all extend gloss") added the repetition to the training batches only, so no author evidence exists for the decoding side. `patches/0002-test-pass-forward-type.patch` is a correctness patch that the assignee kept: `test()` passes `forward_type` to both dev passes but omits it on the final test pass. Both patches touch only the gloss path.
- **Gate `compute-full-plan` (resolved 2026-10-30 by the assignee).** The assignee resolved it on the project lead's standing instruction of 2026-10-29 in `#repro-sign-team-r` to use more Modal compute. No discussion specific to this plan took place in the channel. Approved scope:
  - **PHOENIX-2014T:** the four teachers, Mix-up and KD are done (Results). XmDA attempt 1 was cancelled after 5 h by a client-token expiry, and attempt 2, a from-scratch restart, runs with `--spawn`. Two extra student seeds (57, 58) run for all four rows; seed 56 stays the reported value. The replicates vary only the student seed. The four teachers (seeds 56–59), the frozen gloss embedding from teacher 56 and the multi-reference training pickle are identical across student seeds. The replicate policy was pre-declared at 2026-10-30T10:18:26Z, before any replicate launched: replicates are reported as mean and sample standard deviation over seeds 56, 57 and 58, attached as evidence and never swapped in.
  - **CSL-Daily:** the four teachers, then the Mix-up, KD and XmDA students.
  - **Environment trial (gate `environment-cu110`, resolved 2026-10-30: adopt).** `Dockerfile.cu110` swaps in torch 1.7.1+cu110 / torchtext 0.8.1, with no SignXmDA code changes (image `im-zcYV3stvhV0g565br8nSMZ`, freeze `17115c8c…`). The comparability criterion was first written at 10:27:43Z and amended at 10:59:47Z, both before the validation run finished; the original is kept in the gate as superseded. The reason for the amendment: with three seeds, a fourth draw from the same distribution falls outside their range with probability 1/2 per metric. Amended rule: the new-image seed-56 PHOENIX-2014T baseline must be within max(1.0 BLEU, the three-seed range) of the old-image mean on dev and on test BLEU-4. Result: old-image seeds 56/57/58 give dev 24.78 / 24.33 / 23.98 (mean 24.36) and test 24.53 / 23.36 / 23.04 (mean 23.64). The new image gives dev 24.68 (difference 0.32, allowed 1.00) and test 23.90 (difference 0.26, allowed 1.49), so it passes. It would also have passed the superseded rule. So the CSL-Daily student rows (Mix-up, KD, XmDA, plus their own in-image reference baseline) run in the new image on A10G. The produced CSL-Daily baseline stays the old-image T4 run, the CSL-Daily teachers stay in the old image, and PHOENIX-2014T stays in the old image. Because the student rows share an environment with their own reference baseline, the environment cannot confound within-table deltas; this check only guarded against a broken environment.

  The technical limits stand: Modal's 24 h function cap, and no real resume in the code (see How to repeat this). Validation frequency stays at every 100 steps.
- **Hardware.** The pinned stack (torch 1.4.0, CUDA 10.1, cuDNN 7.6.3) ships kernels only up to sm_75 (cuobjdump on the built image: SASS for sm_35/37/50/60/61/70/75, PTX only for sm_37). It cannot run on Ampere (A10G/A100) without a torch/CUDA upgrade, which would also move torchtext off the 0.5 API SignXmDA uses. Runs therefore stay on T4, and no A10G re-measurement was made.

## Reproduction agents

| Agent ID | Model and version | Agent application | Contribution | Attribution evidence / unknowns |
| --- | --- | --- | --- | --- |
| `claude-opus-5-5-claude-code` | Claude Opus 5.5 (`claude-opus-5-5`) | Claude Code 2.1.312 | Assignment ingestion, target resolution, source search and pinning, data verification, all glue code and both upstream patches, execution of every recorded run (2026-10-28 and 2026-10-29), gate preparation, and this report | Session system metadata exposes the model and model ID; `claude --version` in the session reports 2.1.312. Every run in `reproduction.json.runs` lists this agent in `agent_ids`. |

## Scope and target contract

The assignment was a REPRO-SIGN tracker export (PocketBase `reproductions` record, SHA-256 `c6bf0641…28c0cd`) with the paper PDF. The export schema has no `confirmation` field, so it is recorded as a direct user request, as in `papers/pu-2024-siformer`. It asks for Tables 1 and 2.

- **In scope (48 targets).** The four "Our Experiments" rows of Table 1 (PHOENIX-2014T) and Table 2 (CSL-Daily): Sign Language Transformers on SMKD features, + Cross-modality Mix-up, + Cross-modality KD, and + XmDA. Each row has dev and test BLEU-4, ROUGE-L, and ChrF.
- **Copied baselines (22 targets, `in_scope: false`).** The "Previous Research" rows: SLT (Camgoz et al. 2020b on PHOENIX; Zhou et al. 2021's re-run on CSL-Daily), SignBT (Zhou et al. 2021), and ConSLT (Fu et al. 2022). Each carries `not_produced / copied_baseline` and was not rerun.
- **Metrics.** These are the SignXmDA `signjoey/metrics.py` implementations with the vendored sacrebleu 1.4.2:
  - BLEU-4 is corpus `raw_corpus_bleu` on whitespace tokens.
  - ROUGE-L is the mean sentence-level F1 from `mscoco_rouge`.
  - ChrF is corpus chrF ×100.
  - CSL-Daily text is space-separated characters, so CSL-Daily metrics are character-level.
- **Checkpoint and decoding.** Greedy dev validation runs every 100 steps for 150 epochs; upstream disables early stopping. The best dev-BLEU checkpoint is used. `prediction_XmDA.test` selects beam ∈ {1,2,3,4} × alpha ∈ {−2,−1,0,1} on dev and applies the choice to test. This is the published code, unchanged.
- **Seeds.** One seed per cell, seed 56 (the released config's seed). The paper gives no seed policy.
- **Protocol resolution.** Gate `xmda-protocol` was resolved by the assignee as `baseline-plus-conditional`:
  - The SLT baseline rows are the produced pipeline.
  - Mix-up, KD, and XmDA run only as conditional evidence, using invented teacher substitutions (see the deviations). Their 36 targets stay `not_produced / protocol_ambiguous`, with conditional values attached once run. The teachers are validated and decoded with the two patches described under Environment and patches.
- **Hyperparameters.** Gate `hyperparameters` was resolved as Table 10 for both datasets. The released config is kept as a documented alternative.

## Source provenance

| Artifact | Canonical source | Pinned revision / SHA-256 | Role |
| --- | --- | --- | --- |
| Paper PDF (canonical) | https://aclanthology.org/2023.findings-emnlp.904.pdf | `03c82561f4044d50f157a69e1d71c84c8fc62d668683e58a774229e3af62881a` | Targets and protocol |
| Paper PDF (queue copy) | https://arxiv.org/abs/2305.11096v4 | `6a33179904477bb113de85877f8acba8176b1a6a472bd27d318a0d7ab4173bc0` | Tables 1, 2, 10, 14, 15 numerically identical to the Anthology version |
| SignXmDA | https://github.com/Atrewin/SignXmDA | `b6e2ca4ea81cab41cf858e1d0ce948a778cb09a5` | Official code; the reproduction recipe |
| neccam/slt | https://github.com/neccam/slt | `90588825f6229474bc19ac7a6b30ea3116635ba3` | Upstream signjoey; `requirements.txt` used for the environment |
| PGen | https://github.com/Atrewin/PGen | `2b8a2329d196aead731e4dc7a1449cd57247708c` | Released gloss-text pseudo-pairs (conditional teachers only); its fairseq teacher trainer is unreleased (issue #1) |
| SLTUNet | https://github.com/bzhangGo/sltunet | `b1d1d0e8b3b7275e10cd89229f2632b556df5de9` | SMKD feature extractor code (Zhang et al. 2023a, whose SMKD setup XmDA says it follows) |
| SLTUNet PHOENIX release | https://data.statmt.org/bzhang/iclr2023_sltunet/phoenix.tar.gz | `b0e708b7abe5689475905ad11ac578abb02eb0f9bf00b207bbd6d4342afe5152` | SMKD checkpoint `signemb_ckpt/average.pt`; verified by `fetch_release` |
| SLTUNet CSL-Daily release | https://data.statmt.org/bzhang/iclr2023_sltunet/csldaily2.tar.gz | `55e3da80a0e754e8b8797c11f58f83da5779b276703979c29eb71530526f7c1a` | CSL-Daily SMKD checkpoint; verified by `fetch_release` |
| Atrewin/slt | https://github.com/Atrewin/slt | `4b4759878c34df2efb54f20642d51d85a3caeac9` | Evidence for ff_size 2048; commit `9ecb427b159de5aaaa6bd21a5957f74d752564c1` ("all extend gloss") shows gloss repetition added to training batches only; not executed |
| VAC_CSLR | https://github.com/ycmin95/VAC_CSLR | not pinned | Considered; not used (SLTUNet's adapted SMKD code and checkpoints are released) |

Searches covered the paper, its footnotes, the SignXmDA, PGen, and SLTUNet repositories, and the SLTUNet release server. The SignXmDA repository has no releases, tags, forks, PRs, or issues. No search found XmDA's own SMKD features, its four PGen teachers or their outputs, its frozen gloss embedding, or per-row configs.

## Results

Twelve in-scope targets are produced. Each comes from a single run at seed 56: the best dev-BLEU checkpoint is evaluated, with beam and alpha selected on dev by `prediction_XmDA.test`. Values come from the `[DEV]`/`[TEST] Recognition & Translation` block of each run's `train.log`. The difference is reproduced minus original.

| Target ID | Paper location | Dataset/split | Metric | Original | Reproduced | Difference | Run / raw artifact |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `table1-baseline-dev-bleu4` | Table 1, SLT (SMKD features) | PHOENIX-2014T dev | BLEU-4 | 22.90 | 24.78 | +1.88 | `phoenix14t-baseline-seed56` / `…-dev-results` |
| `table1-baseline-dev-rouge_l` | Table 1, SLT (SMKD features) | PHOENIX-2014T dev | ROUGE-L | 48.05 | 51.15 | +3.10 | as above |
| `table1-baseline-dev-chrf` | Table 1, SLT (SMKD features) | PHOENIX-2014T dev | ChrF | 44.96 | 47.43 | +2.47 | as above |
| `table1-baseline-test-bleu4` | Table 1, SLT (SMKD features) | PHOENIX-2014T test | BLEU-4 | 22.79 | 24.53 | +1.74 | `phoenix14t-baseline-seed56` / `…-test-results` |
| `table1-baseline-test-rouge_l` | Table 1, SLT (SMKD features) | PHOENIX-2014T test | ROUGE-L | 47.33 | 50.88 | +3.55 | as above |
| `table1-baseline-test-chrf` | Table 1, SLT (SMKD features) | PHOENIX-2014T test | ChrF | 44.36 | 48.23 | +3.87 | as above |
| `table2-baseline-dev-bleu4` | Table 2, SLT (SMKD features) | CSL-Daily dev | BLEU-4 | 11.55 | 17.40 | +5.85 | `csl-daily-baseline-seed56` / `…-dev-results` |
| `table2-baseline-dev-rouge_l` | Table 2, SLT (SMKD features) | CSL-Daily dev | ROUGE-L | 36.32 | 46.25 | +9.93 | as above |
| `table2-baseline-dev-chrf` | Table 2, SLT (SMKD features) | CSL-Daily dev | ChrF | 11.14 | 16.18 | +5.04 | as above |
| `table2-baseline-test-bleu4` | Table 2, SLT (SMKD features) | CSL-Daily test | BLEU-4 | 11.61 | 17.45 | +5.84 | `csl-daily-baseline-seed56` / `…-test-results` |
| `table2-baseline-test-rouge_l` | Table 2, SLT (SMKD features) | CSL-Daily test | ROUGE-L | 36.43 | 46.18 | +9.75 | as above |
| `table2-baseline-test-chrf` | Table 2, SLT (SMKD features) | CSL-Daily test | ChrF | 11.15 | 16.28 | +5.13 | as above |

Run details:

| Run | Best step (greedy dev BLEU-4) | Selected beam / alpha | Dev / test WER (not a target) |
| --- | --- | --- | --- |
| `phoenix14t-baseline-seed56` | 8,100 (23.91) | 4 / 1 | 23.06 / 23.81 |
| `csl-daily-baseline-seed56` | 13,100 (16.99) | 2 / 1 | 32.07 / 32.39 |

**Assessment basis.** The paper gives one number per cell, with no seed count or variance, and this attempt ran one seed. So nothing supports treating the gaps as run-to-run noise, and all twelve are recorded as numerical disagreement for the reviewer to judge. CSL-Daily is scored on character tokens, as in the authors' builder, which writes `'text': char` and saves `cslt.word.{split}.pickle.char` (Atrewin/slt `4b47598`, `video2tensor/concat_data.py` L181 and L189). The SMKD features remain a guess (see Guesses and deviations).

**Decoding setting.** The paper does not say how it decoded. The dev beam/alpha grid was therefore re-decoded setting by setting from each best checkpoint (`--step decode_grid`), and test was also decoded greedily. Alpha −2, −1 and 0 give identical outputs at each beam size. Each setting's test pass is in the grid logs, but test scores at non-selected settings other than greedy are deliberately left out of the report: choosing among them would mean selecting on test.

| Dataset | Setting | Dev BLEU-4 / ROUGE-L / ChrF | Test BLEU-4 / ROUGE-L / ChrF |
| --- | --- | --- | --- |
| PHOENIX-2014T | greedy (beam 1) | 23.91 / 50.45 / 47.49 | 23.58 / 49.73 / 47.58 |
| PHOENIX-2014T | beam 2, alpha 1 | 24.66 / 51.07 / 47.79 | — |
| PHOENIX-2014T | beam 3, alpha 1 | 24.63 / 50.86 / 47.49 | — |
| PHOENIX-2014T | **beam 4, alpha 1 (selected)** | 24.78 / 51.15 / 47.43 | 24.53 / 50.88 / 48.23 |
| PHOENIX-2014T | beam 2 / 3 / 4, alpha ≤ 0 | BLEU-4 24.60 / 24.38 / 24.45 | — |
| CSL-Daily | greedy (beam 1) | 16.99 / 45.51 / 15.87 | 17.07 / 45.40 / 15.95 |
| CSL-Daily | **beam 2, alpha 1 (selected)** | 17.40 / 46.25 / 16.18 | 17.45 / 46.18 / 16.28 |
| CSL-Daily | beam 3 / 4, alpha 1 | BLEU-4 17.09 / 16.97 | — |
| CSL-Daily | beam 2 / 3 / 4, alpha ≤ 0 | BLEU-4 17.27 / 16.73 / 16.53 | — |

The grids ran on the patched image and reproduce the training runs' own selected dev and test values exactly. The greedy dev values also equal each run's validation at its best step.

**Released-config CSL-Daily baseline (conditional evidence, gate `hyperparameters`; not a target value).** Run `csl-daily-baseline-released-config-seed56` used `configs/Sign_XmDA.yaml` as is (hidden 256, ff 1024, no label smoothing, beta2 0.998, max output length 60) at seed 56. Only run paths and the baseline's data roles were set. Its best step was 15,600 (greedy dev BLEU-4 13.87), with beam 4 / alpha 1 selected.

| Split | Paper | Table 10 run (produced) | Released config (conditional) | Released minus paper |
| --- | --- | --- | --- | --- |
| dev | 11.55 / 36.32 / 11.14 | 17.40 / 46.25 / 16.18 | 14.30 / 42.38 / 13.91 | +2.75 / +6.06 / +2.77 |
| test | 11.61 / 36.43 / 11.15 | 17.45 / 46.18 / 16.28 | 14.80 / 42.40 / 14.21 | +3.19 / +5.97 / +3.06 |

The fork's experiment name `081_cslt_char_sign_ratio0.6_baseline_S2T_seed56_bsz64_drop15_len30_freq100` also implies batch 64, dropout 0.15 and max length 30, which the released config does not set. That variant was not run.

**CSL-Daily in-image reference baseline (not a target value).** `csl-daily-baseline-cu110-seed56`: Table 10, seed 56, run in the cu110 image on A10G, the same environment as the CSL-Daily student rows. Best step 20,300, with beam 2 / alpha 1 selected on dev. BLEU-4 / ROUGE-L / ChrF:

| Run | Dev | Test |
| --- | --- | --- |
| In-image reference (cu110, A10G) | 17.74 / 47.16 / 16.59 | 18.11 / 47.05 / 16.83 |
| Produced target (`csl-daily-baseline-seed56`, old image, T4) | 17.40 / 46.25 / 16.18 | 17.45 / 46.18 / 16.28 |

The CSL-Daily student rows' within-table deltas are taken against this reference. The produced target stays the old-image T4 run. One seed each, so the 0.34 / 0.66 BLEU-4 difference between the two runs is not attributed to the environment.

**PHOENIX-2014T teachers (conditional; not targets).** These are signjoey `forward_type=gloss` teachers with PGen pseudo-pairs and patches 0001 and 0002, one per seed. Table 15's teachers are PGen fairseq models, so they are a different system; their numbers are shown only for orientation, paired with the runs by seed order only (the paper does not say which of its teachers it used). Seed 56 has the highest dev BLEU-4 and supplies the frozen gloss embedding.

| Run | Best step | Dev beam / alpha | Dev BLEU-4 / ROUGE-L / ChrF | Table 15 teacher dev BLEU-4 | Training-set BLEU-4 (KD references) |
| --- | --- | --- | --- | --- | --- |
| `phoenix14t-teacher-seed56` | 8,000 | 3 / 1 | 26.66 / 52.65 / 49.94 | 27.35 | 97.42 |
| `phoenix14t-teacher-seed57` | 6,700 | 2 / 1 | 26.62 / 52.37 / 49.89 | 27.24 | 91.55 |
| `phoenix14t-teacher-seed58` | 4,500 | 3 / 1 | 26.60 / 52.69 / 50.05 | 27.11 | 80.88 |
| `phoenix14t-teacher-seed59` | 9,100 | 2 / 1 | 26.63 / 52.52 / 50.41 | 27.02 | 97.72 |

**PHOENIX-2014T conditional students (conditional evidence; the targets stay `not_produced / protocol_ambiguous`).** These are single runs at seed 56 with Table 10 settings and the substitute teachers above. Selection is the same as for the baseline: best dev-BLEU checkpoint, then beam and alpha chosen on dev. Values are BLEU-4 / ROUGE-L / ChrF.

| Row | Run | Dev | Test | Paper dev | Paper test | Dev BLEU-4 gain over baseline (ours / paper) |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline (produced) | `phoenix14t-baseline-seed56` | 24.78 / 51.15 / 47.43 | 24.53 / 50.88 / 48.23 | 22.90 / 48.05 / 44.96 | 22.79 / 47.33 / 44.36 | — |
| + Cross-modality Mix-up | `phoenix14t-mixup-seed56` (step 4,000; beam 1 selected) | 25.36 / 50.91 / 48.50 | 24.08 / 49.87 / 47.97 | 23.60 / 48.59 / 46.14 | 23.87 / 48.92 / 46.04 | +0.58 / +0.70 |
| + Cross-modality KD | `phoenix14t-kd-seed56` (step 14,400; beam 3, alpha 1) | 25.45 / 52.13 / 48.21 | 25.59 / 51.85 / 49.25 | 24.82 / 50.04 / 47.98 | 24.77 / 49.53 / 48.04 | +0.67 / +1.92 |
| + XmDA | attempt 1 interrupted; attempt 2 running (see below) | — | — | 25.86 / 52.42 / 50.10 | 25.36 / 49.87 / 51.49 | — |

On test, the gain over our baseline is −0.45 BLEU-4 for Mix-up (paper +1.08) and +1.06 for KD (paper +1.98). Three notes for the reviewer:

- **(a) Smaller KD gain: the teacher–student margin is the leading explanation.** The substitute teachers match Table 15 to within 0.7 dev BLEU-4: 26.60–26.66 against 27.02–27.35. Our baseline, however, is 1.9 BLEU-4 stronger than the paper's (24.78 against 22.90 dev). So the margin a teacher can transfer to the student is about 1.9 dev BLEU-4 here, against 4.1–4.5 in the paper. That smaller margin is the leading explanation for the smaller KD gain (+0.67 against +1.92 dev).
- **(b) Mix-up's −0.45 on test fits what the published code does.** In `training_XmDA.py` at `b6e2ca4`, the mix mask `gloss_predict.ge(1 - mixup_ratio)` is applied to target gloss indices from `ctc_alignment`, which never emits the blank. The mask is therefore true at every frame, and no gloss embedding enters the encoder. The "mixed" input is the sign embedding from a second `sgn_embed` pass with a fresh dropout mask, trained with cross-entropy plus a ×15 JSD consistency term. The branch is gated by `recognition_loss < 5` or `random.randint(0, 10) < 5`. So as released, the Mix-up row is a gated dropout-consistency regularizer, not cross-modality mix-up; the details are in `code_observations`.
- **(c) KD's early stop is upstream scheduler behaviour.** KD stopped in epoch 104, at step 28,000, because the plateau scheduler took the learning rate below `learning_rate_min` (1e-7). `_add_report` then sets `stop`. The same rule is applied to every row; the baseline, the teachers and Mix-up simply never reached the floor within 150 epochs.

**PHOENIX-2014T replicates (pre-declared; evidence only, never swapped in).** Each row adds student seeds 57 and 58 to seed 56, in the same image with the same config. Only the student seed varies: the four teachers (seeds 56–59), the frozen gloss embedding from teacher 56 and the multi-reference pickle are the same for every seed. Values are BLEU-4, given as the three seeds, then mean ± sample std (n = 3). Rows fill in as runs finish.

| Row | Dev BLEU-4 (56 / 57 / 58) | Dev mean ± std | Test BLEU-4 (56 / 57 / 58) | Test mean ± std |
| --- | --- | --- | --- | --- |
| Baseline | 24.78 / 24.33 / 23.98 | 24.36 ± 0.40 | 24.53 / 23.36 / 23.04 | 23.64 ± 0.78 |
| + Mix-up | 25.36 / 24.94 / 25.16 | 25.15 ± 0.21 | 24.08 / 24.95 / 24.79 | 24.61 ± 0.46 |
| + KD | 25.45 / pending / pending | — | 25.59 / pending / pending | — |
| + XmDA | pending (attempt 2) / pending / pending | — | — | — |

Across seeds, the seed-56 Mix-up result of −0.45 on test (note b) is within seed variation. The three-seed means put Mix-up at +0.79 dev and +0.97 test BLEU-4 above the baseline, against +0.70 and +1.08 in the paper. ROUGE-L and ChrF for every seed are in each run entry and in the targets' evidence.

**Not produced (36 targets).** These are the mix-up, KD, and XmDA rows: `not_produced / protocol_ambiguous` (gates `xmda-protocol`, `teacher-gloss-decoding`, and `compute-full-plan`). The PHOENIX-2014T Mix-up and KD rows carry the conditional values above. The XmDA rows wait for its attempt 2, and the CSL-Daily rows wait for gate `compute-full-plan`.

| Target IDs | System | Dataset | Original dev (BLEU-4 / ROUGE-L / ChrF) | Original test |
| --- | --- | --- | --- | --- |
| `table1-mixup-*` | + Cross-modality Mix-up | PHOENIX-2014T | 23.60 / 48.59 / 46.14 | 23.87 / 48.92 / 46.04 |
| `table1-kd-*` | + Cross-modality KD | PHOENIX-2014T | 24.82 / 50.04 / 47.98 | 24.77 / 49.53 / 48.04 |
| `table1-xmda-*` | + XmDA | PHOENIX-2014T | 25.86 / 52.42 / 50.10 | 25.36 / 49.87 / 51.49 |
| `table2-mixup-*` | + Cross-modality Mix-up | CSL-Daily | 15.67 / 42.36 / 14.57 | 15.47 / 42.60 / 14.68 |
| `table2-kd-*` | + Cross-modality KD | CSL-Daily | 17.05 / 44.17 / 15.74 | 16.92 / 43.90 / 15.65 |
| `table2-xmda-*` | + XmDA | CSL-Daily | 21.69 / 49.36 / 19.60 | 21.58 / 49.34 / 19.50 |

**Copied baselines (not rerun, `copied_baseline`).**

Table 1:

| Row (source) | Dev BLEU-4 | Test BLEU-4 | Dev ROUGE-L | Test ROUGE-L |
| --- | --- | --- | --- | --- |
| SLT (Camgoz et al. 2020b) | 22.38 | 21.32 | — | — |
| SignBT (Zhou et al. 2021) | 24.45 | 24.32 | 50.29 | 49.54 |
| ConSLT (Fu et al. 2022) | 21.11 | 21.59 | 47.74 | 47.69 |

Table 2:

| Row (source) | Dev BLEU-4 | Test BLEU-4 | Dev ROUGE-L | Test ROUGE-L |
| --- | --- | --- | --- | --- |
| SLT (Zhou et al. 2021) | 11.88 | 11.79 | 37.06 | 36.74 |
| SignBT (Zhou et al. 2021) | 20.80 | 21.34 | 49.49 | 49.31 |
| ConSLT (Fu et al. 2022) | 14.80 | 14.53 | 41.46 | 40.98 |

**Preflight-only values (not results).** These come from 2 of 150 epochs and only show that the path runs:

| Run | Dev BLEU-4 | Test BLEU-4 |
| --- | --- | --- |
| PHOENIX baseline | 6.30 | 5.17 |
| CSL-Daily baseline | 5.03 (32-sentence dev subset) | 3.56 |
| PHOENIX teacher, attempt 3 (gloss input, patches 0001+0002) | 19.83 (32-sentence dev subset) | 11.56 on the training set (KD references; 5,537 distinct outputs of 7,096) |

**SMKD sanity control (not a target).** This is the greedy CTC gloss WER from the feature extractor, compared with SLTUNet Table 12. The CSL-Daily gap may reflect the mp4 inputs (see the deviations).

| Dataset | Dev WER (SLTUNet) | Test WER | Train WER |
| --- | --- | --- | --- |
| PHOENIX | 19.37 (19.64) | 20.76 | 2.79 |
| CSL-Daily | 30.37 (29.32) | 29.38 | 8.35 |

## How to repeat this

All commands run from `papers/ye-2023-xmda/`. Every Modal call goes through the workspace wrapper. Outputs go to the paper's private Volume `ye-2023-xmda-results`, mounted at `/outputs`. `datasets` is mounted read-only at `/datasets`, and `huggingface-cache` at `/cache/huggingface`.

```bash
M=../../.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh

# 1. Verify data (counts, file hashes, per-video manifests)
$M run modal_app.py --step data_manifest --split phoenix14t
$M run modal_app.py --step data_manifest --split csl-daily

# 2. Fetch and hash-verify the SLTUNet releases that hold the SMKD checkpoints
$M run modal_app.py --step fetch_release --split phoenix
$M run modal_app.py --step fetch_release --split csldaily2

# 3. SMKD features (L4) and SignXmDA pickles, per dataset
for ds in phoenix14t csl-daily; do
  for split in train dev test; do
    $M run modal_app.py --step smkd_features --dataset $ds --split $split
  done
  $M run modal_app.py --step pickles --dataset $ds
done

# 4. Preflight (2 epochs, tiny dev, 2-setting decode grid)
$M run modal_app.py --step train --dataset phoenix14t --kind baseline --run-id preflight-phoenix14t-baseline-attempt-2 --preflight
$M run modal_app.py --step train --dataset csl-daily --kind baseline --run-id preflight-csl-daily-baseline --preflight
$M run modal_app.py --step train --dataset phoenix14t --kind teacher --run-id preflight-phoenix14t-teacher-attempt-3 --preflight

# 5. Full baselines (run 2026-10-29; detached, so they survive the local client)
$M run --detach modal_app.py --step train --dataset phoenix14t --kind baseline --run-id phoenix14t-baseline-seed56
$M run --detach modal_app.py --step train --dataset csl-daily --kind baseline --run-id csl-daily-baseline-seed56

# 6. Dev beam/alpha grid, one setting per prediction_XmDA.test call, from each best checkpoint
$M run modal_app.py --step decode_grid --run-id phoenix14t-baseline-seed56
$M run modal_app.py --step decode_grid --run-id csl-daily-baseline-seed56

# 7. Released-config CSL-Daily baseline (conditional evidence, gate hyperparameters)
$M run --detach modal_app.py --step train --dataset csl-daily --kind baseline --hparams released --run-id csl-daily-baseline-released-config-seed56

# 8. PHOENIX-2014T conditional chain (approved 2026-10-29)
for seed in 56 57 58 59; do
  $M run --detach modal_app.py --step train --dataset phoenix14t --kind teacher --seed $seed --run-id phoenix14t-teacher-seed$seed
done
$M run modal_app.py --step gloss_embedding --teachers phoenix14t-teacher-seed56
$M run modal_app.py --step teacher_pickle --dataset phoenix14t \
  --teachers phoenix14t-teacher-seed56,phoenix14t-teacher-seed57,phoenix14t-teacher-seed58,phoenix14t-teacher-seed59
for kind in mixup kd xmda; do
  $M run --detach modal_app.py --step train --dataset phoenix14t --kind $kind --teachers phoenix14t-teacher-seed56 --run-id phoenix14t-$kind-seed56
done
```

Each `train` call runs `python -m signjoey.training_XmDA --config ...` inside the Dockerfile image. The config is the upstream `configs/Sign_XmDA.yaml` with run paths applied at runtime and, unless `--hparams released`, Table 10 values (`modal_app.py:_config`). Training ends with `prediction_XmDA.test` on the best checkpoint. The function returns, and writes to `runs/RUN_ID/summary.json`, the parsed dev/test BLEU-4, ChrF, and ROUGE-L, the selected beam and alpha, SHA-256 of the `best.IT_*` outputs and `train.log`, peak GPU memory, and timings. The Volume is committed every 10 minutes.

**Two images, equivalent for the baselines.** The Dockerfile applies `patches/0001` and `patches/0002` at build time. The two baselines were launched before `patches/` existed, on Modal image `im-XO3R32x9PQI2frjfVefGye`. That ID was not logged at launch. It was recovered by rebuilding the pre-patch Dockerfile, which was a full content-addressed cache hit (probe app `ap-3EK5d2cgJY0qOUUCwqxaxX`). Every later run uses `im-x0vni1AxwKaBoFixz4bQRH`. The probe compared the two images' SignXmDA trees: only `signjoey/batch.py`, `model.py` and `prediction_XmDA.py` differ, and the differences are the two patches. They change only the `forward_type=gloss` path and the test pass's `forward_type`, which is `sign` for baselines. Empirically, the decode grids, run on the patched image from the baselines' checkpoints, reproduce both baselines' selected dev and test values exactly. The baselines were therefore not rerun.

**Long runs: launch with `--spawn`.** XmDA attempt 1 showed that a run longer than about 5 h must not be launched with `.remote()`. Under `run --detach` the local client stays blocked in `.remote()`. When its session token expired after 5 h 05 min (`AuthError: Jwt is expired`), the client errored, Modal logged "Received a cancellation signal" for the input, and the run was cancelled at step 28,000. With `--spawn`, `main` calls `train.spawn(...)`, which per Modal's API "calls the function with the given arguments, without waiting for the results", prints the function-call ID and returns. Per the CLI docs, `--detach` means "don't stop the app if the local process dies or disconnects".

The run's result is then `runs/RUN_ID/summary.json` on the Volume, read with `$M volume get ye-2023-xmda-results runs/RUN_ID/summary.json`. `train` sets `timeout=24 * 3600` explicitly; Modal allows "between 1 second and 24 hours". Modal's pages do not state in so many words that a spawned call in a detached ephemeral app outlives the client. The evidence is the two quoted statements, the CLI's own notice that detached mode "keeps the last triggered Modal function alive after the parent process has been killed or disconnected", and `preflight-phoenix14t-xmda-spawn`, whose call ran to completion after its client exited 15 s after launch. A preflight cannot show survival past the roughly 5 h token expiry, so that is shown only by the full attempt 2.

```bash
$M run --detach modal_app.py --step train --spawn --dataset phoenix14t --kind xmda \
  --teachers phoenix14t-teacher-seed56 --run-id phoenix14t-xmda-seed56-attempt-2
```

**Conditional chain and resume.** The PHOENIX-2014T chain is step 8 above, with `--spawn` for every run expected to take more than a few hours. For CSL-Daily, run the same commands with `--dataset csl-daily` and run IDs (gate `compute-full-plan`, resolved 2026-10-30).

**No real resume exists in the published code.** This is from reading `training_XmDA.py` at `b6e2ca4`; details are in `code_observations`.
- **Only the best checkpoint is ever saved.** A checkpoint is written only on a new best dev score, and `keep_last_ckpts: 1` keeps only that one. There is no last-step checkpoint.
- **A resume restores less than it seems.** `training.load_model` restores weights, optimizer and scheduler state, the step count and best-score tracking. It does not restore the epoch counter, the iterator position or RNG state.
- **A resume runs 150 more epochs.** The epoch loop always runs `range(150)` from 0, so a "resumed" run trains a further full 150 epochs.
- **`overwrite: true` deletes the run directory** before the checkpoint is loaded.

Changing any of this would change behaviour, so it was not patched. Reloading a checkpoint returns to the last improvement, not to the interruption. XmDA attempt 1 was interrupted at step 28,000, but its only checkpoint was from step 9,000.

So an interrupted CSL-Daily run can only be recovered by a full re-run. That costs about 8 GPU-h per teacher, about 18 for KD and about 20 for XmDA, the last two close to Modal's 24 h limit. A preemption (as with teacher seed 59) also restarts from scratch inside the same call.

## Data provenance and permissions

| Dataset | Version / splits | Source | License and cloud-use basis | Path in Volume `datasets` | Counts / checksums | Deviations |
| --- | --- | --- | --- | --- | --- | --- |
| RWTH-PHOENIX-Weather 2014T | Official v3; 210×260 PNG frames; official corpus CSVs; neccam/slt `pami0` annotations | RWTH FTP `phoenix-2014-T.v3.tar.gz`; Volume populated 2026-09-15 (`raw/PROVENANCE.txt`) | CC BY-NC-SA (non-commercial research), same basis as `papers/camgoz-2018-nslt`; processed in the project Modal workspace; derived features stay on the private results Volume | `rwth-phoenix-2014-t/` | 7,096 / 519 / 642 videos; 827,354 / 55,775 / 64,627 frames; no empty folders. CSV and annotation SHA-256 in `reproduction.json`; per-video frame manifest `2ff0cd16…6a3a6ae7b` | None |
| CSL-Daily | `csl2020ct_v2.pkl` with `split_1.txt`; `videos/*.mp4` | https://ustc-slr.github.io/datasets/2021_csl_daily/ (accessed 2026-10-28) | Research-only release agreement; gate `csl-daily-license` was resolved by the assignee on the `chen-2026-gloss-pretrained` precedent (read-only Modal copy, `cloud_processing_allowed: true`); Team S was not consulted | `csl-daily/` | 18,401 / 1,077 / 1,176 clips (20,654 mp4s, all matched); `split_1.txt` `45960c86…`, `csl2020ct_v2.pkl` `1fd29a33…`; mp4 manifest `d34b7bd0…` | Features from mp4 instead of official 512×512 JPGs; 18,400 usable train clips (one unlabelled clip skipped) |

**CSL-Daily restriction.** Never upload CSL-Daily text, glosses, frames, extracted features, predictions, or checkpoints trained on it to any public host, including the repro-sign Hugging Face organization. The dataset record has no `permission_model_weights`, so weights stay off public hosts too. Nothing from this attempt is published.

**Derived inputs** (private Volume `ye-2023-xmda-results`, recorded in `reproduction.json.runs` and `artifacts`):

| Dataset | Features (`.npy`) | Pickles |
| --- | --- | --- |
| PHOENIX | train `35bdabf9…`, dev `d1bba58f…`, test `c2c9e0f5…` | `e52449e3…`, `d904b51a…`, `219622ea…` |
| CSL-Daily | train `0bcab184…`, dev `d3a80bac…`, test `2860fadf…` | `0197ca3b…`, `6156ff9f…`, `cce7da78…` |

## Environment and patches

- **SignXmDA image (`Dockerfile`).**
  - Base and Python: `nvidia/cuda:11.4.3-devel-ubuntu20.04` (digest pinned) with Miniconda py3.7 (installer SHA-256 checked).
  - Stack: torch 1.4.0 (CUDA 10.1), torchtext 0.5.0, transformers 3.0.2, from neccam/slt `requirements.txt`. Five pins were adjusted only so that pip can resolve them (see the Dockerfile).
  - pip freeze SHA-256 `ad27af9123c8d2a10d06c4c800c749bce40f7c58e8a3562a033a285ff8137e2d`.
  - Modal adds Python 3.11 and pyyaml 6.0.2 for its wrapper only.
  - torch 1.4 has no Ampere (sm_80+) kernels, so training runs on T4.
- **SMKD image (`modal_app.py:SMKD_IMAGE`).**
  - Base: `ghcr.io/sign-language-processing/reproduction:latest` (torch 2.11, Python 3.12).
  - Pip pins: simple-video-utils 0.7.4, PyAV 18.0.0, opencv-python-headless 4.10.0.84, h5py, pyarrow, six, pandas, tqdm.
  - SLTUNet is pinned at `b1d1d0e8…`.
  - The extraction runs on L4.
- **Second SignXmDA image (`Dockerfile.cu110`, CSL-Daily student rows only).** Same as `Dockerfile` except torch 1.7.1+cu110, torchvision 0.8.2+cu110 and torchtext 0.8.1, installed in the same pip resolve. torchtext 0.8.1 is the last release whose `torchtext.data` still has the API SignXmDA uses. Modal image `im-zcYV3stvhV0g565br8nSMZ`, pip freeze SHA-256 `17115c8c3f6c282f870a983f13b69e207dbcfba82b504c51677ccd4424cc7f6c`; `train_cu110` in `modal_app.py`, selected with `--image cu110 [--gpu A10G]`.
- **Hugging Face cache.** Every function mounts `huggingface-cache` with `HF_HOME=/cache/huggingface` and `HF_HUB_CACHE=/cache/huggingface/hub`. No Hub artifact is used.
- **Upstream patches (`patches/`, applied in order by the Dockerfile).** Both affect only the `forward_type=gloss` (teacher) path.

  | Patch | SHA-256 | Concern | Kind |
  | --- | --- | --- | --- |
  | `0001-gloss-decoding-mirror-training.patch` | `7df063f36c8ad27adcfcb62da0787ed2314c1ed6b3b38d1e77c66a681912c33c` | `Batch` builds `gls_input` by repeating each gloss `ratio = sum(sgn_lengths) // sum(gls_lengths)` times (min 1), exactly as `Batch_Sign` does in training. It exposes `gls_input_mask` and `gls_input_lengths`, and `SignMixupModel.run_batch(gloss)` decodes from them. Training path untouched. | **Invented behavior choice** (gate `teacher-gloss-decoding`, resolved by the assignee). Atrewin/slt `9ecb427` added the repetition to training batches only, so no author evidence exists for validation or decoding. |
  | `0002-test-pass-forward-type.patch` | `c0d00eadc7a96edd464fe48bb45c37ac4ba16390f8c56685e48f0ad9545f0312` | Passes `testing.forward_type` to the final test pass in `prediction_XmDA.test()` (L581); upstream omits it, so it falls back to `sign`. | Correctness patch, kept by the assignee without a gate. `test()` passes `forward_type` at L415 and L479 but omits it at L581 (and at L1029 in `test_model_average`). No effect on sign models; see the two-image note above. |

- **Glue at the paper root (not upstream code).**
  - `modal_app.py` applies config values at runtime.
  - `extract_smkd_features.py` extracts features with SLTUNet's own `SignEmbedding`.
  - `build_pickles.py` writes neccam/slt-format pickles.
  - `dump_gloss_embedding.py` dumps the best teacher's `GlossEmbeddings` for the Mix-up and XmDA students.
  - `modal_app.py:decode_grid` re-decodes a finished run's best checkpoint one beam/alpha setting at a time through the upstream `prediction_XmDA.test`.
- **Changes to SLTUNet's extractor at runtime (features unaffected).**
  - The extractor stubs `ctcdecode` and replaces the CTC decoder with a no-op, because MaxDecode fails on float lengths under torch 2.11.
  - It sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for the hash-verified checkpoint.
  - For CSL-Daily only, it replaces mp4 reading with a simple-video-utils decode.
- **Observed upstream behavior, recorded as evidence and not changed:**
  - ×15 factor on the JSD consistency loss and the embedding JS term (`training_XmDA.py` L937).
  - Stochastic mix-up and `_train_batch` gates (L1063–1068, L745).
  - Mix mask `gloss_predict.ge(1 - mixup_ratio)` on label indices.
  - BatchNorm statistics still updating in the frozen gloss embedding.
  - Early stopping disabled.
  - README's `python -m signjoey train_XmDA` is rejected by `__main__.py`.

## Execution evidence

All runs used Modal profile `repro-sign`, app `repro-ye-2023-xmda`, and agent `claude-opus-5-5-claude-code`. Dashboard links are `https://modal.com/apps/repro-sign/main/` followed by the app ID. Times are UTC; the first twelve rows are from 2026-10-28, and the rest are from 2026-10-29.

| Run ID | Attempt / max | Kind | Hardware | Start → end (UTC) | Exit / terminal / reason | Failure class | Ceilings (wall s / GPU-h / CHF) | Modal app / function call |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `smkd-phoenix14t-train` | 1/1 | features | L4 | 13:02 → 15:39 | 0 / succeeded / completed | — | 86400 / 6 / 5.3 | ap-uuOeqrFwQ1HzzEGHSNv0bh / fc-01M619KAEEY4F26RYBVW9C157T |
| `smkd-phoenix14t-test` | 1/2 | features | L4 | 16:13 → 16:15 | 0 / succeeded / completed | — | 86400 / 24 / 19.7 | ap-vf5c5gYiOm8D1bqfS8u1rv / fc-01M61MCFFG3YY9H57JZEVMW1FJ |
| `smkd-phoenix14t-dev` | 1/2 | features | L4 | 16:16 → 16:18 | 0 / succeeded / completed | — | 86400 / 24 / 19.7 | ap-1wNOusI7sHQjqcip0KX1Go / fc-01M61MCFAVE47A04Q3K3DK95X2 |
| `preflight-smkd-csl-daily-dev16` | 1/3 | preflight | L4 | 15:57 → 15:57 | 0 / succeeded / completed | — | 86400 / 6 / 5.3 | ap-z7JFEo3p5AOlXpLC1ejhxC / fc-01M61KM0ZWY6AZKFCZE6861E1C |
| `smkd-csl-daily-dev` | 1/2 | features | L4 | 16:04 → 16:07 | 0 / succeeded / completed | — | 86400 / 24 / 19.7 | ap-sTbJXsQDIefRcpcDsL2ZW8 / fc-01M61KQ7CWJSW3RWQNKW002WBQ |
| `smkd-csl-daily-test` | 1/2 | features | L4 | 16:10 → 16:14 | 0 / succeeded / completed | — | 86400 / 24 / 19.7 | ap-kndVTF9rhx0jqOTEQh0Z4N / fc-01M61MCT2V8ENEKPH34KJ9JE6R |
| `smkd-csl-daily-train-attempt-1` | 1/2 | features | L4 | 16:09 → 16:10 | 1 / failed / command_failed (KeyError on unlabelled clip) | deterministic_code | 86400 / 24 / 19.7 | ap-bB4x6WgeGntYotBIRDe6jS |
| `smkd-csl-daily-train-attempt-2` | 2/2 | features | L4 | 16:15 → 17:36 | 0 / succeeded / completed | — | 86400 / 24 / 19.7 | ap-xqMTkHGEnKsuhdgppGT4ij / fc-01M61MQR90W8FMGJFYBX6XQ6YZ |
| `preflight-phoenix14t-baseline-attempt-1` | 1/3 | preflight | T4 | 16:19 → 16:20 | 1 / failed / command_failed (no yaml in wrapper Python) | deterministic_code | 7200 / 2 / 2 | ap-k3zz2ETrd3NQWKY6wk915D |
| `preflight-phoenix14t-baseline-attempt-2` | 2/3 | preflight | T4 | 16:21 → 16:22 | 0 / succeeded / completed | — | 7200 / 2 / 2 | ap-YC4D9EypjdtFNpnRCcyzKW / fc-01M61N1YDHYVS1MBZAFHWZGJ7R |
| `preflight-phoenix14t-teacher` | 1/3 | preflight | T4 | 16:25 → 16:25 | 1 / failed / command_failed (AttributeError `gls_input_mask` in validation) | deterministic_code | 7200 / 2 / 2 | ap-ckYIYQ0DS46PWvNLMfv95e / fc-01M61N7AQE8XH49H33G8RYHHV2 |
| `preflight-csl-daily-baseline` | 1/3 | preflight | T4 | 17:39 → 17:41 | 0 / succeeded / completed | — | 7200 / 2 / 2 | ap-5VsXhyN0N76e0YqcJNAJGl / fc-01M61SGK386WMCFQ73GWRCXVK5 |
| `phoenix14t-baseline-seed56` | 1/2 | full run | T4 | 08:50 → 09:46 | 0 / succeeded / completed | — | 14400 / 4 / 3 | ap-KjFyxQSiA8O6D5UVbsdBdZ / fc-01M63DMWJY2KT9FBZ1T6T87H9B |
| `csl-daily-baseline-seed56` | 1/2 | full run | T4 | 08:51 → 12:51 | 0 / succeeded / completed | — | 21600 / 6 / 5 | ap-mR6Hje1wpgmTnKCTRzubEk / fc-01M63DMWGSKNETP3Y9BJ2ZRHDQ |
| `preflight-phoenix14t-teacher-attempt-2` | 2/3 | preflight (patch 0001) | T4 | 08:57 → 09:02 | 0 / stopped / invalid_run (training-set translations decoded through the sign path; fixed by patch 0002) | deterministic_code | 7200 / 2 / 2 | ap-YPlaKzVVs6pXWeU6C6OhxW / fc-01M63E0WSC1T7C6D7QEMTVJ6A9 |
| `preflight-phoenix14t-teacher-attempt-3` | 3/3 | preflight (patches 0001+0002) | T4 | 09:10 → 09:15 | 0 / succeeded / completed | — | 7200 / 2 / 2 | ap-jqliSAmvAksHfRFRGABXPO / fc-01M63ES34SM6YQS23NSQNBV5DY |
| `phoenix14t-teacher-seed56` | 1/2 | full run (conditional) | T4 | 14:43 → 17:21 | 0 / succeeded / completed | — | 14400 / 4 / 3 | ap-eKeFvKr5UbGBXxEKXA4dkn / fc-01M641T17T167YFWMMKHRP31HY |
| `phoenix14t-teacher-seed57` | 1/2 | full run (conditional) | T4 | 14:44 → 17:06 | 0 / succeeded / completed | — | 14400 / 4 / 3 | ap-vgGTiI0sgHkuj7WnX78mH3 / fc-01M641TMCZGQF9XDJQK5PAPRE0 |
| `phoenix14t-teacher-seed58` | 1/2 | full run (conditional) | T4 | 14:43 → 17:22 | 0 / succeeded / completed | — | 14400 / 4 / 3 | ap-x3biFBgWLOqKTH51DuyIXb / fc-01M641V7ZES3YRF9EP3HXT46TY |
| `phoenix14t-teacher-seed59` | 1/2 | full run (conditional) | T4 | 15:17 → 18:09 (first container 14:44, preempted 15:16) | 0 / succeeded / completed | — | 14400 / 4 / 3 | ap-t7hTw4oEu6Zsb0IOWvHhv4 / fc-01M641VVSKP0J70JP0XQMAXSF5 |
| `decode-grid-phoenix14t-baseline-seed56-attempt-1` | 1/2 | evaluation | T4 | 14:46 → 14:52 | 0 / stopped / invalid_run (loop overwrote the alpha list; 7 of 16 settings) | deterministic_code | 3600 / 1 / 1 | ap-08MhI0go4dZCAr62eTtVly / fc-01M6420ANBJZJT52KZHVEVSA2F |
| `decode-grid-csl-daily-baseline-seed56-attempt-1` | 1/2 | evaluation | T4 | 14:46 → 14:56 | 1 / stopped / invalid_run (same bug; stopped by the agent) | deterministic_code | 10800 / 3 / 2 | ap-amsodJUIbU8Ed2CjEhGH5D |
| `preflight-csl-daily-baseline-released-config` | 1/3 | preflight | T4 | 14:46 → 14:48 | 0 / succeeded / completed | — | 7200 / 2 / 2 | ap-904vhx25Gw2Ow4q9TlgvAU / fc-01M64217T5V0CEFZ1MBST4C63C |
| `csl-daily-baseline-released-config-seed56` | 1/2 | full run (conditional) | T4 | 14:49 → 17:56 | 0 / succeeded / completed | — | 21600 / 6 / 5 | ap-6laVgBOfEAe0ItpL8e3EP8 / fc-01M6425YG93H5BZPRNK0ESWAGY |
| `preflight-phoenix14t-mixup` | 1/3 | preflight | T4 | 14:50 → 14:52 | 0 / succeeded / completed | — | 7200 / 2 / 2 | ap-hMsaOBDU9cwyT0V4ojUAK7 / fc-01M6428WQ3V5NPVXZ99HCAA4C0 |
| `preflight-phoenix14t-kd` | 1/3 | preflight | T4 | 14:51 → 14:54 | 0 / succeeded / completed | — | 7200 / 2 / 2 | ap-SEOWFunFh1y1ivlvKugSLs / fc-01M64295XJNV9K4TERDQ6KKWVD |
| `preflight-phoenix14t-xmda` | 1/3 | preflight | T4 | 14:51 → 14:55 | 0 / succeeded / completed | — | 7200 / 2 / 2 | ap-K8bwi5MxtP9QCjNVYRthVI / fc-01M6429FNEBVV8YWTP8WDB5XXE |
| `decode-grid-phoenix14t-baseline-seed56-attempt-2` | 2/2 | evaluation | T4 | 14:57 → 15:12 | 0 / succeeded / completed | — | 3600 / 1 / 1 | ap-htJhr5sCDzIJC6SUNDCIDA / fc-01M642KY2C1VFNCAWK3GHWZFJM |
| `decode-grid-csl-daily-baseline-seed56-attempt-2` | 2/2 | evaluation | T4 | 14:57 → 15:38 | 0 / succeeded / completed | — | 10800 / 3 / 2 | ap-ZV4TDBG6hf4tTTB56kskDS / fc-01M642MC68CW3DEKSRKGNPYXQ6 |
| `phoenix14t-gloss-embedding` | 1/3 | data preparation | CPU | 18:09 → 18:10 | 0 / succeeded / completed | — | 1800 / — / 1 | ap-owcuEaWSHseh3n0Dxs2GHp |
| `phoenix14t-teacher-pickle` | 1/3 | data preparation | CPU | 18:09 → 18:10 | 0 / succeeded / completed | — | 14400 / — / 1 | ap-jWLM3VbLyveGsuhDpUUQnJ |

| `phoenix14t-mixup-seed56` | 1/2 | full run (conditional) | T4 | 18:11 → 20:01 | 0 / succeeded / completed | — | 10800 / 3 / 2 | ap-KTzbLXBYkXAAc6FmSHc6qX / fc-01M64DPNBTPWPDWKHD15F60DCM |
| `phoenix14t-kd-seed56` | 1/2 | full run (conditional) | T4 | 18:10 → 20:48 | 0 / succeeded / completed (stopped at `learning_rate_min`, step 28,000) | — | 28800 / 8 / 5 | ap-NSXQb5PKC42B4E6FjgrCva / fc-01M64DPNBWEQSJ0EFTHM5D483F |
| `csl-daily-teacher-seed56` … `-seed59` (2026-10-30) | 1/1 each | full run (conditional), T4 | T4 | 11:18 → 13:23 | 1 / stopped / no_new_hypothesis (projected 22 h against the 24 h cap; replaced by the A10G runs) | budget_protocol | 72000 / 20 / 12 | ap-7d7JmYAaKIEqbsHXjr2EOI, ap-HU3fS11PGMMBuCwmy6jzJM, ap-bSl6c9KtjeQ1KAqlKuGGGc, ap-1DnvIVCeHJdcHYHhXkL0Hf |
| `preflight-phoenix14t-xmda-spawn` (2026-10-30) | 1/3 | preflight (`--spawn`) | T4 | 09:40 → 09:44 | 0 / succeeded / completed | — | 7200 / 2 / 2 | ap-ZYXKS0UqAhwZ8ukb63a3KX / fc-01M662X1QT5DD3MZ5XNBAW8SPN |

**XmDA (group `phoenix14t-xmda-seed56`, ceilings 32400 s / 9 GPU-h / CHF 6 per attempt).**
- **Attempt 1** (run directory `runs/phoenix14t-xmda-seed56/`, app `ap-FNsuLjdoB4QBQHVy7ehfgr`, input `in-01M64DPNGN8AZ30K5PTYGSPW3N`). It ran 18:10 → 23:16 UTC on 2026-10-29 and was interrupted at epoch 101, step 28,000, after about 5.1 GPU-h. Cause: the local client's session token expired while it waited in `.remote()`, and the input was cancelled. Its best dev BLEU-4 was 25.46 (greedy) at step 9,000, the only checkpoint on the Volume. Retained evidence: `train.log` `552dafc5…`, `validations.txt` `034c0257…`, `9000.ckpt` `01cbedb5…`.
- **Attempt 2** (`phoenix14t-xmda-seed56-attempt-2`, a restart from scratch approved by the assignee). It was launched with `--spawn` at 09:44 UTC on 2026-10-30 as app `ap-0wipLeafqGhD0UEUZQAYAo` / `fc-01M6635FNYF7FQ8FDV7H5M3QMT`, and is estimated at about 7.7 GPU-h. It is still running.

Both attempts will be recorded in `reproduction.json.runs` together when attempt 2 ends.

**Ceiling amendments (2026-10-30T13:25:49Z, assignee decision).** These were made before any metric of the affected runs was inspected; only epoch progress was read. The original values are kept as superseded.
- XmDA seed 56 attempt 2 and XmDA seed 58: 9.0 → 11.0 GPU-h. XmDA seed 57 followed at 2026-10-30T13:30:59Z, so all three XmDA seeds share one ceiling.
- KD seeds 57 and 58: 8.0 → 10.5 GPU-h.
- Wall time and the CHF caps were raised in proportion.

Reason: the compute gate's estimate had noted that the mix-up/XmDA per-step cost was unmeasured, and the 2-epoch preflights did not measure full-dev validation cost. The measured pace then projected about 9.0 GPU-h for XmDA attempt 2 and about 9.4 for KD seed 57. From now on, ceilings carry at least 30% margin over the measured estimate.

**CSL-Daily teachers moved from T4 to A10G (assignee decision, 2026-10-30T13:23Z).** On T4, each full-dev validation in the teacher's gloss path took about 152 s, projecting about 22 h per teacher against Modal's 24 h cap. A teacher writes its outputs only at the end, so hitting the cap would lose everything. The four T4 runs were stopped at epochs 13–15 (about 2.1 GPU-h each) and relaunched on A10G in the cu110 image, with a ceiling of 21 h / 21 GPU-h each. The run contract does not allow a stop on a human decision to point a retry at a new attempt, so the T4 runs are recorded as single-attempt groups: `stopped / no_new_hypothesis / budget_protocol / stop`, citing gate `compute-full-plan`, with the A10G successor named in the detail. The A10G runs `csl-daily-teacher-seed56..59-attempt-2` are each attempt 1 of 3 in their own group. The first estimate of about 16 h each was extrapolated. Measured A10G validations take about 112 s, which puts each teacher at about 14–17 h, central about 15.5 h, against a 21 h ceiling. This will be re-checked from the observed epoch rate. Attempt 1 will be `interrupted / external_interruption / interrupted_full_run`, with `retry_decision: retry`, and its `next_run_id` must point to an existing entry.

**Throughput and cost (T4, batch 128).**

| Run | Seconds per step | Full-dev validation | Peak memory (MiB) | GPU-hours |
| --- | --- | --- | --- | --- |
| PHOENIX baseline, full (3 layers, 8,400 steps, 84 validations) | 0.21 | about 12 s | 4,488 | 0.92 |
| CSL-Daily baseline, full (1 layer, 21,600 steps, 216 validations) | 0.13 | about 45 s (two thirds of the run) | 8,187 | 4.01 |
| PHOENIX teacher preflight (gloss input plus interleaved PGen batch) | 0.30 | — | 6,665 | 0.09 |
| PHOENIX teachers, full (4 seeds) | — | — | 7,226–7,642 | 2.36–2.65 each; 3.41 for seed 59 including the preemption |
| CSL-Daily baseline, released config, full (hidden 256) | 0.085 | about 39 s | 7,662 | 3.12 |
| PHOENIX Mix-up / KD / XmDA preflights | 0.32 / 0.24 / 0.33 | — | 6,479 / 6,283 / 11,396 | 0.02 / 0.06 / 0.07 |

Each preflight trained on full data, saved a checkpoint, and had `prediction_XmDA.test` reload it and emit dev and test metrics. The final decode of the full runs (16-setting dev grid plus one test pass) took 314 s on PHOENIX and 1,343 s on CSL-Daily. Decoding the teacher's 7,096-sentence training set took 228 s.

**Compute used.** Finished runs so far used about 34.5 GPU-hours: 4.1 on L4 for SMKD extraction and about 30.4 on T4, about USD 18 for the T4 part at the list price of about USD 0.59/h. The T4 part breaks down into:
- the two baselines, 4.93;
- the four teachers, 11.06;
- the released-config baseline, 3.12;
- Mix-up, 1.85, and KD, 2.64;
- the interrupted XmDA attempt 1, about 5.1;
- the decode grids, 1.19;
- the preflights.

XmDA attempt 2 is estimated at about 7.7 more, which brings the PHOENIX-2014T chain to about 28 GPU-h, as the assignee approved. The overwritten PHOENIX dev/test extractions listed in the dead ends add a little more. GPU details are in each run entry: L4 23,034 MiB and T4 15,360 MiB, driver 610.57.04.

## Guesses and deviations

| Detail | Paper/evidence says | This attempt used | Rationale | Effect on interpretation |
| --- | --- | --- | --- | --- |
| Sign features | "SMKD features", 1024-d; checkpoint and layer not stated | SLTUNet's released SMKD checkpoints; TemporalConv `visual_features` | The paper says it follows Zhang et al. 2023a; WER control matches SLTUNet Table 12 closely on PHOENIX | Guess; feature differences propagate to every target |
| ff_size | Not in Table 10 | 2048 | Authors' earlier fork `Atrewin/slt` `configs/jizhi/default.yaml` | Guess |
| Seeds | Not stated | Seed 56, one per cell | Released config's `random_seed` | Guess; no variance estimate |
| Config precedence | Table 10 conflicts with `configs/Sign_XmDA.yaml` | Table 10 wherever stated; upstream values elsewhere | Gate `hyperparameters` | Guess; released config kept as alternative |
| Embedding activation | Table 10 "activation gelu" | GELU for encoder and decoder embeddings | Table 10 | Guess |
| CSL-Daily text | Level not stated in the paper | `label_char` joined by spaces, level `word` | Evidence, not a guess: the authors' builder writes `'text': char` and saves `cslt.word.{split}.pickle.char` (Atrewin/slt `concat_data.py` L181, L189), matching the upstream config's `.pickle.char` files | CSL metrics are character-level, as the authors computed them |
| CSL-Daily frames | Official 512×512 JPG frames (SLTUNet) | Volume mp4s (H.264 1280×1280, 30 fps), decoded by simple-video-utils and resized to 256×256 by Lanczos | Only the mp4 copy is on the Volume; the study requires simple-video-utils for decoding | Deviation; lossy codec and different source resolution |
| CSL-Daily train size | 18,401 clips in `split_1.txt` | 18,400 | `S000005_P0004_T00` has no annotation (the pkl instead holds `S000007_P0003_T00`, which has no video and no split) | Deviation; one clip |
| Teachers (conditional) | PGen fairseq Gloss2Text, Table 15 | signjoey `forward_type=gloss` with `G2T_pretraining` | PGen's teacher trainer is unreleased; SignXmDA README offers this route | Deviation; conditional runs only |
| Pseudo-pairs (conditional) | Real pairs ×40 plus unreleased ~800k BT set | Released `DA_paralle_sample`: 30,492 De and 358,233 Zh pairs | Only released data | Deviation; conditional only |
| Teacher seeds and subsets (conditional) | "varying seeds and subsets", unspecified | Seeds 56–59, same data | Invented | Deviation; teacher dev BLEU-4 26.66 / 26.62 / 26.60 / 26.63 against Table 15's 27.35 / 27.24 / 27.11 / 27.02 |
| Teacher validation/decoding (conditional) | Not described; pinned code cannot do it | `patches/0001`: gloss input repeated as in `Batch_Sign` training | Gate `teacher-gloss-decoding`, resolved by the assignee; Atrewin/slt `9ecb427` only covers training | Invented behavior choice; conditional only |
| Teacher test pass (conditional) | Upstream omits `forward_type` on the test pass | `patches/0002` passes `testing.forward_type` | Execution bug shown by `preflight-phoenix14t-teacher-attempt-2`; kept by the assignee as a correctness patch | Bug fix, not a deviation; no effect on sign models |
| CSL-Daily student environment (conditional) | Pinned stack torch 1.4.0 / CUDA 10.1, which has no Ampere kernels | `Dockerfile.cu110`: torch 1.7.1+cu110, torchtext 0.8.1, same Python and other pins, no code changes; image `im-zcYV3stvhV0g565br8nSMZ`, freeze `17115c8c…`; A10G | Dependency-compatibility change; semantics audit clean; gate `environment-cu110` criterion passed | Deviation; within-table deltas taken against an in-image reference baseline |
| Frozen gloss embedding (conditional) | From the best teacher; conversion undocumented | Best teacher's `GlossEmbeddings` state_dict and its `gls.vocab` | `build_model_mixup` loads exactly that module | Deviation; conditional only |

## Attempts, failures, and dead ends

- **Ingestion.** `ingest_candidate.py` rejects this PocketBase export layout: the paper is nested under `expand.paper` and there is no `confirmation` field. The assignment is therefore recorded as a direct request with the export hash.
- **SLTUNet extractor under torch 2.11.**
  - MaxDecode raises a TypeError on float `feat_len`. Fix kept: the decoder is replaced with a no-op and a greedy CTC WER is computed separately; features are unchanged.
  - The checkpoint fails to load under the `weights_only` default. Fix kept: `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, for the hash-verified release only.
- **simple-video-utils missing from the ghcr image.** Hypothesis: the package is absent. Test: CSL-Daily preflight `ap-KUd5tJXefETsjQB3fYhPvQ` raised ModuleNotFoundError. Fix kept: pin 0.7.4, matching the root Dockerfile.
- **PyAV 19.0.2 rejects `metadata_errors`.** Test: `ap-81f85y4l5BkxNFK3Cc9Db0` reported "Failed to open video" for every clip. Fix kept: pin av 18.0.0, the lowest version simple-video-utils allows. The 16-clip dev preflight then succeeded.
- **`Path.with_suffix` bug in the glue.** `DATASET.smkd.SPLIT` lost its split suffix, so the first PHOENIX train, dev, and test extractions all wrote the same file and train overwrote the others. The first pickles had 7,096 entries for every split, which exposed the bug before any training. Fix kept:
  - extensions are now appended;
  - verified survivors were renamed and stale pickles deleted;
  - dev and test were re-extracted, and the new files are byte-identical to the hashes the overwritten runs reported.
- **CSL-Daily train KeyError on an unlabelled clip** (`smkd-csl-daily-train-attempt-1`). Hypothesis: one `split_1.txt` name has no label. The fix skips names absent from `csl2020ct_v2.pkl` and prints them. Attempt 2 succeeded with 18,400 clips.
- **Wrapper Python without yaml** (`preflight-phoenix14t-baseline-attempt-1`). Fix kept: pyyaml 6.0.2 in the Modal-added Python only. Attempt 2 succeeded.
- **Teacher validation crash** (`preflight-phoenix14t-teacher`). This is a behavior-defining gap in the published gloss path, not an environment problem. Gate `teacher-gloss-decoding` was opened, and the assignee resolved it with `patches/0001`.
- **Degenerate teacher outputs** (`preflight-phoenix14t-teacher-attempt-2`, with patch 0001). Validation worked (dev BLEU-4 19.83 on 32 sentences), but the training-set translations had BLEU-4 0.76 and WER 100, and 5,012 of 7,096 began with "in der neuen woche". Hypothesis: the test pass decodes through the sign path because `test()` omits `forward_type`. Fix kept: `patches/0002`. Attempt 3 then gave BLEU-4 11.56 with 5,537 distinct outputs after 2 epochs.
- **Full plan above 24 GPU-hours.** Gate `compute-full-plan` was opened. The assignee approved the two baselines separately, and they ran to completion. The assignee then approved the PHOENIX-2014T chain. On 2026-10-30 the assignee resolved the gate for the full plan, on the project lead's standing instruction to use more Modal compute.
- **Estimate revision.** The preflight-only estimate (about 63 GPU-h) assumed about 15 s per CSL-Daily dev validation. The full run measured about 45 s, so the estimate was revised to about 93 GPU-h for the conditional runs.
- **Decode-grid loop bug** (both `decode-grid-*-attempt-1` runs). `decode_grid` updated `config["testing"]` inside the loop, which replaced the alpha list after the first setting, so beams 2–4 ran only alpha 1. Fix kept: the beam and alpha lists are read once before the loop. The PHOENIX attempt finished with 7 of 16 settings and the CSL-Daily attempt was stopped; both attempt 2 runs covered all 16 settings.
- **Teacher seed 59 preemption.** Modal preempted the container at about 15:16 UTC (after epoch 39) and restarted the function with the same input. Because the config has `overwrite: true`, training restarted from scratch; the epoch-1 loss repeats exactly. About 0.54 GPU-h were lost, and the run finished within its 4 GPU-h ceiling. This is the reason resume must be verified before the long CSL-Daily runs.
- **XmDA attempt 1 cancelled by client-token expiry.** Hypothesis: a client blocked in `.remote()` under `run --detach` loses its session token after about 5 h, and the error cancels the call. Evidence: the cancellation came at 5 h 05 min, and the log shows `AuthError: Jwt is expired` right after "Received a cancellation signal". Every earlier run was shorter than 5 h. Fix kept: `--spawn`, checked by `preflight-phoenix14t-xmda-spawn`. Attempt 2 is a from-scratch restart (assignee decision); no resume was attempted (see How to repeat this).
- **Teacher image ID.** The teachers' summaries predate the `MODAL_IMAGE_ID` field. Their image is recorded as `im-x0vni1AxwKaBoFixz4bQRH`, inferred from the released-config run launched 7 minutes later from the same Dockerfile and patches.

## Candidate flags, ethics, and human evaluation

- The queue's copied-score flag refers to the "Previous Research" rows. They are ledgered as `copied_baseline` and were not rerun.
- No human evaluation or participant interaction is involved.
- Both datasets are existing, already-collected research corpora, used under the permission bases above. CSL-Daily is handled under the no-public-upload restriction.

## Author and team contact

- No authors were contacted.
- The CSL-Daily license gate was resolved by the assignee on precedent, without consulting Team S.
- Gates `teacher-gloss-decoding` and `compute-full-plan` were handled in `#repro-sign-team-r` through the assignee. Both are resolved. `compute-full-plan` was resolved on a standing instruction from the project lead to use more Modal compute; no discussion specific to this plan took place.
