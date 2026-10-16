# ub-HRPro (Murai et al., 2025) reproduction

**Paper ID:** `d0d5752366a6ace24df57ba7994f37ffc4657ffc`

**Citation:** Ryota Murai, Naoto Tsuta, Duk Shin, and Yousun Kang. Point-Supervised Japanese Fingerspelling Localization via HR-Pro and Contrastive Learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, pages 4975-4982, October 2025. doi:10.1109/ICCVW69036.2025.00516.

**Paper:** https://openaccess.thecvf.com/content/ICCV2025W/MSLR/papers/Murai_Point-Supervised_Japanese_Fingerspelling_Localization_via_HR-Pro_and_Contrastive_Learning_ICCVW_2025_paper.pdf · **Code/artifacts:** https://github.com/tpu-kanglabs/ub-hrpro (MIT) · https://huggingface.co/kanglabs/ub-hrpro (checkpoints) · https://huggingface.co/datasets/kanglabs/ub-MOJI (gated data)

**Preference level:** 1

**Pipeline status:** `blocked_on_data`

**Numerical agreement:** `not_assessed` — no in-scope target produced a comparable value, so there is no reproduced number to compare against any of the 50 published Table 2 values.

**Attempt date:** 2026-09-24

## Summary

The authors published a complete, well-pinned implementation, and it is the right artifact to run: this attempt stays at preference level 1 and applies no patch. The reproduction is blocked one step earlier, on data.

Two independent data blockers apply to all 50 Table 2 numbers:

1. ~~**Access.**~~ **Resolved 2026-09-24.** The user accepted the ub-MOJI Terms of Use; authenticated access now works. This is no longer a blocker.
2. **The public releases do not contain the paper's splits.** With access granted, I checked this two independent ways before downloading anything.

   *How the split is identified.* The upstream `split_test.txt` is unpublished, so the only authoritative record of the splits actually used is the authors' released `proposal.json`. Its `test` key holds exactly 24 IDs, from exactly two participants (018, 019), in a single session (202310) — matching Section 5.1's "24 untrimmed videos from Dataset A" recorded by two instructors. Agreement on count, participant structure, and subset is strong corroboration that this is the reported evaluation split.

   *Check 1 — file enumeration.* Every file at all four tags, matched against those IDs:

   | Tag | Videos | Size | Paper IDs w/ file | Held-out usable |
   | --- | ---: | ---: | ---: | ---: |
   | `v25.05` | 343 | 3.92 GB | 67 / 259 | 20 / 24 |
   | `v25.07` | 754 | 5.39 GB | 163 / 259 | 19 / 24 |
   | `v25.09` | 957 | 5.79 GB | 163 / 259 | 19 / 24 |
   | `v26.09` (current `main`) | 6,691 | 21.87 GB | 172 / 259 | 19 / 24 |

   **Which version?** The primary analysis is `main` at commit `82d24aa445e8bc739aff6315ee7bbc6766cf3ea6` (2026-09-13), whose `annotations.toml` is byte-identical to tag `v26.09`. But the paper is from October 2025, so I re-ran the whole usability test against all four tags. The paper itself **pins no version** — reference [15] cites only the bare URL and "2025" — even though the dataset card advises specifying one for reproducibility. Training coverage by tag: 47, 142, 142, 151 of 235. So picking a paper-contemporaneous version does not recover anything and makes training coverage much worse.

   Four held-out IDs — `kankoku_019_202310_t001`, `tsurukawa_018_202310_t001`, `tsurukawa_019_202310_t001`, `yokosuka_019_202310_t001` — return **zero** hits when substring-searched across all 6,700 repository paths, while present controls such as `kankoku_018_202310_t001` return exactly one hit under the identical naming convention. That rules out renames, extension changes, and relocation.

   *Check 2 — the dataset's own annotation index.* `annotations.toml` (1,302 annotated video IDs) is keyed by video ID and does not depend on filenames or directory layout at all. It contains **19** of the 24 held-out IDs. It flags the same four missing videos, plus one more the file listing could not catch: `kamakura_019_202310_t001` has an `.mp4` file but **no annotation entry**, so it has no ground-truth segments to score against.

   The two methods agree on every video both cover, and the annotation index is strictly stricter. The usable count is therefore **19 of 24** held-out videos, and **151 of 235** enumerated training videos (the paper reports 240). Six participant IDs the paper used (001, 002, 004, 005, 006, 009) appear in no release; the current release instead adds participants 023–026, which postdate the paper.

   The paper's Section 3.2 statement — *"While our model was trained on data from both sessions, only data from Session 2 (13 signers) is made public due to participant consent agreements"* — explains part of the **training** shortfall but not the missing **evaluation** videos. Table 2 cannot be reproduced on its reported split; evaluating on 19 of 24 would be a different measurement, not a reproduction.

### Re-running the coverage check

`verify_coverage.py` in this directory reproduces both checks from scratch against any revision, printing usable counts and each unusable held-out video with its cause. It needs a Hugging Face token whose account has accepted the ub-MOJI terms, and downloads no video.

```bash
python3 papers/murai-2025-jp-fingerspelling-hrpro/verify_coverage.py main
```

Verified on 2026-09-24 to reproduce the recorded figures exactly (19/24 held-out, 151/235 train, 1,302 annotated IDs, same five unusable videos).

### Is ub-MOJI public?

Two different questions, with different answers — and conflating them would misreport the paper.

**The dataset is genuinely public.** On `main`/`v26.09` it publishes 1,297 continuous videos (1,272 annotated) plus an isolated subset, under a click-through academic-research licence this study has now accepted. It is real, usable, and larger than what the paper used.

**The paper's data is not.** Only **170 of the 259** videos the paper's own splits enumerate are obtainable from any release — **65.6%**:

| | Paper's splits | Available | Missing |
| --- | ---: | ---: | ---: |
| Videos | 259 | 170 | 89 |
| Signers | 18 | 12 | 6 |
| Held-out partition | 24 | 19 | 5 |
| Training | 235 | 151 | 84 |

The release is not a superset of the paper's data. It contains 1,102 videos the paper never used — including four signers (023–026) added *after* publication — while missing six of the paper's eighteen signers entirely.

So the accurate statement is not "the dataset is private." It is: **the released dataset is not the dataset the paper ran on.** Roughly a third of the paper's videos, a third of its signers, and a fifth of its evaluation partition are unavailable at every published version.

One further discrepancy: Section 3.2 says Dataset B had 17 signers and that "only data from Session 2 (13 signers) is made public", implying **four** withheld signers. **Six** of the paper's signers are actually absent. The stated consent reason does not fully account for the absence even at signer granularity — let alone the 38 video-level gaps inside signers who *are* present.

### What exactly is missing, and why

**Both signers and individual videos — and the two are different in kind.** Of the 259 videos the paper's splits enumerate, 170 are usable:

| Loss mechanism | Videos lost |
| --- | ---: |
| Six signers absent entirely (001, 002, 004, 005, 006, 009) | 51 |
| Scattered gaps *inside* the twelve signers that are present | 38 |
| **Usable** | **170** |

The six absent signers have zero videos anywhere in the release — a clean, complete removal. The 38 remaining losses are different: **no present signer is complete** with respect to the paper's splits. 36 have no video file; 2 have a file but no `annotations.toml` entry.

**The only stated explanation is participant-level.** The dataset card says:

> Please note that a portion of the dataset is not publicly available, as some participants did not provide consent for open release.

The paper's Section 3.2 says the same thing in different words, about Dataset B training data. That plausibly covers the six absent signers — but it does **not** explain the 38 video-level gaps, and critically it does not explain the evaluation losses: **all 24 held-out videos belong to participants 018 and 019, both of which are present in the release.** The five unusable held-out videos are video-level losses inside present signers. **The held-out split is not complete** — it is missing data just as the training split is, though proportionally less.

Two further observations complicate the consent story:

- Participants **006 and 009 are listed in `participants.csv`** with full demographic metadata, yet have zero videos. "Did not consent to open release" does not cleanly explain those two.
- `face_visibility` consent does **not** correlate with the gaps — signers with the flag set to both `1` and `0` show losses, and none is complete either way.

**The authors treat at least some gaps as omissions to be fixed.** CHANGELOG v26.09 records *"Added missing videos and corrected annotations for 44 sequence samples"*, and Hub discussion #28, opened by the paper's first author, is titled *"Add the missing annotations and videos."* The release is being incrementally repaired, not held fixed.

**No source states why any specific video or signer is absent.** I reviewed all 29 ub-MOJI Hub discussions, all 3 `ub-hrpro` GitHub issues, the dataset card, LICENSE, CHANGELOG, and the paper. There is one participant-level consent note and nothing more granular.

One pattern is visible but unexplained: word recordings are dropped disproportionately (18 of 63 word videos vs 18 of 145 gojūon-sequence videos among present signers), with the heaviest losses on place and proper nouns — `tsurukawa` (7 of 11 lost) and `kankoku` (7 of 12). I can see the pattern; I cannot say what causes it.

3. **Modal redistribution rights, unresolved.** Access does not authorize copying the data to the study's shared `datasets` Volume: the terms forbid transfer to third parties without the Laboratory's written approval and require access control. Tracked as gate `ub-moji-modal-redistribution` — though with finding 2 standing, a 21.87 GB upload would not unblock anything.

The selected `status.blocker` is `data_permission_blocked`, the earliest blocker that independently prevents the requested pipeline. Blocker 2 is reported separately as gate `ub-moji-consent-withheld-training-data`; it is not hidden behind blocker 1, and unlike blocker 1 it cannot be cleared by an access grant alone.

I verified directly in the Modal `repro-sign` workspace that ub-MOJI is **not** already available to the study: the shared `datasets` Volume holds 18 dataset directories, none of them ub-MOJI, and probes of `ub-moji/`, `ub-MOJI/`, `ubmoji/` and `moji/` all returned "path does not exist". So no pre-existing copy exempts this attempt from the access gate.

A third gate, `modal-wrapper-token-info-incompatible`, is open but is **not** about this paper: it records a defect in the repository's own Modal wrapper that affects every paper in the study. See Environment and patches.

## Reproduction agents

| Agent ID | Model and version | Agent application | Contribution | Attribution evidence / unknowns |
| --- | --- | --- | --- | --- |
| `opus5-claude-code-setup` | Claude Opus 5 (`claude-opus-5`) | Claude Code (version not exposed to the session) | Assignment establishment, target resolution (Table 2 into 50 targets), source discovery and pinning, data gating, and authoring `reproduction.json` and this report. Executed no training or evaluation run. | Interactive Claude Code session on 2026-09-24 in worktree branch `repro/murai-2025-jp-fingerspelling-hrpro`. The session system prompt attests: "You are powered by the model named Opus 5. The exact model ID is claude-opus-5." `harness_version` is `null` because Claude Code exposes no build identifier to the agent. |

No run has `agent_ids` because no run was executed.

## Scope and target contract

The assignment text is `what_to_reproduce: "Table 2"`. Table 2 (page 6; paper page 4980) is captioned *"Detection performance (mAP %) at multiple tIoU thresholds and their averaged values using different input encoders."* It is a 5×10 grid: five encoder configurations × seven per-threshold mAP columns (tIoU 0.1 through 0.7) plus three averaged columns. All 50 cells are recorded as in-scope targets, since the table's claimed comparison — that VideoMAE v2 + Point-Sup. CL wins at 0.1–0.5 while I3D + Angular wins at 0.3–0.7 and 0.1–0.7, and that the trend reverses at tIoU 0.7 — depends on the full grid, not a convenient row.

Published values, as recorded in `reproduction.json.targets`:

| Encoder | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | (0.1:0.5) | (0.3:0.7) | (0.1:0.7) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| I3D (RGB, Optical Flow) | 58.7 | 58.1 | 57.7 | 57.3 | 56.2 | 46.3 | 36.6 | 57.6 | 50.8 | 52.9 |
| I3D (RGB, Optical Flow) + Angular | 94.2 | 93.4 | 92.6 | 89.9 | 84.1 | 77.0 | 57.0 | 90.8 | 80.4 | 84.0 |
| VideoMAE v2 (RGB) | 64.3 | 63.8 | 63.2 | 62.8 | 60.2 | 50.9 | 45.2 | 62.9 | 56.5 | 58.6 |
| VideoMAE v2 (RGB) + Point-Sup. CL | 95.3 | 95.3 | 95.3 | 93.6 | 87.3 | 71.4 | 47.1 | 93.4 | 78.9 | 83.6 |
| VideoMAE v2 (RGB) + Point-Sup. CL + Angular | 94.3 | 93.9 | 92.1 | 89.2 | 85.3 | 75.6 | 56.1 | 90.9 | 79.6 | 83.7 |

**Metric.** The user-supplied export identifies the metric only by the opaque ID `yuirj0rm9qhv5ps` named "mAP". That does not establish semantics, so the definition was read from the paper and the pinned evaluation code (`hrpro/eval/eval_detection.py`): per-class Average Precision from predicted temporal segments matched to ground truth at a given tIoU, averaged over the 46 syllable classes in `hrpro/options.py`. The three averaged columns are arithmetic means over the thresholds in each range, on the `np.linspace(0.1, 0.7, 7)` grid declared in `hrpro/cfgs/ub-moji_hyp.yaml`. Higher is better; percent, one decimal place.

**Split, and a naming caution.** Section 5.1: 264 untrimmed videos, of which 24 Dataset A videos are held out and the remaining 240 used for training.

The paper calls the held-out partition **"validation"** and never uses the word "test" for a data split anywhere in the text. The published code calls the *same* partition phase `test`, loads it from `split_test.txt`, and maps it to the `gt_full.json` subset string `"Test"` in `hrpro/log.py` (`mapping_subset = {'ub-moji': {'train': 'Validation', 'test': 'Test'}}`). There is **one** held-out partition of 24 videos, not two, and no separate untouched test set exists in the paper, the code, or the released artifacts.

This matters for interpreting Table 2: `hrpro/main.py` saves the best checkpoint by mAP measured on this same partition, so the paper both selects on and reports on it. The published numbers are best-epoch scores on the selection set. That is the published protocol and is preserved as-is here, not corrected.

Where the distinction matters below, this report writes "held-out split (paper: validation; code phase: test)". The 24 video IDs — and every coverage count — are unaffected by the naming.

**Checkpoint selection.** `hrpro/main.py` saves a checkpoint whenever periodic evaluation on the held-out partition exceeds the running best (stage 1 every 50 steps over 5000; stage 2 every epoch over 20). Selection and reporting therefore happen on the same 24 videos — see the naming caution above.

**Copied baselines.** None. Section 5.2 presents all five rows as the paper's own ablation ("We evaluated five model configurations to investigate the effectiveness of each component"), on its own splits, citing no external source for any row. The export independently records `copied_scores: "no"`.

No target gate was needed: Table 2's identity, rows, columns, and metric are unambiguous in the paper.

## Source provenance

| Artifact | Canonical source | Pinned revision / SHA-256 | Role |
| --- | --- | --- | --- |
| Paper PDF | https://openaccess.thecvf.com/content/ICCV2025W/MSLR/papers/Murai_Point-Supervised_Japanese_Fingerspelling_Localization_via_HR-Pro_and_Contrastive_Learning_ICCVW_2025_paper.pdf | `1f857eef6b19ff838f48a55f4bcdbf266fe46149530e0a50f54ef1f8459c78f1` | Target and protocol. Open-access CVF version of doi:10.1109/ICCVW69036.2025.00516. |
| Published code | https://github.com/tpu-kanglabs/ub-hrpro | `b660ca9c72fb31a0208f7525dc023e6acdb59b6e` | Official implementation, named in the paper's abstract. MIT licensed. Selected executable artifact. |
| I3D feature extractor (submodule) | https://github.com/v-iashin/video_features | `a2f61b7a4cf0ca6a2d91dcc2182f57e7cfd12664` | Pinned at `feature_extraction/video_features`; supplies I3D RGB + optical-flow features for the two I3D rows. |
| Released checkpoints and run config | https://huggingface.co/kanglabs/ub-hrpro | `63f573c687fa44b68c5ea067bb6ee6c90f2691df` | MIT, ungated, 1.25 GB: `videomae/model.pth` (1182 MB), stage-1 `model1_seed_0.pkl` (42 MB), stage-2 `model2_seed_0.pkl` (11 MB), plus `config.json` and `proposal.json`. Covers **one** Table 2 row; see below. Only the two text files were downloaded. |
| ub-MOJI dataset | https://huggingface.co/datasets/kanglabs/ub-MOJI | `82d24aa445e8bc739aff6315ee7bbc6766cf3ea6` | Evaluation data. Gated; see Data provenance. |
| VideoMAE v2 init weights | https://huggingface.co/OpenGVLab/VideoMAE2 | not pinned | `vit_b_k710_dl_from_giant.pth`, required by `videomae/README.md` as the contrastive-pretraining initialization. Ungated. Not downloaded, because no run reached execution; must be pinned before any retained run. |
| MediaPipe hand landmarker | https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker | not pinned | `hand_landmarker.task`, required for the 20-dimensional joint-angle features. Not downloaded for the same reason. |
| Upstream HR-Pro | https://github.com/pipixin321/HR-Pro | not pinned | Considered, not selected. The method this work forks; the authors' own fork is authoritative for Table 2. |

Searches performed on 2026-09-24: the DOI landing page and the open-access CVF proceedings; the paper's own abstract code link; the GitHub org `tpu-kanglabs`; the Hugging Face org `kanglabs` for both dataset and model repositories; the repository's submodules, branches, and commit history. Code discovery was straightforward — the paper states *"Code is available at: https://github.com/tpu-kanglabs/ub-hrpro"* — so no exhaustive no-code search was required.

**What the released checkpoints cover — and why they are not a shortcut.** The checkpoint repository is small and fully downloadable, so it is worth being precise about what it does and does not give us.

Its stage-1 `config.json` records `backbone: videoMAE`, `feature_dim: 768`, and `RAB_args.num_heads: 8`. By the Section 5.1 head-count rule — 8 heads without joint angles, 2 with — that identifies exactly one Table 2 row: **VideoMAE v2 (RGB) + Point-Sup. CL**, seed 0. The separately released contrastively pretrained `videomae/model.pth` corroborates it. No weights are published for the two I3D rows, the plain VideoMAE v2 row, or the full + Angular row, so **40 of the 50 targets have no released checkpoint at all**.

For the 10 targets it does cover, the checkpoint still cannot be evaluated. `I_test` in `hrpro/test.py` builds its dataset through `hrpro/dataset.py`, whose `__getitem__` loads per-video features with `np.load` and runs `process_feat` on them in stage 2 as well as stage 1. The released `proposal.json` supplies only the `PP` proposal boxes — not the features. The loader also needs `split_test.txt`, `gt_full.json`, and `point_gaussian.csv`. None of those four inputs is published anywhere, and all derive from the ub-MOJI videos; regenerating the features needs the 24 test videos, of which only 20 exist in any release.

So the checkpoints are genuinely useful — as protocol evidence, and as a head start if the data gate is ever resolved — but they do not offer an evaluation-only route around it.

**What the published artifacts do not include.** `hrpro/dataset.py` reads `split_{train,test}.txt`, `gt_full.json`, `point_labels/point_gaussian.csv`, and per-video `.npy` feature files under `features/{phase}/`, named by video ID. None of these are published in either the code or the checkpoint repository — `hrpro/dataset/ub-moji/README.md` and `.../point_labels/README.md` document their formats but ship no data. All of them derive from the gated ub-MOJI videos. The released checkpoints therefore cannot produce Table 2 on their own, even though they are ungated.

## Results

All 50 targets are in-scope, none produced. Terminal reason for every target: `data_permission_blocked`, referencing gates `ub-moji-gated-access` and `ub-moji-consent-withheld-training-data`.

| Target IDs | Paper location | System | Dataset/split | Metric + version | Original | Reproduced | Difference | Terminal reason / evidence |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `t2-i3d-*` (10) | Table 2 row 1 | I3D (RGB + Optical Flow) | ub-MOJI, 24 held-out Dataset A videos | mAP@tIoU, `eval_detection.py` @ `b660ca9` | 58.7 / 58.1 / 57.7 / 57.3 / 56.2 / 46.3 / 36.6 / 57.6 / 50.8 / 52.9 | — | — | `data_permission_blocked` |
| `t2-i3dang-*` (10) | Table 2 row 2 | I3D + Angular | same | same | 94.2 / 93.4 / 92.6 / 89.9 / 84.1 / 77.0 / 57.0 / 90.8 / 80.4 / 84.0 | — | — | `data_permission_blocked` |
| `t2-vmae-*` (10) | Table 2 row 3 | VideoMAE v2 (RGB) | same | same | 64.3 / 63.8 / 63.2 / 62.8 / 60.2 / 50.9 / 45.2 / 62.9 / 56.5 / 58.6 | — | — | `data_permission_blocked` |
| `t2-vmaecl-*` (10) | Table 2 row 4 | VideoMAE v2 + Point-Sup. CL | same | same | 95.3 / 95.3 / 95.3 / 93.6 / 87.3 / 71.4 / 47.1 / 93.4 / 78.9 / 83.6 | — | — | `data_permission_blocked` |
| `t2-vmaeclang-*` (10) | Table 2 row 5 | VideoMAE v2 + Point-Sup. CL + Angular | same | same | 94.3 / 93.9 / 92.1 / 89.2 / 85.3 / 75.6 / 56.1 / 90.9 / 79.6 / 83.7 | — | — | `data_permission_blocked` |

Per-column ordering in each cell is 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, (0.1:0.5), (0.3:0.7), (0.1:0.7); individual `target_id`s use those suffixes (`-tiou0.1` … `-tiou0.7`, `-avg0.1-0.5`, `-avg0.3-0.7`, `-avg0.1-0.7`). No original score was copied from earlier work, so no copied baseline needed separate treatment.

Nothing here is a judgment about the paper's science. The published numbers were not tested; they were not reachable from public artifacts under the dataset's terms.

## How to repeat this

This attempt executed no experiment. The commands below are the published recipe, recorded so a reviewer can resume once the data gates are resolved. They are **not** verified against a retained run.

Get the sources:

```bash
git clone https://github.com/tpu-kanglabs/ub-hrpro.git && cd ub-hrpro && git checkout b660ca9c72fb31a0208f7525dc023e6acdb59b6e
```

The `feature_extraction/video_features` submodule is declared with an SSH URL (`git@github.com:v-iashin/video_features.git`); anonymous checkouts need an HTTPS substitution before `git submodule update --init`, pinning commit `a2f61b7a4cf0ca6a2d91dcc2182f57e7cfd12664`.

Then, per the published READMEs: place `gt_full.json`, `split_train.txt`, `split_test.txt` under `hrpro/dataset/ub-moji/`, `point_gaussian.csv` under `hrpro/dataset/ub-moji/point_labels/`, and per-video features under `hrpro/dataset/ub-moji/features/{train,test}/`. Contrastive pretraining and feature extraction (`videomae/`):

```bash
uv run main.py --data_path "path/to/video" --ann_path "path/to/gt_full.json" --ckpt "vit_b_k710_dl_from_giant.pth" --skip_list "path/to/split_test.txt" --epochs 30
```

```bash
uv run extract_tad_feature.py --data_path "path/to/video" --ckpt_path "path/to/checkpoint"
```

Joint-angle features (`feature_extraction/angle_features/`):

```bash
uv run main.py --video_dir /path/to/videos --model_path /path/to/hand_landmarker.task --output_dir ./output --pool --segment_length 16 --stride 16
```

Two-stage localization and evaluation (`hrpro/`), which emits the Table 2 mAP values:

```bash
uv run main.py --cfg ub-moji --stage 1 --mode train && uv run main.py --cfg ub-moji --stage 1 --mode test && uv run main.py --cfg ub-moji --stage 2 --mode train && uv run main.py --cfg ub-moji --stage 2 --mode test
```

Set `feature_dim` in `hrpro/cfgs/ub-moji_hyp.yaml` per row (I3D RGB+Flow 2048, VideoMAE 768, Angle 20) and `RAB_args.num_heads` to 8 without angle features or 2 with them, per Section 5.1.

Before any retained run, the outstanding work is: pin the VideoMAE v2 and MediaPipe artifact revisions; add a `Dockerfile` and a Modal entry point mounting `datasets` read-only at `/datasets` and `huggingface-cache` read-write at `/cache/huggingface` with `HF_HOME` and `HF_HUB_CACHE` set; and run a representative preflight. None of that was built, because it would be scaffolding for a run that cannot lawfully start.

## Data provenance and permissions

| Dataset | Version/subset/splits | Source and access date | License/permission and cloud-use basis | Path in Volume `datasets` | Counts / manifest / checksum | Deviations |
| --- | --- | --- | --- | --- | --- | --- |
| ub-MOJI | Current `main` = `82d24aa445e8bc739aff6315ee7bbc6766cf3ea6` = tag `v26.09`, dated 2026-09-13 — **after** the October 2025 paper. Four tags exist (v25.05, v25.07, v25.09, v26.09); none contains the paper's splits. Dataset A (2 instructors, 1920×1080 @ 60 fps, 46 syllables) and the public part of Dataset B (13 signers, 1920×1080 @ 30 fps, 76 sign classes). Split files `split_train.txt` / `split_test.txt` are **not** published. | https://huggingface.co/datasets/kanglabs/ub-MOJI, metadata accessed 2026-09-24; content **not** accessed. | Gated `auto`. Terms **accepted by the user on 2026-09-24**; authenticated read access verified. Academic research only; no redistribution, transfer, or sublicensing without the Laboratory's prior written approval; strict access control to authorized research personnel; governed by Japanese law. Cloud-processing basis still **unresolved** — acceptance does not settle whether the data may be copied to the shared Modal Volume (gate `ub-moji-modal-redistribution`). | **Not populated, verified absent.** `modal volume ls datasets /` in workspace `repro-sign` on 2026-09-24 returned 18 dataset directories, none of them ub-MOJI; probes of `ub-moji/`, `ub-MOJI/`, `ubmoji/`, `moji/` each returned "path does not exist". The `huggingface-cache` Volume holds no `kanglabs` or ub-MOJI entry either. | Paper reports 264 videos (240 train / 24 held out); `proposal.json` enumerates 235 + 24 = 259. Usable in the best release (video file **and** annotation entry): **19 of 24** test, **151 of 235** train. Only `annotations.toml` was downloaded (sha256 `dbe43236eca227ed8b77f7da13a3cdbd6e291434ec8ae3c4fcca95162b2c25c4`); no video was fetched. | Not acquired — the release cannot supply the reported splits. No substitute dataset was used or considered. |

`reproduction.json.datasets` is deliberately empty: the record's dataset contract requires per-split file paths with SHA-256 checksums, and this attempt holds no file from ub-MOJI to checksum. Recording a dataset entry would mean inventing provenance for data never obtained. The full provenance and permission analysis lives in gate `ub-moji-gated-access` and in this section.

**Ethics and consent.** The export records `potential_ethical_concerns: "no"` and `includes_human_evaluation: "no"`. This attempt involved no participants and no human evaluation, which is consistent with the second flag. The first flag did not survive verification as a clean "no": the paper itself documents that part of its training data is withheld under participant consent agreements, and the dataset terms prohibit identifying individuals or associating data with specific individuals. That does not block the *public* subset, whose terms clearly permit academic research use once accepted — but it does mean any route to the paper's full training set runs through a consent question, not merely an access question. That is recorded in gate `ub-moji-consent-withheld-training-data` and must go to Team S with ethics review, not to a direct author request for the withheld videos.

## Environment and patches

No container was built, no dependency resolved, and no image digest exists, because no run reached execution.

The upstream environment is already well pinned by the authors and is recorded here for the resumed attempt: Python ≥3.11 (`.python-version` 3.11) with `uv` lockfiles in each of `hrpro/`, `videomae/`, and `feature_extraction/angle_features/`; `torch==2.6.0` / `torchvision==0.21.0` from the `cu124` index; `decord==0.6.0`, `timm==0.4.12`, `deepspeed==0.17.1`, `schedulefree==1.4.1` (VideoMAE stage); `mediapipe==0.10.21` (angle stage); CUDA ≥12.4 and OpenCV per the root README.

**Repository tooling defect found.** `.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh` verifies credentials with `modal token info`, which Modal client 1.2.6 no longer provides ("No such command 'info'"). The wrapper reports that as expired credentials and directs the agent to a `modal setup` human gate that is not actually needed. Since the wrapper is the mandated path for all Modal operations, this blocks wrapper-mediated Modal work for every paper in the study on this CLI version. Recorded as gate `modal-wrapper-token-info-incompatible` for the maintainers; a workspace-verification check that exists in current clients (parsing `modal profile list`, or `modal profile current` plus a cheap authenticated call) would fix it. The ub-MOJI checks in this report were made with direct `modal volume ls` calls under `MODAL_PROFILE=repro-sign`, in the required workspace, with no fallback profile.

**Patches to the published code: none.** The published code is at preference level 1 and needed no correctness patch, because no execution was attempted. One adaptation is known to be required if work resumes — the `video_features` submodule's SSH URL blocks anonymous cloning — but it was not applied or tested here, so it is recorded as future work rather than as a retained patch.

## Execution evidence

No run was executed, so `reproduction.json.runs` and `reproduction.json.artifacts` are empty and no stop policy was declared. The investigation was read-only: fetching the open-access PDF, cloning the pinned repository, reading published READMEs, configs, and the data loader, downloading the ungated `config.json` and `proposal.json`, and probing dataset access (HTTP 401 on every content file).

Modal was used read-only, in workspace `repro-sign`, to verify dataset availability. `modal volume list` returned the workspace's 15 Volumes including both canonical v2 Volumes; `modal volume ls datasets /` returned 18 dataset directories (LSA64, WLASL, WSLP-AACL-2025, arab-sign, arasl-database-grayscale, asl-alphabet, asl-citizen, aslg-pc12, csl-daily, dgs-corpus, dgs3-t, drive-unnamed-1z5f0h0ibdphpl0, hgr1, indian-sign-language-images, isl-hs, lsa64, rwth-phoenix-2014, rwth-phoenix-2014-t), none of them ub-MOJI; probes of the four plausible ub-MOJI slugs each returned "path does not exist".

No Modal app, function call, or GPU time was consumed, and no cost was incurred.

## Guesses and deviations

No protocol guess was applied to any run, because no run happened. The unresolved details below would each require a decision before a resumed attempt, and are recorded now so they are not silently filled in later.

| Detail | Paper/evidence says | This attempt used | Rationale | Effect on interpretation |
| --- | --- | --- | --- | --- |
| Training-set size | Section 5.1: 240 training videos of 264 total | Nothing — not resolved | `proposal.json` lists 235 train IDs (259 total); the best public release supplies only 152 of those 235 | Training on the public subset would change the training distribution, making any resulting score conditional evidence rather than a produced Table 2 value |
| Split membership | 24 Dataset A videos held out (paper: "validation"; code phase: `test`); `split_test.txt` not published | Nothing — not resolved | The 24 test IDs are recoverable from `proposal.json`, but only 19 of those videos are usable in any public release (and the same 4 are absent from every version) | Evaluating on 19 of 24 changes the evaluation set; any resulting mAP is a different measurement, not a reproduction of the published value |
| Point annotations | `point_gaussian.csv` format documented in `hrpro/dataset/ub-moji/point_labels/README.md`; file not published | Nothing — not resolved | Point labels would have to be derived from the dataset's own annotations; the derivation is not published | Any derived point labels are an inferred detail affecting all point-supervised rows |
| Contrastive pretraining length | Section 5.1 gives σ=4, memory bank 65,536, Schedule-Free RAdam lr 1e-4, τ=0.1, but no epoch count | Nothing — not resolved | `videomae/README.md` shows `--epochs 30` as an example, which the paper does not confirm | Epoch count and pretraining checkpoint-selection rule are unstated in the paper and would be a guess |
| Checkpoint selection split | Section 5.1 calls the 24 videos "validation"; Table 2 reports on them; the code calls the same partition `test` | Nothing — not resolved | `hrpro/main.py` selects the best checkpoint by mAP on the same partition it reports; no separate test set exists | Table 2 values are best-epoch scores on the selection set. Published protocol; preserved and noted, not corrected |
| Seeds | Paper states no seed policy; released config and checkpoints use `seed 0` | Nothing — not resolved | The paper reports single values with no mean/std, implying one run per configuration | A resumed attempt should use seed 0 and report single values, matching the published presentation |

## Attempts, failures, and dead ends

1. **Hypothesis: the queue export can be ingested as a `queue_record`.** Test: inspected the export's shape and required fields. Result: it is a single JSON object, not the top-level array the ingestion contract requires, and it has no `confirmation` field — `ingest_candidate.py` rejects it on both counts. Kept: recorded the assignment as `direct_user_request` with the export preserved verbatim under `assignment.queue_record_evidence`, rather than fabricating `confirmation: confirmed`. The record's `status: final`, its `status_history` entry, and `finalized_by` are reported as the evidence that exists.
2. **Hypothesis: official code exists despite `compute_requirements: N/A`.** Test: followed the export's `code_repos` link and the paper's abstract. Result: `tpu-kanglabs/ub-hrpro` is the official MIT-licensed implementation with all three pipeline stages and `uv` lockfiles. Kept: preference level 1, no patch.
3. **Hypothesis: the released checkpoints allow evaluation without the raw videos.** Test: enumerated `kanglabs/ub-hrpro` (ungated) and read `hrpro/dataset.py`. Result: the loader requires per-video `.npy` features, `gt_full.json`, `split_*.txt`, and `point_gaussian.csv`; none are published and all derive from the gated videos. Dead end — evaluation-only is not a route around the data gate.
4. **Hypothesis: ub-MOJI is anonymously downloadable, as the export's `available: "yes"` suggests.** Test: `curl` against four content paths plus `README.md` and `LICENSE.md`. Result: HTTP 401 on all four content files; 200 on the two metadata files. The queue lead did not survive verification. Gate `ub-moji-gated-access` opened.
5. **Hypothesis: accepting the gate would unblock Table 2.** Test: read the paper's dataset section and cross-checked the authors' released `proposal.json` against Section 5.1's counts. Result: 235 train IDs vs 240 reported, and Section 3.2 explicitly withholds Dataset B Session 1 for participant consent. An access grant alone is insufficient. Gate `ub-moji-consent-withheld-training-data` opened as an independent blocker.
6. **Hypothesis: the shared `datasets` Volume may already hold ub-MOJI.** Test: `modal_repro_sign.sh volume list`. Result: the wrapper aborted with "Modal credentials for 'repro-sign' are invalid or expired." First read as an auth gate and not retried.
7. **Hypothesis (after the user offered to re-authenticate): the credentials are genuinely expired.** Test: inspected the wrapper's preflight and ran its checks individually. Result: the diagnosis was wrong. The credentials are valid — `modal profile list` shows `repro-sign` active on workspace `repro-sign`, and `modal volume list` succeeds. The wrapper fails because it gates on `modal token info`, a subcommand removed by Modal client 1.2.6, and reports the missing subcommand as a credential failure. No `modal setup` was needed. Gate `modal-repro-sign-auth` was withdrawn and replaced by `modal-wrapper-token-info-incompatible`, which is a study-infrastructure defect rather than a blocker for this paper.
8. **Hypothesis: ub-MOJI may already be on the shared `datasets` Volume, which would change the access analysis.** Test: `modal volume ls datasets /` plus direct probes of four plausible slugs, and a search of `huggingface-cache`, all under `MODAL_PROFILE=repro-sign`. Result: absent everywhere. The access gate stood on verified fact rather than on an unknown.
9. **Hypothesis (after the user accepted the terms): with access granted, the dataset can be downloaded and the reproduction can proceed.** Test: verified authenticated access (HTTP 200 on previously-401 files), then — before starting a 21.87 GB transfer — enumerated every file at all four published tags and matched basenames against the 259 video IDs in the authors' released `proposal.json`, checking both `sequences/` and `words/` and substring-searching unmatched IDs across all paths. Result: no release contains the paper's splits; best coverage 172/259 overall, with the same four test videos missing from every tag and six paper-used participant IDs absent everywhere. The download was **not** started, because it could not produce the requested targets.
10. **Hypothesis (challenged on how the missing test data was known): the file-listing method may be wrong, or `proposal.json` may not be the reported split.** Test: re-derived the split identity from `proposal.json` structure (24 IDs, 2 participants, 1 session — matching Section 5.1), added present/absent controls to the substring search, and cross-checked against the dataset's own `annotations.toml`, which is keyed by video ID and independent of filenames. Result: the finding held and tightened. The annotation index flagged one video the file listing could not — `kamakura_019_202310_t001` has an `.mp4` but no annotations — so the usable test count is **19 of 24**, not 20. The earlier "20 of 24" figure is corrected throughout.

No build, training, OOM, or interrupted run occurred, and no speculative change was made and reverted.

## Candidate flags, ethics, and human evaluation

- `copied_scores: "no"` — verified. Section 5.2 presents all five rows as the paper's own ablation with no external citation; all 50 targets carry `copied_baseline: false`.
- `includes_human_evaluation: "no"` — consistent. Table 2 is an automatic localization metric; no participant interaction is involved in reproducing it.
- `potential_ethical_concerns: "no"` — did not fully survive verification. The dataset is human video data whose terms forbid identifying individuals, and the paper documents training data withheld under participant consent agreements. The public subset's terms permit academic research use once accepted, so this does not block work on public data; but any route to the withheld Session 1 data requires ethics and consent review rather than a simple access request. See gate `ub-moji-consent-withheld-training-data`.
- `compute_requirements: "N/A"` — no compute estimate was produced, because no preflight ran. The compute gate was never reached.
- `comments` and `flag_reason` are empty in the export.

## Author and team contact

None. No author was contacted, which is correct at this stage: author contact is a gated post-independent-attempt action, and this attempt has not run independently. The two data gates route to Team S for dataset access coordination. Any future approach to the consent-withheld Session 1 data must go through Team S with ethics review, not directly to the authors.
