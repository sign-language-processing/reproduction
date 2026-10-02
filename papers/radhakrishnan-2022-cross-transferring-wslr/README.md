# Radhakrishnan et al. 2022: action-recognition models on MS-ASL (not reproduced)

**Paper ID:** `18b8675237185c3272f727b5c9e6a63db15833a4`

**Citation:** S. Radhakrishnan, N. C. Mohan, M. Varma, J. Varma, and S. N. Pai, "Cross Transferring Activity Recognition to Word Level Sign Language Detection," in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2022, pp. 2445-2452, doi: 10.1109/CVPRW56347.2022.00273.

**Paper:** https://doi.org/10.1109/CVPRW56347.2022.00273 ([CVF open access](https://openaccess.thecvf.com/content/CVPR2022W/ABAW/papers/Radhakrishnan_Cross_Transferring_Activity_Recognition_to_Word_Level_Sign_Language_Detection_CVPRW_2022_paper.pdf)) · **Code/artifacts:** none found after search

**Preference level:** 3

**Pipeline status:** `blocked_on_data`

**Numerical agreement:** `not_assessed` — no experiment was run and no target produced a value.

**Attempt date:** 2026-10-02

## Summary

The paper's Table 2 was measured on the authors' own hand-curated, hand-trimmed 50-class subset of MS-ASL with a random 85/15 split. That set is unpublished and cannot be rebuilt from public MS-ASL: a third of the source videos for the 50 most frequent classes are no longer public. The assignee decided on 2026-10-02 to record all 24 targets as not produced (`data_unavailable`) without running experiments. Preference level 3 is provisional: nothing was implemented.

## Reproduction agents

| Agent ID | Model and version | Agent application | Contribution | Attribution evidence / unknowns |
| --- | --- | --- | --- | --- |
| `claude-code-fable-5-1` | Claude Fable 5.1 (`claude-fable-5-1`) | Claude Code 2.1.267 (Claude desktop app, Code tab) | Whole attempt: record retrieval, target contract, source search, Modal inspection, MS-ASL annotation download, partial video acquisition, availability checks, this report. No training or evaluation. | Session system metadata names the model and ID; version from `claude --version`. The desktop app build version was not exposed. |

There are no runs, so no run attribution.

## Scope and target contract

The survey-tool record (status `final`, finalized 2026-09-08) asks for **Table 2**. Table 2 (p. 2449) reports top-1, top-3 and top-5 test accuracy for eight Kinetics-400-pretrained action-recognition models fine-tuned on the authors' MS-ASL subset: 24 targets, all in scope, none copied from earlier work. The record's `copied_scores: yes` refers to Table 3 (MS-ASL paper baselines), which is not requested.

The export has no `confirmation` field, so the assignment is recorded as a direct request with the email-redacted record attached rather than as ingested queue output.

Two printed values look irregular and are transcribed as printed: P3D has the same top-3 and top-5 (95.75), and R(2+1)D's top-3 (91.31) is only 2.93 points above its top-1.

Metric: accuracy over the 15% test split, 50 classes. The paper does not state the implementation, the clips or crops scored per test video, the checkpoint rule, or seeds, and describes no validation split.

## Source provenance

| Artifact | Canonical source | Pinned revision / SHA-256 | Role |
| --- | --- | --- | --- |
| Paper PDF | CVF open access (link above) | `5eb47fbc642a8b5d956d80e284a0d7903c5d2998f3f5a970b3cccab8ffe54e24` | Target and protocol |
| Published code | none found | — | — |
| Weights/configs/supplements | none found | — | — |
| MS-ASL annotations | https://download.microsoft.com/download/3/c/a/3ca92c78-1c4a-4a91-a7ee-6980c1d242ec/MS-ASL.zip | `a8562008309eea4129e1bc0ed7f654a314fee195227222859657e307b6434c34` | Parent dataset annotations (C-UDA 0.1), used only for the availability checks |

Searched on 2026-10-02: the full PDF (no links, footnotes or availability statement); web search for the exact title; GitHub repository search for the title phrase (0 results) and for "MS-ASL slowfast"; the first author's GitHub account (20 public repositories, none on sign language or video recognition). The CVF workshop listing and the paper's landing page give a PDF link only and no supplemental link, although other papers on the same listing have one. Not checked: IEEE Xplore's page for attached media, Zenodo/OSF, and the other authors' accounts.

The queue record's Zenodo link (record 6674324, `MS-ASL.zip`) is OpenHands pose keypoints, not video, and was not downloaded.

Guess, not verified: the eight systems match the GluonCV Kinetics-400 model zoo by name (including R(2+1)D at 112×112). The paper names no framework.

## Results

| Target ID | System | Metric | Original | Reproduced | Difference | Terminal reason |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `table2-p3d-top1` | P3D | top1 | 85.55 | — | — | `data_unavailable` |
| `table2-p3d-top3` | P3D | top3 | 95.75 | — | — | `data_unavailable` |
| `table2-p3d-top5` | P3D | top5 | 95.75 | — | — | `data_unavailable` |
| `table2-r2plus1d-top1` | R(2+1)D | top1 | 88.38 | — | — | `data_unavailable` |
| `table2-r2plus1d-top3` | R(2+1)D | top3 | 91.31 | — | — | `data_unavailable` |
| `table2-r2plus1d-top5` | R(2+1)D | top5 | 96.88 | — | — | `data_unavailable` |
| `table2-i3d-inceptionv3-top1` | I3D InceptionV3 | top1 | 85.83 | — | — | `data_unavailable` |
| `table2-i3d-inceptionv3-top3` | I3D InceptionV3 | top3 | 95.06 | — | — | `data_unavailable` |
| `table2-i3d-inceptionv3-top5` | I3D InceptionV3 | top5 | 96.05 | — | — | `data_unavailable` |
| `table2-i3d-resnet50-top1` | I3D ResNet 50 | top1 | 84.70 | — | — | `data_unavailable` |
| `table2-i3d-resnet50-top3` | I3D ResNet 50 | top3 | 96.60 | — | — | `data_unavailable` |
| `table2-i3d-resnet50-top5` | I3D ResNet 50 | top5 | 97.45 | — | — | `data_unavailable` |
| `table2-i3d-resnet101-top1` | I3D ResNet 101 | top1 | 88.10 | — | — | `data_unavailable` |
| `table2-i3d-resnet101-top3` | I3D ResNet 101 | top3 | 97.16 | — | — | `data_unavailable` |
| `table2-i3d-resnet101-top5` | I3D ResNet 101 | top5 | 98.58 | — | — | `data_unavailable` |
| `table2-slowfast-4x16-resnet50-top1` | SlowFast 4x16 ResNet 50 | top1 | 89.52 | — | — | `data_unavailable` |
| `table2-slowfast-4x16-resnet50-top3` | SlowFast 4x16 ResNet 50 | top3 | 95.18 | — | — | `data_unavailable` |
| `table2-slowfast-4x16-resnet50-top5` | SlowFast 4x16 ResNet 50 | top5 | 97.73 | — | — | `data_unavailable` |
| `table2-slowfast-8x8-resnet50-top1` | SlowFast 8x8 ResNet 50 | top1 | 90.37 | — | — | `data_unavailable` |
| `table2-slowfast-8x8-resnet50-top3` | SlowFast 8x8 ResNet 50 | top3 | 96.60 | — | — | `data_unavailable` |
| `table2-slowfast-8x8-resnet50-top5` | SlowFast 8x8 ResNet 50 | top5 | 98.58 | — | — | `data_unavailable` |
| `table2-slowfast-8x8-resnet101-top1` | SlowFast 8x8 ResNet 101 | top1 | 92.35 | — | — | `data_unavailable` |
| `table2-slowfast-8x8-resnet101-top3` | SlowFast 8x8 ResNet 101 | top3 | 97.73 | — | — | `data_unavailable` |
| `table2-slowfast-8x8-resnet101-top5` | SlowFast 8x8 ResNet 101 | top5 | 98.87 | — | — | `data_unavailable` |

All rows are Table 2, p. 2449, on the authors' unpublished 15% test split. None is a copied baseline.

Selected blocker: `data_unavailable`. It is the earliest blocker and independently prevents every target. A second, later blocker is recorded separately: the training recipe is under-specified (hyperparameters tuned per model "until we achieved optimal performance", ranges only, no validation split).

## Why the data cannot be recovered

Evidence gathered on 2026-10-02, all in gate `data-authors-curated-subset`:

- **Paper (Sec. 4.1):** the 50 most frequent classes; videos "manually downloaded, screened, and trimmed"; the set "deviated from the original MS-ASL dataset" because many links were dead; 48 videos per class on average, SD 6; random 85/15 split. No clip list, trims, split or seed is published.
- **Split:** unrecoverable even with every video in hand. The paper does not use MS-ASL's official train/val/test files; it draws its own 85/15 split with no file list, seed, or stratification rule, over a manually screened pool that is itself unknown. No supplementary material exists that lists the videos.
- **Modal:** no MS-ASL directory exists in Volume `datasets` (26 directories listed).
- **Top-50 classes:** from the annotations, labels 0–49 hold 3,191 clips over 1,109 videos. The cut is tied at 56 clips between "bored", "water", "computer", "boy" and "help" (label 50), so the class set itself is ambiguous by one class.
- **Availability:** via YouTube's public oEmbed endpoint, 730 of the 1,109 videos are public, 194 removed, 169 private, 16 embedding-restricted. That leaves 2,208 clips (69.2%).
- **The paper's 48 ± 6:** not matched by any available reading.

| Set (labels 0–49) | Unit | Total | Mean per class | SD |
| --- | --- | ---: | ---: | ---: |
| Full 2019 release | clips | 3,191 | 63.8 | 6.2 |
| Public on 2026-10-02 | clips | 2,208 | 44.2 | 5.6 |
| Public, counting embed-restricted | clips | 2,301 | 46.0 | 5.5 |
| Full 2019 release | distinct source videos | 2,545 | 50.9 | 6.0 |
| Public on 2026-10-02 | distinct source videos | 1,745 | 34.9 | 5.1 |

A set rebuilt today would be a different dataset with a different split, so its numbers would be conditional evidence and could not stand in for Table 2. It was not built.

## How to repeat this

There is no training or evaluation to repeat. To repeat the acquisition attempt:

```bash
pip install "yt-dlp[default]" deno   # plus ffmpeg, or FFMPEG=/path/to/ffmpeg
papers/radhakrishnan-2022-cross-transferring-wslr/data.sh STAGING_DIR
```

`data.sh` downloads and hash-checks the official annotations, fetches each source video at up to 720p, cuts one clip per annotation, and writes `manifest.json` with per-clip status and SHA-256. It is resumable and stops if YouTube asks for bot verification.

## Data provenance and permissions

No dataset was used and nothing was uploaded to Modal; `reproduction.json` lists MS-ASL with intended path `ms-asl`, `populated: false`, and the official split files' hashes. MS-ASL annotations are licensed under Microsoft's C-UDA 0.1; the videos are third-party YouTube uploads. The partial staging tree (annotations, 349 clips from 126 videos) stays on the assignee's institutional storage and is not redistributed.

## Environment and patches

No container was built and no patches exist. Tooling used for the checks: Modal CLI 1.6.0, yt-dlp 2026.08.19 with deno 2.9.7, ffmpeg 7.0.2.

## Execution evidence

No retained runs. Modal operations were read-only (`volume list`, `volume ls datasets /`, `check_modal_dataset.sh ms-asl`).

## Guesses, deviations, dead ends, author contact

- **Guess:** GluonCV model zoo as the weight source (above).
- **Deviations:** none taken.
- **Dead end — Modal preflight:** the wrapper reported expired credentials because Modal CLI 1.2.6 lacks `modal token info`; CLI 1.6.0 fixed it.
- **Dead end — video acquisition:** `data.sh` processed 223 of 7,212 videos (126 downloaded, 97 private or removed), then stopped when YouTube required sign-in bot verification. This was not worked around.
- **Author contact:** none. Requesting the authors' clip list and split through Team S remains the only route to the exact data and was not pursued.
