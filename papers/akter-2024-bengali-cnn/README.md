# Automated Bengali Sign Language Character Classification with Deep Learning Techniques

**Pipeline status:** blocked_on_data

**Numerical agreement:** not_assessed

**Preference level:** 3

The independent attempt is **blocked on exact data**. None of the 24 requested Table I values was produced; numerical agreement is not assessed. The paper's actual Kaggle footnote distributes only 36 character samples, duplicated in a repository snapshot, while its experiments require a manually selected and augmented 7,200/2,160 train/test set. Training on those samples would change the experiment.

Tahmina Akter, Tanjim Mahmud, Tikle Barua, Sultana Rokeya Naher, Mohammad Shahadat Hossain, and Karl Andersson (2024), ICCCNT, pp.1–6. [Publisher DOI](https://doi.org/10.1109/ICCCNT61001.2024.10724561). The authorized publisher PDF was read in full and is not redistributed. Its hash and the original tracker export hash are recorded in `reproduction.json`. The current tracker record is a direct user-authorized assignment; it has final status but no legacy confirmation property. No queue fields were fabricated.

GPT-6 using Codex performed source discovery, the data audit, protocol analysis, and reporting. The session exposes no exact model variant or harness version; these limits are preserved in agent attribution. No other model executed an experiment.

## Requested results

All rows describe the authors' own comparison, not copied baselines. Precision, recall, F1, and accuracy are percentages.

| Table I system | Precision | Recall | F1 | Accuracy | Reproduced |
|---|---:|---:|---:|---:|---|
| CNN | 98 | 98 | 98 | 98 | Not produced |
| VGG16 | 85 | 95 | 95 | 95 | Not produced |
| VGG19 | 94 | 94 | 94 | 94 | Not produced |
| ResNet50 | 88 | 88 | 87 | 88 | Not produced |
| ResNet101 | 89 | 88 | 88 | 88 | Not produced |
| ResNet152 | 89 | 88 | 88 | 88 | Not produced |

Table I is on PDF page 5. Section IV defines the usual metrics but does not state micro/macro/weighted averaging. The VGG16 row is inconsistent under the same conventional averaging: precision 85 and recall 95 have harmonic mean 89.72%, and same-weight average per-class F1 cannot exceed that bound. Even rounding cannot explain F1 95. The CNN classification report is not used to silently replace the requested table.

## Data evidence and source search

The paper's section III.A footnote points to [ksanzid/esharalipi-bangla-sign-language-dataset](https://www.kaggle.com/datasets/ksanzid/esharalipi-bangla-sign-language-dataset). Official v2 metadata states CC BY-NC-SA 4.0 and describes a sample repository. The retrieved archive SHA-256 is `3f683d12523e9aae6413369e338edfae874c43770db512a852bebd76540cc94b`. It contains one image for each of 36 characters, nine digit images, and illustrative figures, each duplicated inside an included repository snapshot. The machine record names a planned canonical directory `datasets/ishara-lipi-selected`, explicitly not populated; its observed split describes these 36 public samples and its expected splits describe the absent target data. Its advertised 1,800 character count describes the complete collection, not the downloadable sample contents.

The [original authors' repository](https://github.com/sanzidikawsar/Bangla-Sign-Language/tree/04feaff177eb4b009b9091a537748029f393b414), pinned at `04feaff177eb4b009b9091a537748029f393b414`, and [original laboratory page](https://diu-nlp-ml.github.io/Dataset/) point to `isharalipi.sanzidscloud.com` for the full dataset. That host does not resolve in the retained audit. The different `mmsabid` URL in reference 16 returns HTTP 403; it was not bypassed.

An independent search also found [cloudy4next/Ishara-Lipi](https://github.com/cloudy4next/Ishara-Lipi/tree/6888dce6c19332a8d282c2b47b59fbf073b25a39), a different 2020 paper. Its pinned archive has SHA-256 `68991ad6be75b42004571b8ccde4f7d6750d3c94c4639bd2d109914e70df2deb` and 1,005 images in 36 classes, ranging from 23 to 36 per class. It does not establish that these are the 2024 authors' manually selected source images, nor supply their augmented train/test split. It was inspected as a lead, not substituted or uploaded to canonical storage.

The shared Modal dataset root was inspected through the required repro-sign wrapper; no Ishara-Lipi directory or exact selected-data manifest was present. The generic drive-import root contained AtoZ_3.1 and alphabets, not an identified Ishara-Lipi release.

Exact-title, author, institutional publication, paper-reference and GitHub searches found no attributable target implementation or supplement. The [coauthor institutional publication page](https://www.uits.ac.bd/public/faculty-profile-cse/38) lists the paper without code. The least-invasive next executable route would therefore be paper reconstruction, once the exact data gate is resolved.

## Repeat the audit

From the repository root, run:

```bash
python3 papers/akter-2024-bengali-cnn/audit.py
python3 .agents/skills/reproduce-paper/scripts/validate_reproduction.py papers/akter-2024-bengali-cnn
```

A small CPU-only container is provided:

```bash
docker build -t repro-akter-audit papers/akter-2024-bengali-cnn
docker run --rm repro-akter-audit
```

The executed command was the native Python audit, not the Docker command: local Docker daemon availability was unnecessary. It uses only the Python standard library, fetches public metadata and ZIPs into memory, and retains no images. `source-audit.txt` is the raw successful audit output, with its hash, timestamps, stop ceiling and run ID `source-audit-1` in `reproduction.json`. Network availability can change; the pinned hashes and retained inventory establish what was actually inspected. GPU-hours and cloud cost were zero. No model training, evaluation, checkpoint, or numerical outcome is claimed. Data/train/evaluation entry points remain gated rather than producing misleading synthetic results.

## Remaining questions

1. Which source images and augmentation procedure generated the exact 7,200 training and 2,160 test images, and do originals overlap across splits? The paper also says noisy images were removed and brightness stabilized without a reproducible rule.
2. What corrects the VGG16 precision/recall/F1 row, and which aggregation was used?

The reported 64×64 resize, division by 255, ImageNet VGG/ResNet families, Nadam optimizer, and sparse categorical cross-entropy are usable recipe details. Equation 1 instead depicts binary cross-entropy; a future implementation would follow the explicit 36-class loss text and disclose this inference. No optimizer or seed uncertainty was treated as a blocker. No authors were contacted, no restricted artifacts were published, and no third-party subset was presented as the requested experiment.

## Approximation reassessment

The 2026-10-01 continuation reviewed whether documented approximation could make an informative full attempt. The paper-linked release supplies one original image per character, rather than a nearly complete collection missing a few examples. Expanding those images to the reported train/test sizes would primarily measure transformations of the same source image. The separate 1,005-image collection has no verified relationship to this paper’s selected corpus or split. Neither is a scientifically meaningful substitute for the requested experiment. The remaining blockers are source-data access/identity and the inconsistent VGG16 metric row; ordinary implementation judgments do not require human action.
