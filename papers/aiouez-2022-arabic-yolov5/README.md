# Real-time Arabic Sign Language Recognition based on YOLOv5

**Pipeline status:** blocked_on_data
**Numerical agreement:** not_assessed
**Preference level:** 1

Aiouez, Hamitouche, Belmadoui, Belattar and Souami. IMPROVE 2022, pp.17–25. [Paper](https://doi.org/10.5220/0010979300003209). Attempt date: 2026-10-01. The tracker assignment requests Tables 3 and 4. All 24 numeric cells, including timing-range endpoints, are accounted for in `reproduction.json`; repeated mAP values remain separate table targets. None are copied literature baselines.

The author-linked Kaggle release is public, but both its unaugmented and augmented metadata explicitly report **Unknown** license. The paper describes availability for interested researchers without specifying cloud-storage/processing permission. No data was downloaded, uploaded or processed, and no GPU run was launched. Team S must establish permission before a representative preflight. This is a documented data-gated attempt, not a completed training reproduction.

There is also a concrete corpus/split conflict: the paper expands **5,600** original images to **15,088**, then randomly splits 80/10/10. The author release instead describes **5,832** unaugmented images (4,651/891/290) and **15,086** augmented images (13,926/870/290). These cannot be assumed to represent the same test set. Its augmentation types match the paper: Gaussian blur up to 1 pixel, salt-and-pepper noise up to 5%, 25% grayscale and rotations ±20°.

| Table 3 model | Precision | Recall | mAP@.5 | mAP@[.5:.95] | Reproduced |
|---|---:|---:|---:|---:|---|
| YOLOv5s | 99.2% | 99.4% | 99.3% | 85.9% | Not produced |
| YOLOv5m | 99.2% | 99.2% | 99.3% | 87.75% | Not produced |
| YOLOv5l | 99.5% | 99.4% | 99.4% | 87.2% | Not produced |

| Table 4 model | Parameters | Inference time | FPS | mAP@.5 | mAP@[.5:.95] |
|---|---:|---:|---:|---:|---:|
| YOLOv5s | 7.5 million | 0.007–0.01 s | 121 | 99.3% | 85.9% |
| Faster R-CNN | 105 million | 0.55 s | 1.8 | 98.7% | 81.38% |

No reproduced values are available for Table 4. Exact GPU and timing boundaries are also unspecified, so timing would need its own comparability review. Table 4 parameter counts cannot be tied to an exact author checkpoint/config because no such artifact was found.

The full paper, publisher/conference pages, references, author ResearchGate profile, exact-title/author GitHub searches and Zenodo/OSF searches were inspected. No paper-specific code, split or trained weights were found. The cited generic implementations are [YOLOv5 v6.0](https://github.com/ultralytics/yolov5/tree/956be8e642b5c10af4a1533e09084ca32ff4f21f) and [Detectron2 v0.6](https://github.com/facebookresearch/detectron2/tree/d1e04565d3bec8719335b88be9e9b961bf3ec464), preserved as historically plausible candidate pins rather than asserted author versions. YOLOv5's train/evaluation entry points were inspected. No reimplementation or upstream patch was made.

The paper specifies SGD and image size 416×416. Table 2 gives YOLOv5l: lr0.01/batch24/50 epochs, YOLOv5m: lr0.01/batch16/50 epochs, YOLOv5s: lr0.015/batch16/60 epochs, and Faster R-CNN X101-FPN: lr0.001/batch24/60 epochs. Weights, remaining augmentation defaults and checkpoint selection still need pinning after the data gate. No training command or container is claimed tested.

The shared `datasets` Volume inventory was inspected through the `repro-sign` wrapper. The intended `belmadoui-arabic-sign-language/` root does not exist. `arab-sign/` contains RGB/Depth/Skeleton directories and has not been established as this corpus. `arasl-database-grayscale/` is a distinct 54,049-image, 32-class dataset and was not substituted. The tracker misleadingly associates the same dataset record with both papers.

Raw public Kaggle metadata is retained in `kaggle-unaugmented-metadata.txt` and `kaggle-augmented-metadata.txt`; SHA-256 hashes and URLs live in `reproduction.json`. Dataset split file entries explicitly refer to metadata containing declared counts, not unseen image checksums. `cloud_processing_allowed: false` means permission is not established, not that a prohibition was proved. The full paper PDF is hashed but not redistributed.

To repeat source/gate checks:

```bash
curl -L 'https://www.kaggle.com/api/v1/datasets/view/sabribelmadoui/arabic-sign-language-unaugmented-dataset'
curl -L 'https://www.kaggle.com/api/v1/datasets/view/sabribelmadoui/arabic-sign-language-augmented-dataset'
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh volume ls datasets / --json
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh volume ls datasets belmadoui-arabic-sign-language/ --json
python3 .agents/skills/reproduce-paper/scripts/validate_reproduction.py papers/aiouez-2022-arabic-yolov5
```

After Team S resolves the license/cloud basis, establish the exact augmented corpus and split, then implement an idempotent population command and verify real annotations before preflight. No author contact was made. The paper's ethics flag was investigated as existing human hand-image data; no new participants or webcam collection were introduced.

GPT-6 using Codex performed this source/data investigation and report. Session instructions establish those names; exact model ID and harness version were unavailable. The tracker export is preserved with operator emails redacted and the original source hash retained. Its database IDs differ from paper ID and it has no confirmation field, so this is a direct user-authorized tracker assignment with no invented queue confirmation.
