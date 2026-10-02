# Real-time Arabic Sign Language Recognition based on YOLOv5

**TL;DR:** Reproduction resumed after explicit user authorization for internal use of the public dataset; no new target results yet.
**Decision:** License remains unknown; dataset redistribution is prohibited. No human action is pending.

**Execution state:** in progress (previous terminal data-gated assessment retained in `reproduction.json.assessment_history`)
**Numerical agreement:** not_assessed
**Preference level:** 1

Aiouez, Hamitouche, Belmadoui, Belattar and Souami. IMPROVE 2022, pp.17–25. [Paper](https://doi.org/10.5220/0010979300003209). Attempt date: 2026-10-01. The tracker assignment requests Tables 3 and 4. All 24 numeric cells, including timing-range endpoints, are accounted for in `reproduction.json`; repeated mAP values remain separate table targets. None are copied literature baselines.

The author-linked Kaggle release is public, but both its unaugmented and augmented metadata explicitly report **Unknown** license. The paper describes availability for interested researchers without specifying cloud-storage/processing permission. The initial October 1 attempt stopped before acquisition. On October 2 the user explicitly authorized using it as a public resource without re-releasing the dataset. This is a project research-use exception, not an owner-granted license. Both public version1 archives have been acquired and all20,920JPEGs decoded and hashed. Canonical annotation/split preparation is in progress; training remains pending representative preflight.

There is also a concrete corpus/split conflict: the paper expands **5,600** original images to **15,088**, then randomly splits 80/10/10. The author metadata instead describes **5,832** unaugmented images (4,651/891/290) and **15,086** augmented images (13,926/870/290). Actual archive inspection reconciles the augmented count: **15,088 JPEGs include two byte-identical unannotated copies of valid labeled originals**. We exclude only those redundant copies, leaving15,086 labeled images. Historical test membership is still unknown. Its augmentation types match the paper: Gaussian blur up to 1 pixel, salt-and-pepper noise up to 5%, 25% grayscale and rotations ±20°.

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

The paper specifies SGD and image size 416×416. Table 2 gives YOLOv5l: lr0.01/batch24/50 epochs, YOLOv5m: lr0.01/batch16/50 epochs, YOLOv5s: lr0.015/batch16/60 epochs, and Faster R-CNN X101-FPN: lr0.001/batch24/60 epochs. YOLO starts from the published COCO pretrained weights and keeps the pinned implementation defaults for momentum, Nesterov SGD, nominal-batch64 accumulation, warmup, scheduling, online augmentation and validation-fitness selection; Table 2 overrides learning rate, batch size and epochs. Faster R-CNN uses the cited X101-32x8d-FPN configuration and declared COCO initialization. Its native ReLU is retained despite the paper table listing SiLU; replacing the published architecture would be a larger deviation. Representative compatibility and data preflights are pending.

The shared `datasets` Volume inventory was inspected through the `repro-sign` wrapper. The intended `belmadoui-arabic-sign-language/` root was absent at the initial inventory; the continuation populates that exact root. `arab-sign/` contains RGB/Depth/Skeleton directories and has not been established as this corpus. `arasl-database-grayscale/` is a distinct 54,049-image, 32-class dataset and was not substituted. The tracker misleadingly associates the same dataset record with both papers.

Raw public Kaggle metadata is retained in `kaggle-unaugmented-metadata.txt` and `kaggle-augmented-metadata.txt`; SHA-256 hashes and URLs live in `reproduction.json`. Dataset split file entries explicitly refer to metadata containing declared counts, not unseen image checksums. `cloud_processing_allowed: true` now records the explicit project authorization; the owner license remains unknown. The full paper PDF is hashed but not redistributed.

To repeat source/gate checks:

```bash
curl -L 'https://www.kaggle.com/api/v1/datasets/view/sabribelmadoui/arabic-sign-language-unaugmented-dataset'
curl -L 'https://www.kaggle.com/api/v1/datasets/view/sabribelmadoui/arabic-sign-language-augmented-dataset'
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh volume ls datasets / --json
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh volume ls datasets belmadoui-arabic-sign-language/ --json
python3 .agents/skills/reproduce-paper/scripts/validate_reproduction.py papers/aiouez-2022-arabic-yolov5
```

The continuation first acquires exact public version1 archives into the shared Modal dataset volume, hashes and inspects every image/annotation, then establishes a seeded 80/10/10 split before preflight. No author contact was made. The paper's ethics flag was investigated as existing human hand-image data; no new participants or webcam collection were introduced.

GPT-6 using Codex performed this source/data investigation and report. Session instructions establish those names; exact model ID and harness version were unavailable. The tracker export is preserved with operator emails redacted and the original source hash retained. Its database IDs differ from paper ID and it has no confirmation field, so this is a direct user-authorized tracker assignment with no invented queue confirmation.

## Authorized continuation (2026-10-02)

The public augmented and unaugmented version1 archives were acquired directly inside Modal; the image corpus is not copied to Git or a public artifact host. Owner license metadata stays Unknown. The retained manifest records original archive bytes, individual file hashes, annotation examples and real image dimensions/counts before split reconstruction. No click-through terms or access controls are bypassed.

The first CPU acquisition is limited to 1,800 seconds and CHF1 with no automatic retry. The independent CPU source probe is limited to 3,600 seconds and CHF2. Neither launches GPU training. Each later GPU preflight/full run must have measured throughput, explicit checkpoint/resume behavior and a prospective ledger ceiling.

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/aiouez-2022-arabic-yolov5/scripts/yolo_modal.py --run-id data-acquisition-v1
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/aiouez-2022-arabic-yolov5/scripts/faster_modal.py --run-id faster-source-probe-002 --wheel-run faster-source-probe-001
```

The original report is preserved in Git and its status/target results in the JSON assessment history. Current pending target records are provisional during execution and will be closed honestly before final report validation.

The initial Faster source probe compiled Detectron2 successfully but exited on a missing `cloudpickle` runtime dependency after 36.63 seconds (no GPU). Its exact sources and closed logs are retained. The scoped follow-up adds `cloudpickle==3.1.1`, reuses the SHA-256-verified compiled wheel and has a separate 300-second/CHF0.25 CPU ceiling.

The second Faster probe reached the model imports but exposed Pillow’s removed `Image.LINEAR` alias. A one-line compatibility patch to the equivalent `Image.BILINEAR` fixed it. Probe003 then constructed the native 28-class X101 model successfully in 9.86 seconds: 104,517,980 total parameters (104.52 million, rounding to the paper’s 105 million). No architecture replacement or GPU was needed. The retained source-probe artifacts and hashes are recorded in the JSON.
