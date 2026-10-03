# Real-time Arabic Sign Language Recognition based on YOLOv5

**TL;DR:** YOLO three-model preflight passed; all three bounded full runs are running after the durable-recovery check passed. Faster R-CNN was canceled before its first checkpoint.
**Decision:** License remains unknown; dataset redistribution is prohibited. No human action is pending.

**Execution state:** in progress (previous terminal data-gated assessment retained in `reproduction.json.assessment_history`)
**Numerical agreement:** not_assessed
**Preference level:** 1

Aiouez, Hamitouche, Belmadoui, Belattar and Souami. IMPROVE 2022, pp.17–25. [Paper](https://doi.org/10.5220/0010979300003209). Attempt date: 2026-10-01. The tracker assignment requests Tables 3 and 4. All 24 numeric cells, including timing-range endpoints, are accounted for in `reproduction.json`; repeated mAP values remain separate table targets. None are copied literature baselines.

The author-linked Kaggle release is public, but both its unaugmented and augmented metadata explicitly report **Unknown** license. The paper describes availability for interested researchers without specifying cloud-storage/processing permission. The initial October 1 attempt stopped before acquisition. On October 2 the user explicitly authorized using it as a public resource without re-releasing the dataset. This is a project research-use exception, not an owner-granted license. Both public version1 archives have been acquired and all20,920JPEGs decoded and hashed. Canonical annotation and split preparation is complete; Faster training was canceled before its first checkpoint; all three YOLO models subsequently passed GPU preflight.

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

The full paper, publisher/conference pages, references, author ResearchGate profile, exact-title/author GitHub searches and Zenodo/OSF searches were inspected. No paper-specific code, split or trained weights were found. The cited generic implementations are [YOLOv5 v6.0](https://github.com/ultralytics/yolov5/tree/956be8e642b5c10af4a1533e09084ca32ff4f21f) and [Detectron2 v0.6](https://github.com/facebookresearch/detectron2/tree/d1e04565d3bec8719335b88be9e9b961bf3ec464), preserved as historically plausible candidate pins rather than asserted author versions. YOLOv5's train/evaluation entry points were inspected. The initial investigation made no source changes; continuation compatibility patches and tests are documented below.

The paper specifies SGD and image size 416×416. Table 2 gives YOLOv5l: lr0.01/batch24/50 epochs, YOLOv5m: lr0.01/batch16/50 epochs, YOLOv5s: lr0.015/batch16/60 epochs, and Faster R-CNN X101-FPN: lr0.001/batch24/60 epochs. YOLO starts from the published COCO pretrained weights and keeps the pinned implementation defaults for momentum, Nesterov SGD, nominal-batch64 accumulation, warmup, scheduling, online augmentation and validation-fitness selection; Table 2 overrides learning rate, batch size and epochs. Faster R-CNN uses the cited X101-32x8d-FPN configuration and declared COCO initialization. Its native ReLU is retained despite the paper table listing SiLU; replacing the published architecture would be a larger deviation. Shared data preparation and all four models’ preflight are complete; full-run results remain pending.

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

The annotation audit found one degenerate single-box image (zero height), excluded without inventing a box; the usable corpus therefore has15,085 images. The author YAML still defines28 categories, but NOON/class24 has no annotation instances. We retain the28-class head and all remaining author labels unchanged, cover27 supported categories in preflight, and report NOON as untested. This is a release inconsistency, not a request for new annotation or a reason to halt the detector comparison. Exact excluded IDs and raw rows remain in the internal audit artifact.

Cross-release audit resolves the missing-class issue: all five class-name arrays match, but **14 byte-identical images change from NOON in the original release to ALIF in the augmented release**; four also retain identical boxes. Filename evidence associates203 original NOON families with529 augmented ALIF images, but transformed-image lineage remains heuristic. The primary reproduction follows the released augmented labels unchanged, and its results cannot establish28-letter semantic recognition. No new human annotation or speculative relabeling is introduced. The internal report and source hashes are retained.

Faster R-CNN representative GPU preflight completed successfully in124.03 seconds with10 real updates and10.92GB peak GPU memory. Native model, optimizer and scheduler states restore exactly at step8. The native COCO evaluator produced detections/AP0 at the first tiny evaluation; the final tiny evaluation produced no detections/APNaN, retained as diagnostic evidence rather than target scores. Faster full training subsequently launched under the immutable deadline below.

The now-interrupted Faster R-CNN full run is `faster-full-001` on output Volume `repro-992e7a-results`, app `ap-WnKAcIYIQNxWkd4z5bYrtB`, function call `fc-01M3XYR3J6YA9D2K9DJGP13J59`. Started 2026-10-02 09:22:43.015471 UTC; immutable deadline 2026-10-02 21:22:43.015471 UTC. Provider replay is limited to four total segments within that original wall limit. A replay with no verified recovery checkpoint fails closed. Native RNG/data-loader cursor are not restored; optimizer, scheduler, model and update count are restored.

Inspect through the required workspace wrapper:
```
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh volume get repro-992e7a-results faster-full-001/execution.json /tmp/faster-execution.json
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh volume get repro-992e7a-results faster-full-001/recovery.json /tmp/faster-recovery.json
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh volume get repro-992e7a-results faster-full-001/selection.json /tmp/faster-selection.json
```
`execution.json` lists every segment and native command. Read the current segment's `console-segment-NN.log`; earlier interrupted segments have unknown true native exit/time. A live snapshot is provisional, not terminal evidence. `recovery.json` contains the latest durable training checkpoint SHA and completed updates. `selection.json` contains validation AP improvements and separate model-only best checkpoints. Never use test metrics for selection, restart the full command with the same run ID, or replace the original deadline.

After the GPU run is terminal, prospectively declare the CPU collector (2 CPU, 4 GiB, 900 seconds, no GPU, one attempt, estimated below CHF1), then:
```
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/aiouez-2022-arabic-yolov5/scripts/faster_collect.py --run-id faster-full-001
```
The collector hashes all closed files with four bounded readers and verifies the selected checkpoint. It preserves GPU execution/native result bytes. Download `evidence.json`, `collected-metrics.json` if the GPU run succeeded, `result.json`, `execution.json`, `selection.json`, and the timestamped `collection-*.json` receipt. The receipt is separately hashed and excluded from its own manifest. Raw predictions/checkpoints stay internal on Modal under the explicit no-redistribution permission exception.

On success, native `result.json.test.bbox.AP50` and `.AP` are percentages. Parameter count is `all_parameters / 1e6`; full timing reports FP32 batch1 mean seconds and its reciprocal FPS, using all 1509 held-out images after five warmups and excluding file decode. The native model has 28 outputs but only 27 observed classes; COCO ignores absent NOON when averaging. Preserve native NaNs for unsupported categories/areas in raw evidence and use null in strict machine-readable aggregate records. Report the hardware/protocol difference from unspecified Colab measurements.

Expected training length is 30180 updates: 60 logical epochs × ceil(12068 / 24) = 503 updates each under the native infinite sampler. Exactly 60 epoch boundaries are validation opportunities. Native validation `bbox/AP` selects the first maximum (strict improvement); test is evaluated only after loading the selected checkpoint. Initial forecast was 8–10 hours, but first20 full-corpus updates showed cold I/O at 2.09 seconds/update; monitor the next windows and first epoch before trusting the forecast.


The continuation has an aggregate24GPU-hour/CHF90 ceiling including preparation and diagnostics. Independent single-GPU model jobs may overlap. Faster full has an immutable12-hour/CHF42 allocation; combined YOLO full training is limited to9GPU-hours/CHF32 after measured preflight. Prices and conservative reservation assumptions are recorded in the JSON; these are ceilings, not actual bills.

YOLO CPU diagnostics retain every failure. The first image build stopped on a DNS error before any function ran; the provisioning retry then exposed a Debian PyYAML package without a pip RECORD. Retaining the base's compatible PyYAML6.0.1 completed the build. The first real smoke invocation verified all168 preflight image/label pairs, then exposed a wrapper import mistake (`intersect_dicts` belongs in `utils.torch_utils`); the corrected import is being tested. These are environment/glue checks, not target experiments. Closed logs, exact source bytes and hashes remain linked in the ledger.

The corrected YOLO import passed in smoke v4. Its native loader read all112 training images, then failed on NumPy's removed `np.int` alias. The narrow `patches/yolo-numpy-int.patch` replaces that alias with the historically identical builtin `int`; no numerical algorithm or training setting changes. Smoke v5 tests this correction.

Smoke v5 passed real training/validation loading and COCO model forward, then exposed modern PyTorch rejecting floating clamp bounds for integer grid indices. `patches/yolo-torch-grid-bounds.patch` uses the original integer grid dimensions that supplied those exact bounds. Smoke v6 checks the real loss and finite backward as well as all five NumPy alias call sites. These are compatibility corrections, not changed loss or geometry.


## October 3 continuation

The Modal platform log confirms that Faster R-CNN received a cancellation at October2 09:39:19UTC. Its wrapper’s `finally` block recorded default exit1, but the native child continued logging through480updates at09:40:27. This is an interrupted run with unknown true native exit, not evidence of a deterministic model failure. No503-update recovery checkpoint, validation selection or held-out result exists. The original deadline has expired; no restart or deadline extension was attempted. CPU collection independently hashed all12closed files. The parameter count from the completed native source/preflight remains valid; Faster detection accuracy, timing and the direct trained-detector comparison remain unavailable.

YOLO smoke v7 corrected a smoke-only helper argument (explicit28-element class weights rather than its80-class default), then passed real image loading, COCO initialization, finite native loss and finite backward. The all-model GPU preflight completed in204.68seconds on A10080GB. Native small-model weights, optimizer, EMA and update counts restored exactly in a fresh process. Warm batch times for small/medium/large were0.08367/0.23175/0.43680seconds; measured GPU peaks1.26/2.41/5.63GB. These imply1.05/2.43/3.05traininghours before full validation, checkpoint and setup overhead. Planned full ceilings are1.8/3.2/4GPU-hours respectively (9combined/CHF32), within the aggregate24GPU-hour/CHF90 continuation ceiling. These are forecasts, not completed results.

The new durable-recovery proof stores immutable copies of native epoch checkpoints and validation-selected checkpoints, verifies hashes before native restore, and tests evaluation-only replay after training finishes. Native FP16 model/EMA serialization and optimizer restore are preserved; the authors’ code does not preserve RNG/data-loader position or AMP GradScaler state. This limitation is reported rather than described as bitwise trajectory continuity. Each full call has an immutable original deadline and a maximum of four provider segments; replay before a durable checkpoint fails closed. No score-driven recipe tuning is permitted.

The October3 continuation agent is GPT-6 using Codex; exact model and harness versions were not exposed. It collects/reviews earlier work and executes new YOLO diagnostics. Earlier acquisition, Faster execution and YOLO diagnostics retain their original executor attribution.

Durable recovery proof v2 passed: exact native metrics and selected model tensors after evaluation replay. A local fixture executes the actual full-wrapper function and confirms cancellation kills its child, records an unknown native exit, rejects a duplicate logical call and refuses an expired deadline without resetting it. Repeat with `python3 papers/aiouez-2022-arabic-yolov5/scripts/yolo_guard_check.py`. Independent root review approved the exact launcher and training source hashes recorded per run.

## Current YOLO execution and collection

All three full runs launched independently on October3 at approximately18:25UTC. Each uses one A10080GB and the exact committed source (`5741511`); no multi-GPU training is introduced. These are original deadlines and must never be reset.

| Model | Run | App | Function call | Deadline UTC |
|---|---|---|---|---|
| s | `yolo-s-full-001` | `ap-HYBot17D083xtvujzvHqBq` | `fc-01M41G64B0ZDEVVTAVAACTKZZA` | 2026-10-03T20:13:14.674843+00:00 |
| m | `yolo-m-full-001` | `ap-SInmo7oY4Ggf812hW4bKQ3` | `fc-01M41G65PB6EPPPM2DFJS9N22C` | 2026-10-03T21:37:10.398795+00:00 |
| l | `yolo-l-full-001` | `ap-Vo2K1d1Q3GkddT1gYJs5X4` | `fc-01M41G67EA03JHAKXMDW6P5APY` | 2026-10-03T22:25:13.728302+00:00 |

Monitor the immutable claim, `execution.json`, current `console-segment-NN.log`, and `MODEL/recovery.json` in each run directory on `repro-992e7a-results`, always through the required wrapper. Each recovery pointer records completed epoch and SHA-256 of the native checkpoint and validation-selected checkpoint. A running or submitted job is not a completed reproduction.

After a run is terminal, declare one CPU collection attempt (2 CPU,4GiB,900seconds,CHF1,noGPU,no retry), then run the following, substituting its recorded run ID:
```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/aiouez-2022-arabic-yolov5/scripts/yolo_collect.py --run-id yolo-s-full-001
```
The collector verifies60/50/50 native CSV epochs, selected checkpoint hash, and1509test timing samples on successful runs; it independently recomputes the common COCO metrics from saved predictions and the frozen1509-image annotation file, then hashes every closed artifact. Download `evidence.json`, timestamped `collection-*.json`, `execution.json`, `MODEL/metrics.json`, and `MODEL/train/results.csv` individually. Keep predictions, images and weights internal. Native YOLO target values are `native_test_metrics[0:4]` (P,R,AP50,AP50:95), multiplied by100 for percent; independent common COCO metrics are reported separately. Native validation fitness selects the best checkpoint (ties use the native last matching epoch); no test selection is performed.
