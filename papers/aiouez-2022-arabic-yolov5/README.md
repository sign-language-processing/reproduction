# Real-time Arabic Sign Language Recognition based on YOLOv5

**TL;DR:** All three YOLO full runs completed: AP50 ≈99.5%, AP50:95 87.41/89.17/89.99%; high detection performance and real-time A100 inference are supported.
**Decision:** Partial reproduction (16/24 comparable cells); no human blocker. Faster comparison is unavailable; historical timing and the release's NOON labels limit interpretation.

**Pipeline status:** partial
**Numerical agreement:** does_not_agree — at the declared published-cell precision
**Preference level:** 2

Pinned published implementations with scoped compatibility patches; all execution is terminal.

Aiouez, Hamitouche, Belmadoui, Belattar and Souami. IMPROVE 2022, pp.17–25. [Paper](https://doi.org/10.5220/0010979300003209). Attempt dates: 2026-10-01–03. The tracker assignment requests Tables 3 and 4. All 24 numeric cells, including timing-range endpoints, are accounted for in `reproduction.json`; repeated mAP values remain separate table targets. None are copied literature baselines.

The author-linked Kaggle release is public, but both its unaugmented and augmented metadata explicitly report **Unknown** license. The paper describes availability for interested researchers without specifying cloud-storage/processing permission. The initial October 1 attempt stopped before acquisition. On October 2 the user explicitly authorized using it as a public resource without re-releasing the dataset. This is a project research-use exception, not an owner-granted license. Both public version1 archives have been acquired and all20,920JPEGs decoded and hashed. Canonical annotation and split preparation is complete; Faster training was canceled before its first checkpoint; all three YOLO models subsequently completed full training and evaluation.

There is also a concrete corpus/split conflict: the paper expands **5,600** original images to **15,088**, then randomly splits 80/10/10. The author metadata instead describes **5,832** unaugmented images (4,651/891/290) and **15,086** augmented images (13,926/870/290). Actual archive inspection reconciles the augmented count: **15,088 JPEGs include two byte-identical unannotated copies of valid labeled originals**. We exclude only those redundant copies, leaving15,086 labeled images. Historical test membership is still unknown. Its augmentation types match the paper: Gaussian blur up to 1 pixel, salt-and-pepper noise up to 5%, 25% grayscale and rotations ±20°.

| Table 3 model | Precision, paper → reproduced | Recall | AP50 | AP50:95 |
|---|---:|---:|---:|---:|
| YOLOv5s | 99.2 → 99.7971% | 99.4 → 99.8246% | 99.3 → 99.5000% | 85.9 → 87.4106% |
| YOLOv5m | 99.2 → 99.6924% | 99.2 → 99.8284% | 99.3 → 99.4983% | 87.75 → 89.1688% |
| YOLOv5l | 99.5 → 99.7230% | 99.4 → 99.8931% | 99.4 → 99.4978% | 87.2 → 89.9931% |

Table 4 repeats the small model's AP50 and AP50:95 cells above. Its unfused model has **7.095145 million parameters**, compared with 7.5 million reported. Native Faster X101-32x8d-FPN with the 28-class head has **104.517980 million**, rounding to the reported 105 million. Faster's reported AP50/AP50:95 (98.7/81.38%), inference time (0.55 s; two endpoint targets) and FPS (1.8) were not produced: its original execution was externally canceled before a durable checkpoint, and its immutable attempt window has closed.

The small model's separately measured A100 FP32 batch-one latency is **0.003400–0.009311 s**, mean **0.003607 s**, or **277.21 FPS**, versus the paper's 0.007–0.010 s and121 FPS. This supports real-time operation on the documented A100 setup. These three timing cells are **conditional evidence**, because the exact historical accelerator and measurement boundary are unknown; they are not comparable reproduced timing targets. The 24-cell ledger therefore contains16 produced cells,3 conditional timing cells and5 unavailable trained Faster cells.

The broad finding of high detector performance on this release is supported. The fine-grained ranking is different: the paper's medium model leads AP50:95; here large leads (89.99%), followed by medium (89.17%) and small (87.41%). Small has the highest precision and marginally highest AP50 here; large has the highest recall. No trained Faster comparison or full 28-letter semantic-recognition conclusion is established. These are single-seed descriptive comparisons, not significance tests.

The full paper, publisher/conference pages, references, author ResearchGate profile, exact-title/author GitHub searches and Zenodo/OSF searches were inspected. No paper-specific code, split or trained weights were found. The cited generic implementations are [YOLOv5 v6.0](https://github.com/ultralytics/yolov5/tree/956be8e642b5c10af4a1533e09084ca32ff4f21f) and [Detectron2 v0.6](https://github.com/facebookresearch/detectron2/tree/d1e04565d3bec8719335b88be9e9b961bf3ec464), preserved as historically plausible candidate pins rather than asserted author versions. YOLOv5's train/evaluation entry points were inspected. The initial investigation made no source changes; continuation compatibility patches and tests are documented below.

The paper specifies SGD and image size 416×416. Table 2 gives YOLOv5l: lr0.01/batch24/50 epochs, YOLOv5m: lr0.01/batch16/50 epochs, YOLOv5s: lr0.015/batch16/60 epochs, and Faster R-CNN X101-FPN: lr0.001/batch24/60 epochs. YOLO starts from the published COCO pretrained weights and keeps the pinned implementation defaults for momentum, Nesterov SGD, nominal-batch64 accumulation, warmup, scheduling, online augmentation and validation-fitness selection; Table 2 overrides learning rate, batch size and epochs. Faster R-CNN uses the cited X101-32x8d-FPN configuration and declared COCO initialization. Its native ReLU is retained despite the paper table listing SiLU; replacing the published architecture would be a larger deviation. Shared data preparation and all four models’ preflight completed. The three YOLO full runs completed; Faster was interrupted.

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

The original report is preserved in Git and its status/target results in the JSON assessment history. Every current target has a terminal result; prior gated results remain historical evidence.

The initial Faster source probe compiled Detectron2 successfully but exited on a missing `cloudpickle` runtime dependency after 36.63 seconds (no GPU). Its exact sources and closed logs are retained. The scoped follow-up adds `cloudpickle==3.1.1`, reuses the SHA-256-verified compiled wheel and has a separate 300-second/CHF0.25 CPU ceiling.

The second Faster probe reached the model imports but exposed Pillow’s removed `Image.LINEAR` alias. A one-line compatibility patch to the equivalent `Image.BILINEAR` fixed it. Probe003 then constructed the native 28-class X101 model successfully in 9.86 seconds: 104,517,980 total parameters (104.52 million, rounding to the paper’s 105 million). No architecture replacement or GPU was needed. The retained source-probe artifacts and hashes are recorded in the JSON.

The annotation audit found one degenerate single-box image (zero height), excluded without inventing a box; the usable corpus therefore has15,085 images. The author YAML still defines28 categories, but NOON/class24 has no annotation instances. We retain the28-class head and all remaining author labels unchanged, cover27 supported categories in preflight, and report NOON as untested. This is a release inconsistency, not a request for new annotation or a reason to halt the detector comparison. Exact excluded IDs and raw rows remain in the internal audit artifact.

Cross-release audit resolves the missing-class issue: all five class-name arrays match, but **14 byte-identical images change from NOON in the original release to ALIF in the augmented release**; four also retain identical boxes. Filename evidence associates203 original NOON families with529 augmented ALIF images, but transformed-image lineage remains heuristic. The primary reproduction follows the released augmented labels unchanged, and its results cannot establish28-letter semantic recognition. No new human annotation or speculative relabeling is introduced. The internal report and source hashes are retained.

Faster R-CNN representative GPU preflight completed successfully in124.03 seconds with10 real updates and10.92GB peak GPU memory. Native model, optimizer and scheduler states restore exactly at step8. The native COCO evaluator produced detections/AP0 at the first tiny evaluation; the final tiny evaluation produced no detections/APNaN, retained as diagnostic evidence rather than target scores. Faster full training subsequently launched under the immutable deadline below.

The now-interrupted Faster R-CNN full run is `faster-full-001` on output Volume `repro-992e7a-results`, app `ap-WnKAcIYIQNxWkd4z5bYrtB`, function call `fc-01M3XYR3J6YA9D2K9DJGP13J59`. Started 2026-10-02 09:22:43.015471 UTC; immutable deadline 2026-10-02 21:22:43.015471 UTC. Provider replay is limited to four total segments within that original wall limit. A replay with no verified recovery checkpoint fails closed. Native RNG/data-loader cursor are not restored; optimizer, scheduler, model and update count are restored.

Inspect through the required workspace wrapper:
```
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh volume get repro-992e7a-results faster-full-001/execution.json /tmp/faster-execution.json
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh volume get repro-992e7a-results faster-full-001/evidence.json /tmp/faster-evidence.json
```
`execution.json` lists every segment and native command. Read the current segment's `console-segment-NN.log`; earlier interrupted segments have unknown true native exit/time. A live snapshot is provisional, not terminal evidence. In a run reaching an epoch boundary, `recovery.json` would contain the durable checkpoint SHA and completed updates and `selection.json` the validation improvements. Neither exists for this interrupted attempt. Never use test metrics for selection, restart the full command with the same run ID, or replace the original deadline.

After the GPU run is terminal, prospectively declare the CPU collector (2 CPU, 4 GiB, 900 seconds, no GPU, one attempt, estimated below CHF1), then:
```
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/aiouez-2022-arabic-yolov5/scripts/faster_collect.py --run-id faster-full-001
```
The collector hashes all closed files with four bounded readers and verifies the selected checkpoint. It preserves GPU execution/native result bytes. For this interrupted attempt, collect `evidence.json`, `execution.json`, console and the timestamped `collection-*.json` receipt. Result/selection files are absent; the collector does not invent them. The receipt is separately hashed and excluded from its own manifest. Raw predictions/checkpoints stay internal on Modal under the explicit no-redistribution permission exception.

On success, native `result.json.test.bbox.AP50` and `.AP` are percentages. Parameter count is `all_parameters / 1e6`; full timing reports FP32 batch1 mean seconds and its reciprocal FPS, using all 1509 held-out images after five warmups and excluding file decode. The native model has 28 outputs but only 27 observed classes; COCO ignores absent NOON when averaging. Preserve native NaNs for unsupported categories/areas in raw evidence and use null in strict machine-readable aggregate records. Report the hardware/protocol difference from unspecified Colab measurements.

Expected training length is 30180 updates: 60 logical epochs × ceil(12068 / 24) = 503 updates each under the native infinite sampler. Exactly 60 epoch boundaries are validation opportunities. Native validation `bbox/AP` selects the first maximum (strict improvement); test is evaluated only after loading the selected checkpoint. Initial forecast was 8–10 hours, but the first20 full-corpus updates showed cold I/O at2.09 seconds/update. Cancellation prevented reaching the first validation boundary.


The continuation has an aggregate24GPU-hour/CHF90 ceiling including preparation and diagnostics. Independent single-GPU model jobs may overlap. Faster full has an immutable12-hour/CHF42 allocation; combined YOLO full training is limited to9GPU-hours/CHF32 after measured preflight. Prices and conservative reservation assumptions are recorded in the JSON; these are ceilings, not actual bills.

YOLO CPU diagnostics retain every failure. The first image build stopped on a DNS error before any function ran; the provisioning retry then exposed a Debian PyYAML package without a pip RECORD. Retaining the base's compatible PyYAML6.0.1 completed the build. The first real smoke invocation verified all168 preflight image/label pairs, then exposed a wrapper import mistake (`intersect_dicts` belongs in `utils.torch_utils`); the corrected import subsequently passed. These are environment/glue checks, not target experiments. Closed logs, exact source bytes and hashes remain linked in the ledger.

The corrected YOLO import passed in smoke v4. Its native loader read all112 training images, then failed on NumPy's removed `np.int` alias. The narrow `patches/yolo-numpy-int.patch` replaces that alias with the historically identical builtin `int`; no numerical algorithm or training setting changes. Smoke v5 passed this correction before exposing the next compatibility issue.

Smoke v5 passed real training/validation loading and COCO model forward, then exposed modern PyTorch rejecting floating clamp bounds for integer grid indices. `patches/yolo-torch-grid-bounds.patch` uses the original integer grid dimensions that supplied those exact bounds. Smoke v6 exercised the real loss and all five NumPy alias call sites, then exposed a smoke-only class-weight argument error resolved in v7. These are compatibility corrections, not changed loss or geometry.


## October 3 continuation

The Modal platform log confirms that Faster R-CNN received a cancellation at October2 09:39:19UTC. Its wrapper’s `finally` block recorded default exit1, but the native child continued logging through480updates at09:40:27. This is an interrupted run with unknown true native exit, not evidence of a deterministic model failure. No503-update recovery checkpoint, validation selection or held-out result exists. The original deadline has expired; no restart or deadline extension was attempted. CPU collection independently hashed all12closed files. The parameter count from the completed native source/preflight remains valid; Faster detection accuracy, timing and the direct trained-detector comparison remain unavailable.

YOLO smoke v7 corrected a smoke-only helper argument (explicit28-element class weights rather than its80-class default), then passed real image loading, COCO initialization, finite native loss and finite backward. The all-model GPU preflight completed in204.68seconds on A10080GB. Native small-model weights, optimizer, EMA and update counts restored exactly in a fresh process. Warm batch times for small/medium/large were0.08367/0.23175/0.43680seconds; measured GPU peaks1.26/2.41/5.63GB. These imply1.05/2.43/3.05traininghours before full validation, checkpoint and setup overhead. Planned full ceilings are1.8/3.2/4GPU-hours respectively (9combined/CHF32), within the aggregate24GPU-hour/CHF90 continuation ceiling. These are forecasts, not completed results.

The new durable-recovery proof stores immutable copies of native epoch checkpoints and validation-selected checkpoints, verifies hashes before native restore, and tests evaluation-only replay after training finishes. Native FP16 model/EMA serialization and optimizer restore are preserved; the authors’ code does not preserve RNG/data-loader position or AMP GradScaler state. This limitation is reported rather than described as bitwise trajectory continuity. Each full call has an immutable original deadline and a maximum of four provider segments; replay before a durable checkpoint fails closed. No score-driven recipe tuning is permitted.

The October3 continuation agent is GPT-6 using Codex; exact model and harness versions were not exposed. It collects/reviews earlier work and executes new YOLO diagnostics. Earlier acquisition, Faster execution and YOLO diagnostics retain their original executor attribution.

Durable recovery proof v2 passed: exact native metrics and selected model tensors after evaluation replay. A local fixture executes the actual full-wrapper function and confirms cancellation kills its child, records an unknown native exit, rejects a duplicate logical call and refuses an expired deadline without resetting it. Repeat with `python3 papers/aiouez-2022-arabic-yolov5/scripts/yolo_guard_check.py`. Independent root review approved the exact launcher and training source hashes recorded per run.

## Completed YOLO execution and collection

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


All three full runs finished on October3, each with one execution segment and no resume. Small finished19:04:47UTC in2372.61seconds, medium19:06:54UTC in2504.28seconds, and large19:19:26UTC in3253.22seconds. Their combined conservative GPU allocation time was **2.2584 GPU-hours**, below the9-hour/CHF32 full-run ceiling. At the retained planning rates, GPU+4CPU+64GiB cost is approximatelyCHF7.22 equivalent; this is an estimate using1USD=1CHF for reservation, not an invoice, and excludes storage/build/collection. Whole-paper billing remains unknown, particularly the earlier canceled native Faster child; its12-hour reservation is not represented as measured usage. Full-run peaks were1.298/2.501/5.806GB and native training spans2318.53/2451.85/3167.22seconds.

Training really consumed12,068 images each epoch for60/50/50 epochs at416×416. **Native YOLO validation/test uses its standard letterbox padding and stride, yielding448×448 tensors despite requested image size416.** This is the pinned implementation's default evaluation path, preserved before observing scores; the detector values are interpreted under the explicitly reconstructed native protocol, not an assertion about undocumented author tensor shapes. The separate FP32 timing measurement uses actual416×416 tensors. No post-score architecture, split, threshold or padding changes were made.

Each separate CPU collector verified the full epoch sequence, selected checkpoint SHA and1509 unique test-image timing samples, then independently recomputed COCO metrics from frozen test annotations and saved predictions. Its results exactly matched the stored common-evaluator values. Common pycocotools2.0.11 AP50/AP50:95 (%) are small100.0000/87.5115, medium99.9967/89.2250, large99.9957/90.0976. These are an independent audit and must not replace the native YOLO values in the target table. Native P/R are macro averages at the confidence maximizing mean F1; AP averages the27 observed labels. The absent NOON value is null rather than native YOLO's misleading global-average fallback.

The current source commit for execution is5741511. GPU base image is `ghcr.io/sign-language-processing/reproduction@sha256:305b6165d306192996358ca312d9a751fa409f43063a76dc7758880a8f905291`; Dockerfiles preserve explicit dependency pins and every full run retains its pip freeze, NVIDIA driver report, exact source/patch bytes, claim, console, options, hyperparameters, predictions and checkpoint on the internal output Volume. The dataset manifest SHA is `bdb2b2a87d2b93af07d977dafac14cf5f99c2e3333e97350cada17da8475356e`; test COCO SHA is `3a75cd1b9528ffc7c69b2df9df7d54579bbb4b45ed63f45f5c728c7e3cb36049`. Seed42 partitions12068/1508/1509 eligible images; native training seed is0. Byte identities are disjoint, while augmented source families may cross the random split as in the paper's augmentation-before-split order. This does not establish signer-independent generalization.

For a new, prospectively declared attempt, first run `./setup.sh` once per clone and verify the canonical dataset and cache:
```bash
.agents/skills/reproduce-paper/scripts/check_modal_dataset.sh belmadoui-arabic-sign-language manifest.json
python3 papers/aiouez-2022-arabic-yolov5/scripts/yolo_guard_check.py
```
The exact acquisition, annotation, preflight, recovery, train and collection commands are retained per run in `reproduction.json.runs[].command`. Full launch commands below illustrate the completed settings; **do not relaunch these closed IDs or reset their original deadlines**. A future new attempt requires a newly declared budget and run ID.
```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/aiouez-2022-arabic-yolov5/scripts/yolo_full.py --run-id yolo-s-full-001 --model s --weights-sha256 c3b140f32001a9eec4afa07120b3851eb1b6c2c7c7e7a4303af9eadfacbeb598 --max-wall-seconds 6480
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/aiouez-2022-arabic-yolov5/scripts/yolo_full.py --run-id yolo-m-full-001 --model m --weights-sha256 4947bf5605d671037cd4b4cb250e5f52f836c0c5b97c1084aca391997f78d593 --max-wall-seconds 11520
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/aiouez-2022-arabic-yolov5/scripts/yolo_full.py --run-id yolo-l-full-001 --model l --weights-sha256 84f0a2d6945dfd65d64b0595ea7b912f6addac32dd23fe77bc3fa6d3c1385f16 --max-wall-seconds 14400
```
Each launcher verifies the canonical manifest and initial weight hash, stages the hash-verified source archive to ephemeral disk, uses read-only canonical data and the shared Hugging Face cache, and refuses another logical invocation of the same run. Large artifacts and restricted data remain internal; no weights or dataset were published. Closed artifact SHA-256 identifiers, exact Modal app/call IDs, UTC timestamps and collection receipts live in the ledger, including every meaningful failed diagnostic. All target values point to those immutable metric/checkpoint hashes.

Attribution: `yolo-agent` investigated sources; `yolo-continuation-agent` acquired/prepared data and initial YOLO diagnostics; `faster-continuation-agent` executed Faster probes/preflight/full attempt; `independent-protocol-reviewer` reviewed protocol and audited cross-release labels; `continuation-orchestrator` handled permission/protocol review and report coordination; `continuation-oct3-agent` executed the retained YOLO full runs and CPU collections and finalized the report. All exposed sessions identify GPT-6 using Codex; exact model IDs and harness versions were unavailable. Root's later independent evidence review does not change executor attribution. Evidence and contribution limits are preserved in `reproduction.json.agents`.

No human decision is needed to close this bounded attempt. The substantive unresolved historical question is which annotation release the authors used: the current augmented release has verified NOON→ALIF corruption. Historical test membership and GPU timing details also remain unavailable. No author contact, new human annotation, label guessing or unauthorized full restart was performed.

Validation: the repository reproduction validator passed on the final record; strict finite JSON, Python AST parsing, guard fixture, internal evidence references, privacy scan and `git diff --check` passed. Root independently checked all three downloaded metric/manifest/receipt hashes, epoch counts, timing IDs and common COCO statistics. Closed YOLO artifact manifests cover approximately0.281/0.779/1.687GB for small/medium/large; storage remains internal.
