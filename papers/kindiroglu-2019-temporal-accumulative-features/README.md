# Kindiroglu et al. 2019, Temporal Accumulative Features: reproduction

**Paper ID:** `13fbc66718d1a4b1c05e0aad78da9642cb61925e`

**Citation:** Kindiroglu, A. A., Ozdemir, O., Akarun, L. (2019). Temporal Accumulative Features for Sign Language Recognition. IEEE/CVF International Conference on Computer Vision Workshops (ICCVW). arXiv:2004.01225

**Paper:** https://arxiv.org/abs/2004.01225 · **Code/artifacts:** none published; found after search: none.

**Preference level:** 3

**Pipeline status:** `insufficient_information` — the pipeline ran end to end, but only under guessed protocol details, so no target counts as produced

**Numerical agreement:** `not_assessed`

**Attempt date:** 2026-11-24 to 2026-11-26

## Reproduction agents

| Agent ID | Model and version | Agent application | Contribution | Attribution evidence / unknowns |
| --- | --- | --- | --- | --- |
| `reproduction-agent` | Fable 5.2 (`claude-fable-5-2`) | Claude Code 2.1.330 | Assignment ingestion, target resolution, source search, data gating, the dataset population code, the reimplementation, execution of every run, evaluation and report. | Session metadata on 2026-11-24: the session's system context names the model 'Fable 5.2' with model ID 'claude-fable-5-2'.; Command output on 2026-11-24: 'claude --version' printed '2.1.330 (Claude Code)'. |

Every run records `agent_ids: ["reproduction-agent"]`. No other agent contributed.

## Scope and target contract

The queue record asks for the highest-accuracy line of **Table 5** (p. 8): row "Keyshot, FL-4, Single", accuracy 81.58% and top-5 accuracy 96.94%, on the BosphorusSign General subset (174 signs; 4,839 training videos; 949 test videos of user 4). Both numbers are in `reproduction.json.targets` and in scope. The paper's own experiments produced them (not copied baselines).

The system: joint heatmaps from Kinect skeletons are accumulated over time into color images in which hue encodes the segment between keyframes (Temporal Accumulative Features); keyframes come from hand-speed minima and density clustering; binary hand masks of five frames around each of four keyframes ("keyshot") are appended; a small CNN classifies the result.

Decisions taken at human gates, recorded in `reproduction.json.gates`:

| Gate | Type | Decision |
| --- | --- | --- |
| `bsign22k-data-cloud-use` | data | A study team member confirmed that the study has permission for this use and that the empty queue fields are an omission on the annotation platform, and authorized mirroring the dataset folder to /datasets/bosphorussign22k on the shared Volume. The copy stays private to the workspace; nothing from the corpus is committed, published or redistributed. |
| `gdrive-client-id` | data | No new credential was created. A study team member downloaded the folder, without depth.tar.gz, through the Google Drive web interface as 42 zip archives and two CSV files, and asked for it to be uploaded from that local copy. The same person asked for the partial copy left by populate-dataset-1 and populate-dataset-2 to be removed from the Volume; /datasets/bosphorussign22k was deleted on 2026-11-24 before the upload. |
| `hand-datasets-cloud-use` | data | A study team member authorized placing HGR2A, HGR2B and the Rendered Handpose Dataset on the shared Volume as separate datasets, downloaded on Modal from the official pages. |
| `taf-unstated-protocol` | protocol | Decided by a study team member before any training run. CNN: 2 blocks of 2 convolutional layers, 128 initial filters, dropout 0.25 (the Table 3 row that Table 4 reuses). Joint positions: the Kinect files' color-image pixel coordinates scaled from 1920x1080 to 116x64; Gaussian standard deviation 2 pixels; 'upper arms' are the shoulder joints. Seven channels per joint: hue-colored image (3), intensity (1), normalized image (3). Keyframes: density-peaks clustering on (frame, speed) points of the dominant hand's speed minima, the 6 strongest peaks, first and last dropped; FL-4 means 4 keyframes kept, giving 20 mask channels and 97 input channels. Hand masks: whole-frame mask of both hands resized to 116x64, one channel per frame. Segmentation: DeepLab v3+ with an ImageNet-pretrained ResNet-50 backbone in PyTorch, on the Rendered Handpose Dataset plus HGR1, HGR2A and HGR2B; HGR images are trained as hand versus background only, so no left/right label is invented. Classifier training: Adam, learning rate 0.001, batch 32, 100 epochs, weights after the last epoch, no selection on the test set; seed 0 is the reported value and seeds 1 and 2 are supplementary. Framework: PyTorch on the study base image. Values obtained under these choices are conditional evidence and do not count as produced targets. |

## Source provenance

| Artifact | Canonical source | Pinned revision / SHA-256 | Role |
| --- | --- | --- | --- |
| `paper-pdf` | https://arxiv.org/pdf/2004.01225 | `a4a90b325a74ce0d0fd3e4c75758b9cb6fbe3485a4f9817c4e1853d24b58933a` | Targets (Table 5) and the disclosed method; the only source of the protocol. |

**Search for published code (2026-11-26).** No code for this paper was found. Preference level 3: reimplementation from the paper.

- Paper PDF (arXiv:2004.01225), all pages: no code, data or supplement link; the dataset is described as 'available on request'.
- CVF open-access page for the ICCVW 2019 paper and the arXiv abstract page, as returned by web search: no code link.
- Papers with Code entry for the paper, as returned by web search: no code implementations listed.
- Web searches: the exact title with 'code github'; the three author surnames with 'temporal accumulative features TAF PoTion sign language implementation repository': paper pages only.
- GitHub repository search for 'temporal accumulative features', 'temporal accumulative features sign language' and 'BosphorusSign TAF': no repositories. Search for 'bosphorussign22k': one repository, the dataset's web page.
- Public repositories of two authors' GitHub accounts (found by name search) and of the authors' laboratory organization: none implements this paper. One, isolated-slr (last pushed 2019-09-20), holds 3D-CNN and LSTM data loaders and BosphorusSign split files for a different method; code search for 'accumulative' and 'potion' in that account returns nothing.
- PoTion (Choutas et al. 2018), the cited basis of the representation, is a different paper's method and was not used as code.

## Results

No target is marked produced. The values below are **conditional evidence**: they come from a reimplementation under the guesses listed further down, as decided at gate `taf-unstated-protocol` before any training run.

| Target | Published | Conditional value (seed 0) | Difference | Seed 1 | Seed 2 |
| --- | --- | --- | --- | --- | --- |
| Accuracy, Table 5 "Keyshot, FL-4, Single" | 81.58 | 70.39 | -11.19 | 67.97 | 66.17 |
| Top-5 accuracy, same row | 96.94 | 94.31 | -2.63 | 94.20 | 92.52 |

- Seed 0 classifies 668 of 949 test videos correctly at top 1 and 895 at top 5 (recomputed from the saved class scores).
- Values are taken after the 100th epoch, without selection on the test set. Test accuracy varies strongly between epochs: over the last 10 epochs it ranges from 63.96 to 72.60 for seed 0. The highest test accuracy of any epoch, which this attempt does not use, is 73.87, 71.97 and 74.71 for seeds 0, 1 and 2.
- Hand segmentation: validation IoU 92.6% (left hand) and 93.1% (right hand) on 428 held-out images, against the paper's about 78% and 83%. The validation images are almost all rendered ones; mask quality on the sign videos is not measured, since they have no ground truth. The masks mark about 0.85% of a frame's pixels as hand.

## How to repeat this

```bash
./setup.sh   # once per clone; then `modal setup` for workspace repro-sign
P=papers/kindiroglu-2019-temporal-accumulative-features
M=.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh
# Data (needs the dataset owners' Drive export; see data.sh for the file list)
$P/data.sh FILE...
$M run $P/modal_app.py::hash_hgr1
# Segmentation training, keyshot masks, classifier for seeds 0, 1, 2
$M run --detach $P/modal_app.py::full
# One stage alone, e.g. a small preflight:
$M run $P/modal_app.py::stage --name seg --out preflight --limit 640 --epochs 2
```

Results are written to the Modal Volume `kindiroglu-2019-results` under `full/`. Finished stages are skipped on a re-run.

## Data provenance and permissions

| Dataset | Version / subset | Source | Permission basis | Modal path |
| --- | --- | --- | --- | --- |
| BosphorusSign22k (Özdemir et al. 2020), General subset | The dataset owners' Google Drive folder as of 2026-11-24, without depth.tar.gz: 68,423 files, 86,376,739,485 bytes. Experiments use the rows of BosphorusSign22k.csv with SubsetID 'General': 5,788 videos, 174 classes; per video the Kinect v2 skeleton file and the RGB video. | Shared with the study by the dataset owners through a Google Drive folder; exported through the Drive web interface and uploaded by data.sh. Queue dataset record yv6ary4mdg8kw7q names the earlier BosphorusSign corpus; the dataset owners state it is a subset of BosphorusSign22k. | Bosphorus Sign Dataset End User License Agreement (SHA-256 b7922c66dbcaed30453eeeac0664bc3b3384479bf2f25954ba8fa244fe96fda6): internal non-commercial research, no transfer to third parties, no corpus portions in publications. Gate bsign22k-data-cloud-use records the permission to process it in the study's Modal workspace. | `/datasets/bosphorussign22k` |
| Rendered Handpose Dataset v1.1 (Zimmermann and Brox 2017) | RHD_v1-1.zip, SHA-256 e04310f19ccb0429c999efaf2a1e2a2b046f3024cada6794585a08fb92fb876f; the 41,258 training samples (color image and part mask). | https://lmb.informatik.uni-freiburg.de/data/RenderedHandpose/RHD_v1-1.zip, downloaded on Modal by data.sh. | Dataset page: 'provided for research purposes only and without any warranty. Any commercial use is prohibited.' Gate hand-datasets-cloud-use. | `/datasets/rendered-handpose-dataset` |
| HGR hand gesture database, series HGR1 (Kawulok et al. 2014) | 899 images with binary skin masks (downscaled set). | Already in the shared Volume (imported 2026-09-10 by another reproduction); not modified. | Database page: free to use for research, non-commercial purposes, with citation. Gate hand-datasets-cloud-use. | `/datasets/hgr1` |
| HGR hand gesture database, series HGR2A (Kawulok et al. 2014) | 85 images with binary skin masks (downscaled set). | https://sun.aei.polsl.pl/~mkawulok/gestures, downloaded on Modal by data.sh. | Database page: free to use for research, non-commercial purposes, with citation. Gate hand-datasets-cloud-use. | `/datasets/hgr2a` |
| HGR hand gesture database, series HGR2B (Kawulok et al. 2014) | 574 images with binary skin masks (downscaled set). | https://sun.aei.polsl.pl/~mkawulok/gestures, downloaded on Modal by data.sh. | Database page: free to use for research, non-commercial purposes, with citation. Gate hand-datasets-cloud-use. | `/datasets/hgr2b` |

BosphorusSign22k was verified file by file against Google Drive's own MD5 list (68,423 files, none missing, none different). `depth.tar.gz` was not acquired; the method does not use depth images. No dataset file, video frame or mask is committed or published.

(Portions of) the research in this paper used the Bosphorus Sign Dataset made available by the Computer Engineering Department of Bogazici University, Bebek, Istanbul.

- Camgöz, N. C., Kındıroğlu, A. A., Karabüklü, S., Kelepir, M., Özsoy, A. S., Akarun, L. (2016). BosphorusSign: A Turkish Sign Language Recognition Corpus in Health and Finance Domains. LREC 2016.
- Özdemir, O., Kındıroğlu, A. A., Camgöz, N. C., Akarun, L. (2020). BosphorusSign22k Sign Language Recognition Dataset. LREC 2020 Workshop on the Representation and Processing of Sign Languages.
- Zimmermann, C., Brox, T. (2017). Learning to Estimate 3D Hand Pose from Single RGB Images. ICCV 2017.
- Kawulok, M., Kawulok, J., Nalepa, J., Smolka, B. (2014). Self-adaptive algorithm for segmenting skin regions. EURASIP Journal on Advances in Signal Processing.

## Environment and patches

- Image: ghcr.io/sign-language-processing/reproduction@sha256:305b6165d306192996358ca312d9a751fa409f43063a76dc7758880a8f905291 plus segmentation-models-pytorch==0.5.0, simple-video-utils==0.7.4, av==18.1.1.
- PyTorch 2.11.0a0+eb65b36914.nv26.02; full `pip freeze` at `modal://kindiroglu-2019-results/full/pip-freeze-train.txt` (SHA-256 `a5252f0c8043010e2eab2476161aa1a0b1ed2e6f735fd9f4db71ca318c2bc4b6`).
- GPU: NVIDIA A10, driver 620.70.06, 23028 MiB (Modal A10G). Shared cache `huggingface-cache` mounted at `/cache/huggingface` with `HF_HOME` and `HF_HUB_CACHE` set.
- The pinned study image has no `simple-video-utils`; version 0.7.4 (the one the repository Dockerfile installs) is added, with `av==18.1.1` because PyAV 19 rejects an argument that 0.7.4 passes.
- No upstream code exists, so there are no patches. `taf.py` is the reimplementation; `python taf.py self-check` tests the keyframe and feature logic on a synthetic skeleton.

## Execution evidence

| Run | Kind | Start to end (UTC) | Terminal | Modal apps | Outcome |
| --- | --- | --- | --- | --- | --- |
| `populate-dataset-1` | data_population | 2026-11-24T08:26:32Z to 2026-11-24T08:33:07Z | stopped / invalid_run | ap-u8nVnqBtJ1VSdR0KaJ0VH7, ap-53FTl2zoWkIbbT9cg4esbf | Stopped by the agent after 6.5 minutes. rclone had copied 248 small files (about 1 file per second) and reported 163 of 10,187 queued transfers with an estimated two weeks remaining, far beyond the 24-hour ceiling. A disposable diagnostic (app ap-53FTl2zoWkIbbT9cg4esbf) copied 26 to 32 files of the same kind in 2 to 3 seconds from a fresh container, to local disk and to the Volume alike, so neither Drive downloads nor Volume writes are slow in themselves. Working hypothesis, refuted by populate-dataset-2: the per-folder listing of 22,542 folders in each of three trees consumed the Drive request quota that the transfers share. The local modal run command ended with a RemoteError traceback when the app was stopped (exit code 1). |
| `populate-dataset-2` | data_population | 2026-11-24T08:34:10Z to 2026-11-24T08:39:47Z | stopped / invalid_run | ap-KWmKFChUQcKKu57BWtCqVi | Stopped by the agent after 5.5 minutes: 221 files copied, 197 of 10,213 queued transfers, estimated 11 days remaining. The single recursive listing changed nothing, so the listing hypothesis of populate-dataset-1 is refuted. A local download of 434 files with debug logging then showed the cause: 97 responses 'rateLimitExceeded' on the quota 'defaultPerMinutePerProject' of drive.googleapis.com. The login uses rclone's shared Google client ID, whose request quota is shared by all rclone users; rclone backs off on every such response. Throughput therefore depends on a credential, not on code. Gate gdrive-client-id was opened on this evidence and later resolved as not needed: populate-dataset-3 uploads a local export of the folder instead of copying from Drive. The local modal run command ended with a RemoteError traceback when the app was stopped (exit code 1). |
| `populate-dataset-3` | data_population | 2026-11-24T10:44:02Z to 2026-11-24T17:17:24Z | succeeded / completed | ap-gYWVat4XPDjAmjPnLDIzmh, ap-HbmOhyXw2EB6RfdT9XOwKM | All 45 files uploaded without failure (2026-11-24T10:44Z to about 15:27Z). The populate function was preempted by Modal twice during hashing and restarted from the start by Modal each time; the third pass finished. 68,423 files, 86,376,739,485 bytes; no file missing, no MD5 mismatch and no file absent from Drive's list. |
| `populate-hand-datasets-1` | data_population | 2026-11-24T09:55:41Z to 2026-11-24T10:10:20Z | succeeded / completed | ap-584ACz1ICpZddsFNSx0bWX | All 13 archives downloaded from the official pages, tested, extracted and hashed. hgr2a: 85 images, 85 skin masks and 85 feature-point files in each of the downscaled and full-size sets. hgr2b: 574 of each. rendered-handpose-dataset: 123,774 training PNGs (41,258 samples x color, depth, mask) and 8,184 evaluation PNGs (2,728 samples x 3). |
| `full-1` | full | 2026-11-26T10:31:14Z to 2026-11-26T13:16:13Z | succeeded / completed | ap-qoPVQn8iIcrsSxSTJUd71N | All three stages completed without preemption: 20 segmentation epochs, masks for 5,788 videos, and 100 classifier epochs with a weight save/reload check for each of seeds 0, 1 and 2. |

`full-1` used 2.745 A10G GPU-hours and cost USD 4.94 (billed; ceiling USD 20 and 12 GPU-hours). All Modal apps of this paper, including data population and disposable preflights and probes, were billed USD 6.16 in total as read on 2026-11-26.

| Artifact | URI | SHA-256 |
| --- | --- | --- |
| `full-1-seed0-metrics` | `modal://kindiroglu-2019-results/full/classifier-seed0/metrics.json` | `e9455536407cf929c8249d48e812f5623835467b45bbb24c8d9c62e4278e12aa` |
| `full-1-seed1-metrics` | `modal://kindiroglu-2019-results/full/classifier-seed1/metrics.json` | `2fb3219573d0f6a81c23e1d922f06ed283e345e691de0cf50903e7a2103915a3` |
| `full-1-seed2-metrics` | `modal://kindiroglu-2019-results/full/classifier-seed2/metrics.json` | `3075960d7958b8513f09e09245457d3b659b564f67f46c46e0ea1f2f4bd8060b` |
| `full-1-seed0-scores` | `modal://kindiroglu-2019-results/full/classifier-seed0/scores.npz` | `8e97cb1dbb78033d0a2ae7d9fff3181229daae78561596685487cce90d35c373` |
| `full-1-segmentation-log` | `modal://kindiroglu-2019-results/full/seg.json` | `01aa5a7dae35396bc3dec853ef0555b33cf4c125a21c1a70b26da6fae9d4ebc4` |
| `full-1-pip-freeze` | `modal://kindiroglu-2019-results/full/pip-freeze-train.txt` | `a5252f0c8043010e2eab2476161aa1a0b1ed2e6f735fd9f4db71ca318c2bc4b6` |

## Guesses and deviations

| ID | Kind | Paper says | Used | Basis | Effect |
| --- | --- | --- | --- | --- | --- |
| `cnn-depth-and-dropout` | paper_internal_conflict | Sec. 3.4: '6 convolutional layers', dropout 0.5, and also '2 blocks of layers, each with 2 convolutional layers'. | 2 blocks x 2 convolutions (4 layers), 128 initial filters, dropout 0.25 after every convolution, 'same' padding, ReLU, one linear layer after global average pooling. | Table 3's row '0.25, 2, 2, 128: 78.21, 96.84' is the value Table 4 reuses as its baseline, and Sec. 4.3 says that model was used from then on. Decided at gate taf-unstated-protocol before any training run. | Unknown; could change both target values. |
| `joint-projection-and-heatmaps` | guess | Sec. 3.1.1: 3D Kinect joints, 'Gaussian uncertainties on the joint locations', 10 joints of the 'upper arms, elbows, wrists, hand tips and thumbs' plus a background heatmap of all joints. | The Kinect files' color-image pixel coordinates scaled from 1920x1080 to 116x64; Gaussian standard deviation 2 pixels; 'upper arms' are the shoulder joints; the background map is the maximum over all 25 joints. | 1920x1080 and 116x64 have almost the same aspect ratio. Decided at gate taf-unstated-protocol before any training run. | Unknown; could change both target values. |
| `seven-channels-per-joint` | guess | Table 1: 'Hue: 11 joints, 7 channels'. Sec. 3.1.2: hue = subunit, saturation = time within the unit, value = heatmap strength. Sec. 3.2: hue n x 180 / (K + 1). | Per joint, PoTion's three descriptors on an RGB rendering of the HSV image: colored sum (3), intensity (1), normalized (3). Hue n / (K + 1) of the color circle (OpenCV's 0-180 scale); saturation rises linearly within a segment. | PoTion's 2c+1 gives 7 for 3 color channels, and Figure 3 shows three rows of which one is 'obtained by normalizing'. Decided at gate taf-unstated-protocol before any training run. | Unknown; could change both target values. |
| `keyframe-detection` | guess | Sec. 3.2: local minima of the dominant hand's speed, K + 2 keyframes with the first and last discarded, 'density clustering' [40]. Table 5: 'FL-# means # keyframes'. | Hand joint speed in camera coordinates; dominant hand = longer path; density-peaks clustering on normalized (frame, speed) points with a Gaussian kernel at the 2nd-percentile distance; the 6 points with the largest density x distance; first and last dropped; 4 keyframes kept, 5 colored segments. Fewer than 6 minima: evenly spaced frames. | Decided at gate taf-unstated-protocol before any training run. | Unknown; could change both target values. |
| `static-keyshot-masks` | guess | Sec. 3.3: masks of the selected frames are 'resized, converted to binary masks and added to the TAF representation'; keyshot = 'five consecutive frames centered at the keyframe'. Table 1: 5K static frames. | Per keyshot frame one 116x64 channel: the whole-frame mask of both hands, a cell set if any hand pixel falls in it. 20 mask channels, 97 input channels. | Table 1 counts one channel per static frame. Decided at gate taf-unstated-protocol before any training run. | At 116x64 a hand covers a few cells, so little hand shape is kept; a hand crop would be a different feature. Unknown; could change both target values. |
| `hand-segmentation-training` | guess | Sec. 3.3: DeepLab v3+, three classes, Rendered Handpose plus 'the HGR dataset', 1% validation, mIoU about 78% left and 83% right. No backbone, input size, schedule, or rule for HGR's missing left/right labels. | DeepLab v3+ (segmentation-models-pytorch 0.5.0) with an ImageNet ResNet-50 backbone; 320x320; Adam 1e-4, batch 32, 20 epochs, no augmentation; HGR1, HGR2A and HGR2B; HGR skin pixels trained as 'left or right hand'; video frames segmented at 960x544. | Backbone, HGR series and the HGR loss were decided at gate taf-unstated-protocol; resolution, optimizer, epochs and the frame size are this attempt's own choices, fixed before the run. | The masks' quality on the sign videos is not measured (no ground truth). Unknown; could change both target values. |
| `classifier-training` | guess | Sec. 3.4: Adam, batch size 32, Xavier initialization, random horizontal flip. No learning rate, epoch count, checkpoint rule or seed. | Adam with Keras defaults (learning rate 0.001), 100 epochs, weights after the last epoch, no selection on the test set; seed 0 reported, seeds 1 and 2 supplementary. | Decided at gate taf-unstated-protocol before any training run. | The paper has no validation split, so its checkpoint may have been chosen on the test user; last-epoch values can be lower than a best-epoch value. |
| `pytorch-instead-of-keras` | paper_deviation | Sec. 4: 'implemented using the Keras library with a Tensorflow backend'. | PyTorch on the study base image, with Keras defaults mirrored (Glorot-uniform weights, batch-norm epsilon 1e-3 and momentum 0.99, Adam epsilon 1e-7). | Decided at gate taf-unstated-protocol before any training run. | Numerical differences between frameworks are expected to be small against the other guesses. |

## Noncompliance

None.

## Attempts, failures, and dead ends

- **Data transfer.** Two direct Drive-to-Modal copies (`populate-dataset-1`, `-2`) ran at about one file per second because Google rate-limits the copy tool's shared client ID. The folder was instead exported through the Drive web interface and uploaded from a local copy (`populate-dataset-3`), which Modal preempted twice before the third pass completed.
- **Preflights (disposable, not retained as runs).** A first mask preflight failed because the pinned image lacks `simple-video-utils`; a second failed because PyAV 19 is incompatible with version 0.7.4. Both are fixed by the two pins above. The final preflight ran segmentation, masks and classifier on small subsets, including the weight save/reload check.
- **No tuning.** No setting was changed after seeing a test score. The single full run is the only one.

## Candidate flags, ethics, and human evaluation

The queue record flags no ethical concern and no human evaluation; none was performed. The data are videos of identifiable signers, used under the dataset's licence agreement for internal non-commercial research; no video, frame or mask leaves the study's Modal workspace. The queue lists the dataset as "BosphorusSign"; the dataset owners state that it is contained in BosphorusSign22k, and the General subset's counts equal the paper's.

## Author and team contact

After run `full-1` had finished, a study team member sent the authors 25 questions on 2026-11-26, covering every guess listed above and asking for the original code. No result of this attempt was disclosed. No reply as of 2026-11-26, so the contact has changed nothing yet. The dataset owners gave the study access to the data and confirmed the permission to process it in the study's cloud workspace (gate `bsign22k-data-cloud-use`).
