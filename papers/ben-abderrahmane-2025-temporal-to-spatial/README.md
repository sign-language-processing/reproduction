# Temporal-to-Spatial ArSL ResNet reproduction

**Summary.** The task is unusual: the paper classifies whole sentence videos into 50 fixed classes, which is neither isolated sign recognition nor continuous recognition, so its numbers are not comparable with either literature. No code was published, so this is a clean-room reimplementation from the paper and its cited prior work. The paper never states which ten frames form a sample, and the two defensible readings differ by 25 accuracy points, so no target is reproduced and the status is `insufficient_information`. Under the reading that follows the paper's own cited source, all four Table III metrics land within 0.0113 of the published values; under the paper's literal wording, accuracy falls 25 points short.

**Paper ID:** `8b224ecbd42da766efba45d17b34b3a1255c2345`

**Citation:** Ben-Abderrahmane, H.; Oulad-Naoui, S.; Cherif, A.; Chagha, A. Temporal-to-Spatial Transmutation for Enhancing Arabic Sign Language Recognition via ResNet. In *2025 7th International Conference on Pattern Analysis and Intelligent Systems (PAIS)*, IEEE, 2025. doi:10.1109/PAIS66004.2025.11126496.

**Paper:** https://doi.org/10.1109/PAIS66004.2025.11126496 · **Code/artifacts:** none found; see Source provenance.

**Preference level:** 3

**Pipeline status:** `insufficient_information`

**Numerical agreement:** `not_assessed` — the paper does not state which ten frames form a sample, so no run reproduces its protocol; conditional values are reported below

**Attempt dates:** 2026-08-28 to 2026-09-21, rebased onto `8d069b1`

## Reproduction agents

| Agent ID | Model | Application | Contribution |
| --- | --- | --- | --- |
| `reproduction-agent` | Claude Opus 5 (1M context), `claude-opus-5[1m]` | Claude Code 2.1.12 | Every stage, including both runs |

Identity as reported by the live session; the version is the installation on the executing machine, reached through its VS Code extension. Both runs record `agent_ids: ["reproduction-agent"]`.

## Scope and target contract

The portal requests **Table 3** — TABLE III (Model Comparison), whose columns give the proposed ResNet (0.9777, 0.9790, 0.9777, 0.9773) and an LSTM comparison (0.8875, 0.8930, 0.8875, 0.8867). All eight are in `reproduction.json.targets`. No machine-readable portal export existed, so the record was transcribed into `assignment.normalized`; `assignment.kind` stays `direct_user_request`, and four radio fields that did not render are `null` rather than guessed.

§IV states the paper "adopted the same training parameters as our previous work[19]", so [19] (Oulad-Naoui et al., ICEIS 2024) and the ArabSign dataset paper were pinned; they supply most of what this paper omits.

**Table III's LSTM column is a copied baseline.** [19] reports 88.75% test accuracy, exactly Table III's 0.8875, and §II's "88.5%" is [19]'s own abstract figure. [19] publishes accuracy only, so the LSTM precision, recall, and F1 are unsourced.

**Labels:** one sample per video, labelled by sentence — [19] converts "the input video into a sequence of ten frames", and SentenceID is a directory level in the archives. No gloss alignment is involved. **Which ten frames is undetermined**, and it blocks the reproduction; see Results and gate `frame-window-ambiguity`.

**The task is not ISLR in the usual sense.** ArabSign is a continuous corpus of sentences, benchmarked by its authors with WER and BLEU over gloss sequences. This paper instead treats each whole sentence video as one example of one of 50 fixed classes: the model never segments, never identifies individual signs, and never produces language — it picks a sentence label. That is closed-set classification, not isolated sign recognition, which labels individual signs from a vocabulary, and not continuous recognition, which outputs gloss sequences. Table III's figures are therefore not comparable with either literature, including ArabSign's own WER 0.50 benchmark on the same data.

Inconsistencies in the paper, none affecting the targets: the abstract claims "91.3% of accuracy" where Table III says 0.9777, while §IV's "nearly 10% improvement" matches Table III (0.9777 − 0.8875); Table III gives LSTM training time as 3 h where §IV says "eight hours", and that time ran on [19]'s RTX 4060 against this paper's CPU-only Ryzen 3; and Table II's frame count is copied from [19] and mislabelled (see Data provenance).

### Extracted protocol contract

| Element | Value | Determined? |
| --- | --- | --- |
| Keypoint extractor | MediaPipe pose, 33 landmarks, x and y (§III) | variant not given |
| Sample construction | 10 frames → `(10, 33, 2)`, one per video (§III, [19]) | yes, but **not which ten** |
| Labels | 50 sentence classes; SentenceID is a directory level | yes |
| Split | Official directories in the archives: 7,489/1,841 | **no** — no paper references them |
| Optimizer / loss | Adam, lr 0.001 [19]; categorical cross-entropy (§IV) | yes |
| Batch / epochs | 32 / 100 (§IV) | yes |
| Framework / hardware | Keras on TensorFlow, CPU only (§IV) | yes, no versions |
| Padding / strides | `same`, stride 2 in blocks 2–3 — forced by (10,33)→(5,17)→(3,9) | by arithmetic |
| Architecture | 755,346 parameters, see below | by arithmetic |
| Metric implementation | weighted averaging, confirmed by Recall≡Accuracy | by arithmetic |
| Initialisation, seed | not stated | **no** |
| Augmentation, checkpoint rule | none; fixed 100 epochs, final-model metrics | by absence |

**Table I's rows are inconsistent; its total determines the architecture.** The rows sum to 689,490 against a stated 755,346, blocks 2 and 3 having been counted at their output widths rather than their declared inputs. One natural design reaches the stated total exactly — two 3×3 convolutions per block, **no batch normalisation**, 1×1 projection shortcut on width or stride change — the 65,856 gap being those projections plus the correct widening convolutions. The block internals are therefore determined, not guessed, and the implementation asserts 755,346 before training. Figure 2 labels block 3 `(3,8,128)` against Table I's `(3,9,128)`; the 3,456 flatten is 3×9×128, so the figure is wrong.

## Source provenance

| Artifact | Canonical source | Pinned revision / SHA-256 | Role |
| --- | --- | --- | --- |
| Paper PDF | doi:10.1109/PAIS66004.2025.11126496 | `0afdb6457dbfe7300d6a2c883d06676527bd39ecb658e59c17f48514db9d147c` | Targets and disclosed protocol |
| Prior LSTM paper [19] | doi:10.2991/978-94-6463-496-9_24 | `c0d48f5e3c2a1ef2c560a7dd2bf36a41f6be7c9207648904720944b1a101a95f` | Inherited preprocessing, split, optimizer, and the copied accuracy |
| ArabSign dataset paper | arXiv:2210.03951 | `7be6d45db9a17116201957bae100940d95b0f7fe36cf587b59b6aaa5ea94ccc8` | Dataset composition; documents no split though the distribution ships one |
| ArabSign data/code | github.com/Hamzah-Luqman/ArabSign | `8f6e127a054052228027d6da98005f991d300a7b` | Project page; no data, no licence. Links the Drive folders used |
| Published code, weights, supplements | none found | — | See below |

The target PDF is IEEE-watermarked and subscription-only; [19] is open access CC BY-NC 4.0; ArabSign is on arXiv. All retrieved 2026-08-28, referenced by hash, not committed.

**Independent source search, 2026-08-28: no code exists, hence preference level 3.** Nothing found in either paper including footnotes and availability statements, the IEEE Xplore and Atlantis Press landing pages, web searches on the title and author names against Zenodo, OSF and GitLab, or GitHub repository and user searches for `arabsign`, `ben-abderrahmane`, `oulad-naoui`, and `temporal-to-spatial sign language`.

## Results

**No target was produced.** The two defensible readings of the 10-frame window differ by 25 accuracy points, so the choice is behaviour-changing and every run is conditional evidence rather than a reproduction of the published protocol. Blocker `protocol_ambiguous`, gate `frame-window-ambiguity`.

| Target ID | Metric | Original | Conditional (uniform) | Difference | Terminal reason |
| --- | --- | ---: | ---: | ---: | --- |
| `table3-resnet-accuracy` | Accuracy, weighted | 0.9777 | 0.988593 | +0.010893 | `not_produced` / `protocol_ambiguous` |
| `table3-resnet-precision` | Precision, weighted | 0.9790 | 0.988908 | +0.009908 | `not_produced` / `protocol_ambiguous` |
| `table3-resnet-recall` | Recall, weighted | 0.9777 | 0.988593 | +0.010893 | `not_produced` / `protocol_ambiguous` |
| `table3-resnet-f1` | F1, weighted | 0.9773 | 0.988585 | +0.011285 | `not_produced` / `protocol_ambiguous` |
| `table3-lstm-*` (4 rows) | all four metrics | 0.8875 / 0.8930 / 0.8875 / 0.8867 | — | — | `copied_baseline`; accuracy equals [19]'s 88.75% exactly, the rest unpublished in [19] |

| Sampling | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Paper, Table III | 0.9777 | 0.9790 | 0.9777 | 0.9773 |
| Uniform across clip (`train-uniform10`) | 0.988593 | 0.988908 | 0.988593 | 0.988585 |
| First 10 consecutive (`train-first10`) | 0.722977 | 0.730843 | 0.722977 | 0.721519 |

Both readings were run before either was selected, with the decision rule recorded first. The literal reading yields a different experiment, not a near miss; the uniform reading, which follows [19] and which §IV incorporates by reference, lands within 0.0113 of every published value. Neither can be entered as a target result.

Two things were confirmed rather than assumed, and hold independently of the window: weighted recall equals accuracy in both runs, the identity Table III shows in both columns and which holds for weighted but not macro or micro averaging; and the model was asserted at 755,346 parameters before the first step.

Both runs memorised their training sets — 0.9844 and 1.0000 training accuracy against 0.7230 and 0.9886 test. ArabSign gives ~150 near-duplicate takes per class from the same six signers and sessions, with signers coached to minimise variation, so these numbers measure recall of same-distribution takes rather than generalisation to new signers. That is a property of the paper's protocol, not of this reproduction.

**No weights are published or retained.** The study publishes to [`repro-sign`](https://huggingface.co/repro-sign) only "when their license permits redistribution", and ArabSign has no licence; the basis here covers *use*, not redistribution of a model that memorised video of six identifiable signers. The model regenerates in 331 s from the committed code, the pinned features, and seed 2026; raw predictions and per-epoch history are in `runs/uniform10/history.npz`.

## How to repeat this

From a fresh checkout, authenticated to Modal workspace `repro-sign`:

```bash
./setup.sh
W=.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh
A=papers/ben-abderrahmane-2025-temporal-to-spatial/modal_app.py

$W run $A::populate                  # populate datasets/arab-sign/RGB
$W run $A::extract --mode uniform10  # MediaPipe pose -> pose-uniform10/*.npz
$W run $A::train   --mode uniform10  # train, evaluate, write runs/uniform10/
```

`::train --mode uniform10` reaches every target metric; `--mode first10` reproduces the rejected alternative. Each step is idempotent. `::verify` audits the archives, `::summary --mode MODE` the feature set, `::smoke --mode MODE` the full path on 24 videos without writing. Inputs are Volume `datasets` read-only at `/datasets` plus `huggingface-cache` at `/cache/huggingface` as the workspace requires; outputs go to Volume `ben-abderrahmane-2025-results`, all SHA-256 pinned in `reproduction.json.artifacts`.

## Data provenance and permissions

| Dataset | Version/subset/splits | Source and access date | License/permission and cloud-use basis | Path in Volume `datasets` | Counts / manifest / checksum | Deviations |
| --- | --- | --- | --- | --- | --- | --- |
| ArabSign | RGB modality; official directories 7,489/1,841 | Public links on the author's project page, 2026-09-18 | Publicly available; **no licence published anywhere**, so permission rests on the author's direct grant | `arab-sign/RGB` | 9,330 mp4; manifest `645e22de…`; per-archive SHA-256 | 5 videos absent from the RGB release |

Gate `arabsign-access` is resolved; `check_modal_dataset.sh arab-sign _drive_imports/72178996f6c31e6c/manifest.json` confirms the path.

**Availability and permission.** The dataset is publicly available — the author's project page publishes direct Drive links for all three modalities, and the RGB archives were downloaded from them without a request. What is missing is **licence information, not access**: a 2026-09-18 search found no licence, terms, consent, ethics, or IRB statement in the dataset paper, in the 60 paths of `Hamzah-Luqman/ArabSign` (API reports `"license": null`), on the project page, or on the author's KArSL page — only a citation request. The permission basis is the author's direct grant: the assignee contacted Hamzah Luqman (KFUPM), who supplied the official download page for REPRO-SIGN use, and authorised study use and project-cloud processing on that basis. The copy stays private to the workspace; no imagery is redistributed or published.

**Acquisition and verification.** `arab-sign/RGB` holds the six signer archives, 19,622,200,244 bytes, each matching its source `Content-Range` exactly, transferred 2026-09-18 (`fc-01M2S6VRWS8D0ZVVXQ5RF5YKW2`); an earlier Team S transfer supplied Skeleton and partial Depth but no RGB video. `::verify` finds 9,330 mp4 entries, 50 sentence IDs, zero unparsed paths, all matching `signer/split/sentenceID/signer_sentence_(timestamp)_c.mp4`, so labels and split both come from the path. All three `ArabSignGroundTruth.txt` copies hash to `2b64f16bb14859ca…`.

**Five missing videos.** RGB holds 9,330 against the 9,335 both papers claim. Skeleton holds exactly 9,335 and the present Depth archives match it per signer, so 9,335 is right for the dataset and the RGB release is short; RGB is a strict subset with no extras. Absent: `01/test/0006/…(16_02_21_19_36_11)`, `01/test/0044/…(20_03_21_20_09_50)`, `01/train/0020/…(17_03_21_21_36_28)`, `02/train/0020/…(02_03_21_21_01_08)`, `05/train/0042/…(20_03_21_19_25_26)`. Effective counts 7,489/1,841 against Skeleton's 7,492/1,843 — 0.05%, immaterial but recorded. **Depth remains incomplete**, missing `01.7z` and `05.7z`.

ArabSign is 9,335 samples of 50 sentences by 6 signers, ≥30 repetitions each, Kinect V2 colour at 1920×1080/30 fps, clips 1.3–10.4 s. Its 25-joint skeleton stream is unused, since the paper runs MediaPipe over colour frames. Its "around 200,000 frames" is **per signer**, reconciling Table II's apparent error: 9,335/6 ≈ 1,556 × 130.3 ≈ 203,000. All signers are male, 21–30 — identifiability is addressed at gate `arabsign-signer-consent`.

## Environment and patches

**No patches**: preference level 3, so there is no upstream source to patch.

| Purpose | Image |
| --- | --- |
| Transfer and archive audit | `debian_slim` (3.12) + `p7zip-full` |
| Pose extraction | root `Dockerfile` + `p7zip-full` + `mediapipe==0.10.18` — the study image supplies `simple-video-utils` 0.7.4, the mandated decoder; the pin matches `papers/ahmad-2022-intelligent-landmarks` |
| Training | `tensorflow/tensorflow:2.17.0` + `scikit-learn==1.6.1` — §IV states Keras on TensorFlow, CPU; the pin matches `papers/mohamed-2024-densenet121-hho` |

All work was CPU-only, matching the paper. Extraction used six containers of 32 CPUs; training used 16 and took 331 s against the paper's reported 30 minutes. Pose estimation uses `model_complexity=1`, which [19] calls "the standard version of the pose model", with `static_image_mode` off for contiguous frames and on for uniformly sampled ones.

## Execution evidence

Both runs: profile `repro-sign`, environment `main`, CPU only, seed 2026, agent `reproduction-agent`, attempt 1 of 1, exit 0, terminal `succeeded`/`completed`, ceiling 21,600 s wall.

| Run ID | Targets | Hardware | Start/end UTC | Artifacts |
| --- | --- | --- | --- | --- |
| `train-uniform10` | all four in-scope | 16 CPU, 0 GPU | 03:36:40 → 03:42:11, 331 s | `fc-01M310JMS6YZFNCFK8J4AHZM56`; `run-uniform10`, `history-uniform10` |
| `train-first10` | rejected alternative | 16 CPU, 0 GPU | 03:36:33 → 03:47:40, 667 s | `fc-01M310JC95SP6Z72JB8KYDMJVT`; `run-first10`, `history-first10` |

Preceding jobs, for traceability rather than retained runs: the RGB transfer `fc-01M2S6VRWS8D0ZVVXQ5RF5YKW2` and the two pose extractions, whose twelve `.npz` outputs are SHA-256 pinned.

**Preflight deviation.** Extraction was preflighted on 24 videos per mode via `::smoke`. Training was not: the full run is 331 s of CPU, so a preflight would cost nearly what it de-risks, and the parameter assertion fires before the first optimizer step.

## Guesses and deviations

| Detail | Paper says | This attempt used | Effect on interpretation |
| --- | --- | --- | --- |
| **Frame sampling** | §III "10 consecutive frames" vs [19] "a sequence of ten frames"; neither states a window | Ten indices evenly spaced across each clip, selected at gate `frame-window-ambiguity` | **The blocker.** Behaviour-changing by demonstration: the readings differ by 25 accuracy points, so no target is produced |
| Window position | Not stated | n/a under uniform sampling; the rejected run used frames 0–9 | Would have been a second undocumented choice under the consecutive reading |
| Metric averaging | Not stated | Support-weighted over 50 classes | **Confirmed**, not guessed: our runs reproduce Table III's Recall≡Accuracy identity |
| Coordinates | "x and y coordinates"; no scale | MediaPipe's native normalisation | Values reach 0.118–1.806 where MediaPipe extrapolates past the frame; ~⅓ of landmarks are extrapolation under upper-body framing |
| Train/test split | Target silent; [19] "the common 80/20"; ArabSign documents none but ships one | Official directories, 7,489/1,841 | Not established either paper used them. Signer-dependent by design, so no signer-independent claim follows |
| Undetected pose | No policy stated | Nearest detected pose, carried forward or backward | 15 frames of 93,300 under uniform, 1 under consecutive. Immaterial |
| Seed | Not stated | 2026, single run | The paper reports single values, not mean ± std, so it ran once |
| Versions | No versions given | TF 2.17.0, sklearn 1.6.1, MediaPipe 0.10.18, simple-video-utils 0.7.4 | Differences from the authors' unstated environment are unquantifiable |
| Sample count | Both papers claim 9,335 | 9,330 | 0.05%; cannot materially affect a metric |

## Attempts, failures, and dead ends

**Undetected pose frames.** Extraction was written with no fill policy so the case would surface rather than be masked; it raised on `03/train/0046/…(09_04_21_22_08_19)` frame 7. The fill was added only then, and the measured rate — 1 frame in 93,300 consecutive, 15 uniform — confirmed no policy was load-bearing.

**Three conclusions were recorded and later corrected**, each by evidence: Table I's architecture was called unrecoverable until a conventional block form hit its stated total exactly; "ArabSign publishes no official split" holds for the paper but not the distribution; and the dataset was reported absent under slug `arabsign` before being found at `arab-sign`, though RGB was genuinely missing either way. A separate detection-measurement pass was written and discarded once extraction reported the same statistics.

## Candidate flags, ethics, and human evaluation

The portal's ethics, human-evaluation, and copied-score radio values did not survive transcription and are `null` — a gap in our copy, not a clearance. The assignee raised `copied_scores` with the annotator; that query is outstanding and changes no recorded result. The paper reports no human evaluation.

**Ethics gate `arabsign-signer-consent`, resolved 2026-09-21.** ArabSign is colour video of six identifiable signers with no consent basis documented anywhere. Access, processing, storage, and reporting rest on the direct grant of the author, who collected the corpus and is its data controller; the reproduction adds no participants, keeps its copy private, redistributes nothing, and publishes no imagery, every retained artifact being a pose array, run manifest, or metric history. The residual unknown is the original recording's consent basis, which no reproduction of this paper can establish. If the study later publishes weights or derived imagery for ArabSign, those terms should be obtained from KFUPM first.

## Author and team contact

**Dataset author, for data access.** The assignee contacted Hamzah Luqman (KFUPM), who supplied the official download page for REPRO-SIGN use; reported 2026-09-18. Team S had separately obtained a copy from him, transferred 2026-09-10. Data-access requests are exempt from the post-attempt restriction, and the exchange concerned access only — no methodological help was requested or received.

**Target paper authors.** Not contacted. The independent attempt is now complete, so contacting them is permitted and is the action that would resolve gate `frame-window-ambiguity`: only they can state which ten frames Table III used. Any reply would have to be reported here.
