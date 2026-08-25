# Intelligent Signs Language Understanding with Autonomous Landmarks for E-learning Context

**Paper ID:** `8526aecd1407305d815883725a864405e31a54c1`  
**Citation:** Muhammad Jamil Hussain and Ahmad Shaoor. *2022 19th International Bhurban Conference on Applied Sciences and Technology (IBCAST)*, pp. 219-224. DOI: [10.1109/IBCAST54850.2022.9990143](https://doi.org/10.1109/IBCAST54850.2022.9990143).  
**Preference level:** `3` — no author implementation was found after a documented source search.  
**Pipeline status:** `insufficient_information`  
**Numerical agreement:** `not_assessed`

## Scope and target contract

The confirmed assignment says “TABLE III.” Table III is a comparison table with one `Accuracy (%)` column. Nine rows are copied comparisons, so they are retained in [`reproduction.json`](reproduction.json) but are not reimplemented as part of this paper. The two paper-owned rows are the proposed MediaPipe-landmark + Random Forest method:

| Dataset | Table III system | Paper accuracy | Status |
| --- | --- | ---: | --- |
| ASL Alphabet | Proposed method / Random Forest | 98.68% | Not produced |
| ISL-HS | Proposed method / Random Forest | 98.76% | Not produced |

The target paper does not state the Table III split, whether the number is a 10-fold mean or a hold-out result, seed, feature-reduction rule, frame-to-video aggregation, or metric implementation. It reports 10-fold *learning curves*, but does not tie that procedure to the table. Its detailed tables round Random Forest accuracy to `0.987` on both datasets, which cannot uniquely yield Table III's 98.68% and 98.76%. A run with an invented protocol would therefore be a conditional experiment, not a faithful Table III reproduction.

The complete Table III ledger, including the copied baselines and their terminal `not_produced` status, is in [`reproduction.json`](reproduction.json).

## Re-read protocol

The author-uploaded [full text](https://www.researchgate.net/publication/361570854_Intelligent_Signs_Language_Understanding_with_Autonomous_Landmarks_for_E-learning_Context) specifies the following.

- OpenCV passes each image/frame to MediaPipe’s palm detector and joint locator, producing 21 indexed **x/y** hand landmarks (Section II.A).
- For every ordered pair of distinct landmarks, it computes `s = (y_j - y_i) / (x_j - x_i)` and `atan(s)`: 21 × 20 = 420 angle features (Section II.B, Algorithm 1). These are ordered pairs; reducing them to 210 unordered pairs would change the stated algorithm.
- It saves the five finger slopes from landmark pairs `(0,4)`, `(5,8)`, `(9,12)`, `(13,16)`, and `(17,20)`, then computes the ordered pairwise line value `abs((s_j - s_i) / (1 + s_i*s_j))`: 5 × 4 = 20 more features (Section II.B, Algorithm 1). The raw vector is therefore 440 features per processed frame.
- It says only that “more than half” of the features are removed by correlation/dimensionality reduction. It does not identify a reduction algorithm, threshold, retained IDs/count, or whether fitting occurs inside each split.
- Its Random Forest statement establishes only the default 100 trees; no seed, sklearn version, other forest settings, or tuning protocol is reported (Section II.C).
- ISL-HS has 26 classes × 18 videos; only the first 60 frames of each video are used to limit landmark-orientation variation (Section III.B). The paper does not say whether frame features are classified individually, pooled per video, or split by video/person.
- Ten-fold learning curves are plotted for Random Forest, but the paper does not say that Tables I–III are 10-fold means or define folds, grouping, shuffling, or a seed (Section III.C).

`pose-format==0.14.1` can run a MediaPipe **Holistic** extractor, but it does not provide a pure 21-point MediaPipe Hands extractor. Using it to detect the paper’s landmarks would silently replace the stated estimator. The reconstruction will therefore use MediaPipe Hands directly and may use pose-format only to inspect or serialize already-extracted landmarks, where that adds evidence without changing coordinates.

The paper describes ASL Alphabet as 87,000 200×200 colour images and “28 gestures.” The cited source instead has 29 class directories—A-Z, SPACE, DELETE, and NOTHING—and 87,000 = 29 × 3,000. The inclusion decision is not published.

## Conditional ISL-HS run decisions

The user authorized a documented attempt to obtain the expected result despite the unpublished Table III evaluation details. The following are reconstruction decisions, not claims about the authors’ setup:

| Missing detail | Conditional decision | Why |
| --- | --- | --- |
| Landmark implementation | `mediapipe==0.10.18`, direct `Hands`; sequential tracking within each video; one hand; model complexity 1; both confidence thresholds 0.5 | Preserves the stated 21-point Hands path. The paper gives no version or configuration. |
| Video decoding | `simple-video-utils==0.7.4`, RGB display-oriented frames | Study-wide decoder policy replaces the paper’s unspecified OpenCV decoding details. |
| Landmark coordinates | Multiply MediaPipe's normalized `x/y` by the decoded frame width/height | The paper describes pixel coordinates. |
| Zero division | Preserve vertical-slope `atan(±∞)`; replace indeterminate/non-finite feature values with 0 | The equations give no zero-division policy. |
| Feature reduction | Greedy absolute-Pearson correlation filter, threshold 0.95, fit on each training fold only | The paper only says more than half the 440 features are correlation/dimensionally reduced. This avoids test-fold leakage. |
| Random Forest | 100 trees, seed 2026, eight CPU workers, all other `scikit-learn==1.6.1` defaults | Only the 100-tree default is stated. |
| Evaluation | Run shuffled 10-fold frame-stratified CV (seed 2026) and unshuffled 10-fold video-grouped CV; report both | Frame CV may be closest to an unspecified frame-level implementation but leaks video siblings; grouped CV is the leakage audit. Neither is silently selected as Table III. |

The real-data preflight uses two videos/class and two folds to exercise decoding, detection, the exact 440-feature extractor, fold-local reduction, fitting, and both evaluators. The full conditional run uses all 18 videos/class and ten folds.

### Retained preflight evidence

The corrected preflight decoded the first 60 frames from 52 videos (two per
class) and MediaPipe detected a hand in all 3,120 frames. Its seeded,
frame-stratified two-fold result was **99.97% ± 0.05%**; strict video-grouped
two-fold CV was **87.12% ± 4.17%**. This sharp gap is evidence that random
frame splitting lets closely related frames from the same video enter both
train and test. It is not evidence that the paper used either protocol.

The immutable output is
`modal://volume/8526aecd-landmark-results/preflight-frame-shuffled/run.json`
(SHA-256 `8641f9b7f4ca297b924897fc0a704732b696550c0834372fec60061856efb9d2`),
from Modal app `ap-Vw43983kOrGHMepFeFRB4c`, function call
`fc-01M0VSBGR8E7CVJ6NCC137N3W1`. An earlier unshuffled preflight is retained as
diagnostic evidence rather than a result: the video-contiguous source order
made its nominal frame and grouped folds identical. The smallest correction was
to shuffle only the frame-level splitter with the recorded seed; the grouped
split is unchanged.

## Data gates

The required `datasets` and `huggingface-cache` Volumes exist in Modal `repro-sign`. ISL-HS is populated below; ASL Alphabet is still absent.

| Dataset | Authoritative source | Permission status | Required action |
| --- | --- | --- | --- |
| ASL Alphabet | [Kaggle grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet), v1 | Kaggle metadata declares GPL-2.0; account terms and project-cloud processing have not been verified | Resolve its 28/29 class choice and terms before population at `datasets/asl-alphabet` |
| ISL-HS | [marlondcu/ISL at `d1d50bb`](https://github.com/marlondcu/ISL/tree/d1d50bb65540b904e3e0a6ffe0997872c4e9e645) | The repository has no published license; the user explicitly authorized this study's project-cloud use on 2026-08-25 | Populated and validated at `datasets/isl-hs`; do not redistribute data or derivatives without separate permission |

The committed `datasets/isl-hs/manifest.json` records six source archives, all 468 extracted `.mov` files, and the pinned source revision. Its SHA-256 is `d8a278a87aa05898159e848d5f6c206364d0af74af84d3ea88e7d5c34f58e9b5`; its deterministic relative-video-path hash is `00db8120a603ae8d1a2896aff7ed9f1e68e77662066f5f42ac3b9c0ea71b9d76`. ISL-HS has not been decoded, evaluated, or published. No ASL data, predictions, or trained weights have been acquired, produced, or published.

## Source search

The canonical [IEEE record](https://ieeexplore.ieee.org/document/9990143/) and the proceedings [table of contents](https://www.proceedings.com/content/066/066913webtoc.pdf) identify the paper. IEEE required login for PDF retrieval and the author-upload endpoint rejected retrieval on 2026-08-24, so no error response was treated as a paper PDF or hashed.

The paper contains no code release. Exact-title/method searches across GitHub, Zenodo, OSF, Hugging Face, and author accounts found no author code, archive, model, or supplement. The two author accounts inspected contained no relevant repository. The related CMC article above is openly available but is a distinct work with different reported results.

## Next faithful step

The ISL-HS data gate is resolved. The remaining Table III protocol and ASL class/terms gates must be resolved before a target-result run. Then the minimal implementation is a CPU-only, pinned MediaPipe/OpenCV/scikit-learn pipeline: verify dataset manifests, extract the stated landmarks/features, run a real-data preflight, then evaluate only the authorized protocol. ISL video decoding will use `simple-video-utils`; no video decoding is currently performed.

To populate the authorized ISL-HS source idempotently:

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run \
  papers/ahmad-2022-intelligent-landmarks/modal_app.py::populate_isl_hs
```

Then run the preflight and the full conditional evaluation through the same `repro-sign` wrapper:

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run \
  papers/ahmad-2022-intelligent-landmarks/modal_app.py::preflight
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run \
  papers/ahmad-2022-intelligent-landmarks/modal_app.py::evaluate_isl_hs
```

Every Modal operation will continue to use the `repro-sign` wrapper and mount the shared `huggingface-cache` volume. There are no runs, patches, artifacts, author contacts, human participants, or model publications yet.
