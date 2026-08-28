# Intelligent Signs Language Understanding with Autonomous Landmarks for E-learning Context

**Paper ID:** `8526aecd1407305d815883725a864405e31a54c1`  
**Citation:** Muhammad Jamil Hussain and Ahmad Shaoor. *2022 19th International Bhurban Conference on Applied Sciences and Technology (IBCAST)*, pp. 219-224. DOI: [10.1109/IBCAST54850.2022.9990143](https://doi.org/10.1109/IBCAST54850.2022.9990143).  
**Preference level:** 3

No author implementation was found after a documented source search.

**Status:** `partial`
**Numerical agreement:** `not_fully_reproduced`

## Scope and target contract

The confirmed assignment says “TABLE III.” Table III is a comparison table with one `Accuracy (%)` column. Nine rows are copied comparisons, so they are retained in [`reproduction.json`](reproduction.json) but are not reimplemented as part of this paper. The two paper-owned rows are the proposed MediaPipe-landmark + Random Forest method:

| Dataset | Table III system | Paper accuracy | Status |
| --- | --- | ---: | --- |
| ASL Alphabet | Proposed method / Random Forest | 98.68% | 99.17% ± 0.10% (conditional) |
| ISL-HS | Proposed method / Random Forest | 98.76% | 97.96% ± 0.94% (conditional, video-grouped CV) |

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

The paper describes ASL Alphabet as 87,000 200×200 colour images and “28 gestures.” The cited source instead has 29 class directories—A-Z, SPACE, DELETE, and NOTHING—and 87,000 = 29 × 3,000. The entire source release, including NOTHING and its test files, is retained on Modal. The user resolved the *training* ambiguity on 2026-08-25: train/evaluate with A-Z, SPACE, and DELETE (28 classes / 84,000 training images), excluding NOTHING. This is a documented reconstruction choice, not a claim that the paper identifies NOTHING as the omitted class.

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

## Conditional results

### ASL Alphabet

The full 28-class run used A-Z, `space`, and `del`, retaining but excluding
`nothing`. Of 84,000 images, MediaPipe detected 63,673 hands (75.80%).
Shuffled 10-fold image-level CV yielded **99.1692% ± 0.0961%**, +0.4892 points
from Table III's 98.68%. This is conditional because the paper does not state
its split, missing-detection handling, or exact 28-class definition.

Raw output: `modal://volume/8526aecd-landmark-results/asl-alphabet-conditional/run.json`
(SHA-256 `f0f5ee4363d6d9a5b4df543e2d62f4543eb27bdd2d56fd1cfa456e6736ce0cc6`),
from Modal app `ap-5urNHhjWI9XM0mCt1BRPq5`. A two-fold, ten-images-per-class
real-data preflight detected 238 of 280 images and scored 94.5378% ± 1.7826%;
the full result above—not this deliberately small preflight—is the reported
conditional result. Its raw output is
`modal://volume/8526aecd-landmark-results/asl-alphabet-preflight/run.json`
(SHA-256 `784e58a9fb41dfc8b60c1fc5e4a6a89eaf045b10382b40ea2c05f9b67a4e4a5f`).

### ISL-HS

The full 468-video run decoded 28,080 requested frames and detected landmarks
in 28,071 (99.968%). Its output is
`modal://volume/8526aecd-landmark-results/isl-hs-conditional/run.json`
(SHA-256 `c917c6577b862bbd6228c7d44e7a80b417034bfc676123192d338ac0ff9f86ff`),
from Modal app `ap-lJNFft0jeNJaEp5qGW3E0g`, function call
`fc-01M0VSFT8XTC04A012ABDWXW5A`. It completed in about 17 minutes of eight-CPU
Modal time and retained its raw JSON only on the results Volume.

| Evaluation interpretation | Accuracy (mean ± fold SD) | Relation to Table III (98.76%) |
| --- | ---: | --- |
| Shuffled frame-stratified 10-fold CV | 99.8682% ± 0.0713% | Not a valid direct comparison: frames from a video can appear in train and test. |
| Video-grouped 10-fold CV | 97.9580% ± 0.9444% | Conservative conditional result; −0.8020 percentage points from Table III. |

No split or fold was selected to obtain the paper number. One grouped fold is
98.7589%, nearly Table III's 98.76%, but the published aggregation rule is
unknown and the run reports the mean of all ten folds instead. The target paper
does not establish whether it used a random frame split, a grouped split, a
hold-out, a video-level aggregation, or another feature-reduction method; this
run therefore does **not** claim to reproduce the Table III score.

## Data gates

The required `datasets` and `huggingface-cache` Volumes exist in Modal `repro-sign`. Both datasets are populated.

| Dataset | Authoritative source | Permission status | Required action |
| --- | --- | --- | --- |
| ASL Alphabet | [Kaggle grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet), v1 | Kaggle metadata declares GPL-2.0; the user authorized this study’s use on 2026-08-25 | Stored and validated at `datasets/asl-alphabet`; train/evaluate with A-Z, SPACE, DELETE only |
| ISL-HS | [marlondcu/ISL at `d1d50bb`](https://github.com/marlondcu/ISL/tree/d1d50bb65540b904e3e0a6ffe0997872c4e9e645) | The repository has no published license; the user explicitly authorized this study's project-cloud use on 2026-08-25 | Populated and validated at `datasets/isl-hs`; do not redistribute data or derivatives without separate permission |

The committed `datasets/isl-hs/manifest.json` records six source archives and 468 videos. Its SHA-256 is `d8a278a87aa05898159e848d5f6c206364d0af74af84d3ea88e7d5c34f58e9b5`. The ASL v1 archive is 1,100,887,034 bytes (SHA-256 `7c572f14fbaff94f98835cfe71c7582dd379a5176e7c4f83dbf3a30e4b3f68c4`); its post-population manifest SHA-256 is `012e786c2f72e1f731f4384adbcf190c4e7084f80c64c8c17e3ad585693a453d`.

## Source search

The canonical [IEEE record](https://ieeexplore.ieee.org/document/9990143/) and the proceedings [table of contents](https://www.proceedings.com/content/066/066913webtoc.pdf) identify the paper. IEEE required login for PDF retrieval and the author-upload endpoint rejected retrieval on 2026-08-24, so no error response was treated as a paper PDF or hashed.

The paper contains no code release. Exact-title/method searches across GitHub, Zenodo, OSF, Hugging Face, and author accounts found no author code, archive, model, or supplement. The two author accounts inspected contained no relevant repository. The related CMC article above is openly available but is a distinct work with different reported results.

## Remaining faithful limitation

Both paper-owned targets have conditional terminal results, but the Table III
protocol gate remains open. An author-provided split, aggregation rule,
reduction configuration, or seed could turn these into stricter reproductions.

The structured run policies were backfilled from the pre-migration report
(whose SHA-256 is retained in `reproduction.json`). Committed two-hour/eight-hour
launcher ceilings and known ISL timestamps are preserved; declaration times and
retry maxima remain explicit unknowns, as do the unrecorded ASL run timestamps.

To populate the authorized datasets idempotently:

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run \
  papers/ahmad-2022-intelligent-landmarks/modal_app.py::populate_isl_hs
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run \
  papers/ahmad-2022-intelligent-landmarks/modal_app.py::populate_asl_alphabet
```

Then run the real-data preflights or the full conditional evaluations through the
same `repro-sign` wrapper. The functions fail rather than overwrite retained
evidence, so a fresh output Volume is required for an independent rerun.

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run \
  papers/ahmad-2022-intelligent-landmarks/modal_app.py::preflight
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run \
  papers/ahmad-2022-intelligent-landmarks/modal_app.py::evaluate_isl_hs
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run \
  papers/ahmad-2022-intelligent-landmarks/modal_app.py::preflight_asl_alphabet
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run \
  papers/ahmad-2022-intelligent-landmarks/modal_app.py::evaluate_asl_alphabet
```

Every Modal operation uses the `repro-sign` wrapper and mounts the shared
`huggingface-cache` volume. There are no model publications, author contacts,
or human participants.
