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

The target paper does not state the Table III split, whether the number is a 10-fold mean or a hold-out result, seed, feature-reduction rule, or metric implementation. It reports 10-fold *learning curves*, but does not tie that procedure to the table. Its detailed tables round Random Forest accuracy to `0.987` on both datasets, which cannot uniquely yield Table III's 98.68% and 98.76%. A run with an invented protocol would therefore be a conditional experiment, not a faithful Table III reproduction.

The complete Table III ledger, including the copied baselines and their terminal `not_produced` status, is in [`reproduction.json`](reproduction.json).

## What is known

- The proposed pipeline uses OpenCV and MediaPipe Hands to obtain 21 hand landmarks, then derives slope/angle and inter-finger-line features, reduces correlated dimensions, and classifies with Random Forest. The related same-author [CMC paper](https://www.techscience.com/cmc/v72n3/47480/html) describes this method and a 100-tree forest, but reports different scores. It is a method lead, not substitute target evidence.
- The paper describes ISL-HS as 26 letters with 18 roughly three-second videos per class and says it uses the first 60 frames of each video.
- It describes ASL Alphabet as 87,000 200×200 colour images and “28 gestures.” The cited source instead has 29 class directories—A-Z, SPACE, DELETE, and NOTHING—and 87,000 = 29 × 3,000. The inclusion decision is not published.

## Data gates

Neither required data tree exists in Modal `repro-sign` Volume `datasets`; `huggingface-cache` and `datasets` themselves were verified to exist.

| Dataset | Authoritative source | Permission status | Required action |
| --- | --- | --- | --- |
| ASL Alphabet | [Kaggle grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet), v1 | Kaggle metadata declares GPL-2.0; account terms and project-cloud processing have not been verified | Resolve its 28/29 class choice and terms before population at `datasets/asl-alphabet` |
| ISL-HS | [marlondcu/ISL at `d1d50bb`](https://github.com/marlondcu/ISL/tree/d1d50bb65540b904e3e0a6ffe0997872c4e9e645) | No repository license or cloud-processing permission is published | Team S must establish permission before download, storage, decoding, or processing at `datasets/isl-hs` |

No data, frames, predictions, or trained weights have been acquired, produced, or published.

## Source search

The canonical [IEEE record](https://ieeexplore.ieee.org/document/9990143/) and the proceedings [table of contents](https://www.proceedings.com/content/066/066913webtoc.pdf) identify the paper. IEEE required login for PDF retrieval and the author-upload endpoint rejected retrieval on 2026-08-24, so no error response was treated as a paper PDF or hashed.

The paper contains no code release. Exact-title/method searches across GitHub, Zenodo, OSF, Hugging Face, and author accounts found no author code, archive, model, or supplement. The two author accounts inspected contained no relevant repository. The related CMC article above is openly available but is a distinct work with different reported results.

## Next faithful step

Once the two open data permissions and the Table III protocol gate are resolved, the minimal implementation is a CPU-only, pinned MediaPipe/OpenCV/scikit-learn pipeline: verify dataset manifests, extract the stated landmarks/features, run a real-data preflight, then evaluate only the authorized protocol. ISL video decoding, if authorized, will use `simple-video-utils`; no video decoding is currently performed.

Every Modal operation will continue to use the `repro-sign` wrapper and mount the shared `huggingface-cache` volume. There are no runs, patches, artifacts, author contacts, human participants, or model publications yet.
