# Kemkar et al. 2023 sign-language-to-text reproduction

**Paper ID:** `4c8f3c2ec7947693b3b02faf7a63b4f6327a960b`

**Citation:** T. Kemkar, V. Rai, and B. Verma, "Sign Language to Text Conversion using Hand Gesture Recognition," in 2023 8th International Conference on Communication and Electronics Systems (ICCES), Coimbatore, India, 2023, doi: 10.1109/ICCES57224.2023.10192820.

**Paper:** https://doi.org/10.1109/ICCES57224.2023.10192820 · **Code/artifacts:** https://github.com/kemkartanya/Sign-Language-Recognition at `415849b52a3c6db79ea03a40545c4f1445dea386` (first dataset only; not linked from the paper)

**Preference level:** 1

**Pipeline status:** `partial` — the 85 Sign Language MNIST targets were produced; the 121 American Sign Language Dataset targets were not, because their protocol cannot be recovered.

**Numerical agreement:** `does_not_agree` — 23 of the 85 produced targets equal the published value at the paper's precision; 62 do not.

**Attempt date:** 2026-10-02

Preference level 1 applies to the first dataset, where the upstream notebook is executed unmodified. The second dataset has no code; its only run is a conditional reimplementation that produces no target value.

## Reproduction agents

| Agent ID | Model and version | Agent application | Contribution | Attribution evidence / unknowns |
| --- | --- | --- | --- | --- |
| `claude-code-fable-5-1` | Fable 5.1 (`claude-fable-5-1`) | Claude Code | Everything: assignment, target resolution, source search, data population, scripts, launching and monitoring every run, this report. | Session system metadata on 2026-10-02 names the model and the application. The application version was not exposed. |

Every run in `reproduction.json.runs` lists this agent in `agent_ids`.

## Scope and target contract

The assignment is the survey tool's single-paper export for this paper ID (status `final`). That export has no `confirmation` field, so `ingest_candidate.py` rejects it and the record is preserved as a direct assignment with email addresses redacted and the export's SHA-256 recorded.

The record asks for "Figure 3.4 and Table 3.1 for MNIST, Figure 3.5 and Table 3.2 for the American Sign Language Dataset". This became 206 targets:

- **Table 3.1** (Sign Language MNIST): precision, recall and F1 for 24 classes and for the micro, macro, weighted and samples averages. 84 numbers.
- **Fig. 3.4**: the confusion-matrix cells are not legible in the PDF. The comparable number is the accuracy the paper states when introducing the figure, 99%. 1 number.
- **Table 3.2** (American Sign Language Dataset): the same metrics for 36 classes and four averages. 120 numbers.
- **Fig. 3.5**: accuracy stated with the figure, 96%. 1 number.

No score is copied from earlier work. The paper reports one run per dataset and no seeds.

The paper contradicts itself on the headline numbers. The abstract and Section IV give 99% accuracy for MNIST and 96% for ASL. The tables give the reverse: every average in Table 3.1 is 0.95 or 0.96, and every average in Table 3.2 is 0.99. A third figure, "roughly 95%", is derived in the text from numbers (a sum over 1400 samples) that do not belong to this dataset. Both the table values and the stated accuracies are kept as targets, each against its own published value.

Table 3.1's Support column equals the per-class counts of the official `sign_mnist_test.csv` (7172 samples), which fixes the evaluation split and the row order for the first dataset.

## Source provenance

| Artifact | Canonical source | Pinned revision / SHA-256 | Role |
| --- | --- | --- | --- |
| Paper PDF | https://doi.org/10.1109/ICCES57224.2023.10192820 | `731dd7728ca4ddcf0e6a337c85c80c9d807afcf951da6e34539538c1196d642e` | Targets. Supplied by the operator; not committed. |
| Published code | https://github.com/kemkartanya/Sign-Language-Recognition | `415849b52a3c6db79ea03a40545c4f1445dea386` | Training recipe for the first dataset |
| Released checkpoint | `models/experiment-dropout-0` in the same commit | same commit | Supplementary evaluation only |
| Tutorial repository | https://github.com/Sathwick-Reddy-M/Sign-Language-Recognition | `46de18cb6f7fe1f4a5ced4eccdc65b2cd7793fae` | Origin of the upstream code; its checkpoint is evaluated as supplementary evidence |

The paper links no code and the queue record says `N/A`. The repository was found by searching GitHub for the first author's name. It is attributed to the paper on three grounds: the account name matches the first author; its `model.png` is the same diagram as the paper's Fig. 3.2, including the auto-generated layer names (`conv2d_3`, `dropout_1`, `flatten_1`, `dense_3`); and the notebook's final model is the architecture of Section III.C. It was last pushed on 2023-01-21, before the June 2023 conference. It has no license file. Its README points to a Towards Data Science article, not to the paper.

The repository is not original work by the paper's authors. It is a re-run copy of an earlier tutorial, https://github.com/Sathwick-Reddy-M/Sign-Language-Recognition (created July 2022, pinned at `46de18cb6f7fe1f4a5ced4eccdc65b2cd7793fae`), described in the Towards Data Science article the README links. `model.png` is byte-identical in both, so the paper's Fig. 3.2 is the tutorial's diagram. The notebook code differs in two trivial lines; the outputs and checkpoints differ because the notebook was run again. The paper author's copy stays the selected source because it is the version the author ran.

The repository has two limits. It contains nothing for the second dataset. And it never computes a classification report or confusion matrix: it prints test accuracy only (0.973 with raw pixels, 0.985 with pixels divided by 255), neither of which is the 0.96 of Table 3.1.

Searched on 2026-10-02: the PDF, the queue record, the IEEE Xplore page, the Semantic Scholar record, a web search for the exact title with authors, a GitHub repository search for the exact title (0 results), and a GitHub user search for the first author. The linked article and its tutorial repository were also read; they cover Sign Language MNIST only. Not checked: author and institution pages, Zenodo/OSF.

## Results

### Table 3.1 and Fig. 3.4: Sign Language MNIST (produced)

Primary run `mnist-notebook-1-attempt-2`, official test CSV, pixels divided by 255. Reproduced values are shown to four decimals; the paper prints two.

| Row | Support | P paper | P repro | R paper | R repro | F1 paper | F1 repro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 331 | 1.00 | 0.9851 | 1.00 | 1.0000 | 1.00 | 0.9925 |
| 1 | 432 | 1.00 | 1.0000 | 0.94 | 1.0000 | 0.97 | 1.0000 |
| 2 | 310 | 0.99 | 1.0000 | 1.00 | 1.0000 | 1.00 | 1.0000 |
| 3 | 245 | 1.00 | 1.0000 | 1.00 | 1.0000 | 1.00 | 1.0000 |
| 4 | 498 | 0.96 | 0.9595 | 1.00 | 1.0000 | 0.98 | 0.9794 |
| 5 | 247 | 0.98 | 1.0000 | 1.00 | 1.0000 | 0.99 | 1.0000 |
| 6 | 348 | 0.98 | 0.9452 | 0.87 | 0.9914 | 0.92 | 0.9677 |
| 7 | 436 | 1.00 | 0.9811 | 0.94 | 0.9541 | 0.97 | 0.9674 |
| 8 | 288 | 0.99 | 1.0000 | 1.00 | 1.0000 | 1.00 | 1.0000 |
| 9 | 331 | 0.89 | 1.0000 | 0.94 | 0.9154 | 0.91 | 0.9558 |
| 10 | 209 | 0.85 | 1.0000 | 0.99 | 1.0000 | 0.92 | 1.0000 |
| 11 | 394 | 1.00 | 0.9899 | 0.99 | 1.0000 | 1.00 | 0.9949 |
| 12 | 291 | 1.00 | 1.0000 | 0.99 | 0.9828 | 1.00 | 0.9913 |
| 13 | 246 | 1.00 | 1.0000 | 1.00 | 1.0000 | 1.00 | 1.0000 |
| 14 | 347 | 1.00 | 1.0000 | 0.93 | 1.0000 | 0.97 | 1.0000 |
| 15 | 164 | 0.80 | 1.0000 | 1.00 | 0.9756 | 0.89 | 0.9877 |
| 16 | 144 | 0.77 | 0.8780 | 0.99 | 1.0000 | 0.87 | 0.9351 |
| 17 | 246 | 0.97 | 1.0000 | 0.91 | 0.9146 | 0.94 | 0.9554 |
| 18 | 248 | 0.90 | 0.9689 | 0.69 | 0.8790 | 0.78 | 0.9218 |
| 19 | 266 | 1.00 | 1.0000 | 0.92 | 0.9511 | 0.96 | 0.9750 |
| 20 | 346 | 0.95 | 1.0000 | 0.94 | 1.0000 | 0.94 | 1.0000 |
| 21 | 206 | 0.97 | 1.0000 | 1.00 | 1.0000 | 0.98 | 1.0000 |
| 22 | 267 | 0.81 | 0.9228 | 1.00 | 0.9850 | 0.90 | 0.9529 |
| 23 | 332 | 1.00 | 0.9405 | 0.94 | 1.0000 | 0.97 | 0.9693 |
| micro avg | 7172 | 0.96 | 0.9822 | 0.96 | 0.9822 | 0.96 | 0.9822 |
| macro avg | 7172 | 0.95 | 0.9821 | 0.96 | 0.9812 | 0.95 | 0.9811 |
| weighted avg | 7172 | 0.96 | 0.9830 | 0.96 | 0.9822 | 0.96 | 0.9821 |
| samples avg | 7172 | 0.96 | 0.9822 | 0.96 | 0.9822 | 0.96 | 0.9822 |

| Target | Paper | Reproduced | Difference |
| --- | ---: | ---: | ---: |
| `fig-3-4-accuracy` (percent) | 99 | 98.2153 | -0.7847 |

The reproduced averages are about two points above Table 3.1 (0.982 against 0.96) and about one point below the 99% stated in the text. The per-class pattern also differs: the paper's weakest class is row 18 (recall 0.69), where this run has 0.879, and the largest single difference is row 15 precision (0.80 in the paper, 1.00 here). Exact per-target values and differences are in `reproduction.json.targets`.

Supplementary evidence, not target values:

| Run | What it is | Accuracy, pixels / 255 | Accuracy, raw pixels | Macro F1, pixels / 255 |
| --- | --- | ---: | ---: | ---: |
| `mnist-notebook-1-attempt-2` | primary retraining | 0.9822 | 0.9707 | 0.9811 |
| `mnist-notebook-2-attempt-2` | repeat | 0.9755 | 0.9679 | 0.9721 |
| `mnist-notebook-3-attempt-3` | repeat | 0.9809 | 0.9759 | 0.9791 |
| `mnist-released-checkpoint` | checkpoint committed upstream, no training | 0.9855 | 0.9731 | 0.9835 |
| `mnist-article-checkpoint` | checkpoint committed in the tutorial repository, no training | 0.9632 | 0.9632 | 0.9570 |

The notebook sets no TensorFlow seed, so the three retrainings differ; they span 0.9755 to 0.9822. The released checkpoint reproduces the two accuracies printed in the upstream notebook (0.985 and 0.973), which confirms the evaluation path and environment. None of these four models gives 0.96 under either input scaling.

The tutorial's own checkpoint does score 0.963, which rounds to Table 3.1's averages. It was tested as the possible origin of the table and is not: only 30 of the 84 Table 3.1 values match at two decimals. The origin of Table 3.1 remains unidentified.

### Table 3.2 and Fig. 3.5: American Sign Language Dataset (not produced)

All 121 targets are `not_produced` with reason `protocol_ambiguous`, gate `asl-dataset-protocol` (open).

The paper gives only the dataset URL and the class count. It states no split, input size, colour handling or training schedule, and Fig. 3.2 shows a 24-way output layer that cannot classify 36 classes. The Kaggle archive stores every image twice: 2515 files under `asl_dataset/` and the same 2515 again under `asl_dataset/asl_dataset/`. Table 3.2's total support, 5030, is that full doubled count. So the reported numbers were computed over the whole archive, training images included, and no held-out test set can be identified. The table also lists support 140 for every class, which sums to 5040; class `t` has only 130 files.

One conditional run was made and is kept as evidence. It carries the notebook's pipeline over to this dataset with invented choices (listed under guesses) and evaluates on all 5030 files, as the table's support implies. Its numbers are not reproduced target values.

| Row | Support | P paper | P cond. | R paper | R cond. | F1 paper | F1 cond. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 140 | 0.99 | 0.9492 | 0.99 | 0.8000 | 0.99 | 0.8682 |
| 1 | 140 | 1.00 | 0.9459 | 0.96 | 1.0000 | 0.98 | 0.9722 |
| 2 | 140 | 1.00 | 0.8630 | 0.97 | 0.9000 | 0.99 | 0.8811 |
| 3 | 140 | 1.00 | 1.0000 | 1.00 | 1.0000 | 1.00 | 1.0000 |
| 4 | 140 | 1.00 | 0.9306 | 1.00 | 0.9571 | 1.00 | 0.9437 |
| 5 | 140 | 1.00 | 0.9559 | 1.00 | 0.9286 | 1.00 | 0.9420 |
| 6 | 140 | 1.00 | 0.9762 | 0.67 | 0.5857 | 0.80 | 0.7321 |
| 7 | 140 | 1.00 | 1.0000 | 1.00 | 1.0000 | 1.00 | 1.0000 |
| 8 | 140 | 1.00 | 1.0000 | 1.00 | 1.0000 | 1.00 | 1.0000 |
| 9 | 140 | 1.00 | 1.0000 | 1.00 | 1.0000 | 1.00 | 1.0000 |
| 10 | 140 | 1.00 | 1.0000 | 1.00 | 1.0000 | 1.00 | 1.0000 |
| 11 | 140 | 1.00 | 1.0000 | 1.00 | 1.0000 | 1.00 | 1.0000 |
| 12 | 140 | 1.00 | 1.0000 | 1.00 | 1.0000 | 1.00 | 1.0000 |
| 13 | 140 | 0.99 | 0.9859 | 1.00 | 1.0000 | 0.99 | 0.9929 |
| 14 | 140 | 1.00 | 1.0000 | 1.00 | 1.0000 | 1.00 | 1.0000 |
| 15 | 140 | 1.00 | 1.0000 | 1.00 | 1.0000 | 1.00 | 1.0000 |
| 16 | 140 | 1.00 | 0.9859 | 1.00 | 1.0000 | 1.00 | 0.9929 |
| 17 | 140 | 1.00 | 1.0000 | 1.00 | 0.9857 | 1.00 | 0.9928 |
| 18 | 140 | 1.00 | 1.0000 | 1.00 | 1.0000 | 1.00 | 1.0000 |
| 19 | 140 | 1.00 | 1.0000 | 1.00 | 1.0000 | 1.00 | 1.0000 |
| 20 | 140 | 1.00 | 0.9859 | 1.00 | 1.0000 | 1.00 | 0.9929 |
| 21 | 140 | 1.00 | 1.0000 | 1.00 | 1.0000 | 1.00 | 1.0000 |
| 22 | 140 | 1.00 | 0.9211 | 1.00 | 1.0000 | 1.00 | 0.9589 |
| 23 | 140 | 1.00 | 1.0000 | 1.00 | 0.9143 | 1.00 | 0.9552 |
| 24 | 140 | 0.99 | 0.8272 | 0.99 | 0.9571 | 0.99 | 0.8874 |
| 25 | 140 | 1.00 | 1.0000 | 1.00 | 1.0000 | 1.00 | 1.0000 |
| 26 | 140 | 1.00 | 1.0000 | 1.00 | 1.0000 | 1.00 | 1.0000 |
| 27 | 140 | 1.00 | 1.0000 | 1.00 | 1.0000 | 1.00 | 1.0000 |
| 28 | 140 | 1.00 | 1.0000 | 1.00 | 1.0000 | 1.00 | 1.0000 |
| 29 | 130 | 1.00 | 1.0000 | 1.00 | 1.0000 | 1.00 | 1.0000 |
| 30 | 140 | 1.00 | 0.9589 | 1.00 | 1.0000 | 1.00 | 0.9790 |
| 31 | 140 | 0.97 | 0.9219 | 1.00 | 0.8429 | 0.99 | 0.8806 |
| 32 | 140 | 0.80 | 0.7041 | 1.00 | 0.9857 | 0.89 | 0.8214 |
| 33 | 140 | 1.00 | 1.0000 | 1.00 | 1.0000 | 1.00 | 1.0000 |
| 34 | 140 | 1.00 | 1.0000 | 1.00 | 1.0000 | 1.00 | 1.0000 |
| 35 | 140 | 0.99 | 1.0000 | 0.99 | 0.9143 | 0.99 | 0.9552 |
| micro avg | 5030 | 0.99 | 0.9658 | 0.99 | 0.9658 | 0.99 | 0.9658 |
| macro avg | 5030 | 0.99 | 0.9698 | 0.99 | 0.9659 | 0.99 | 0.9652 |
| weighted avg | 5030 | 0.99 | 0.9697 | 0.99 | 0.9658 | 0.99 | 0.9652 |
| samples avg | 5030 | 0.99 | 0.9658 | 0.99 | 0.9658 | 0.99 | 0.9658 |

| Conditional run `asl-conditional-seed-42` | Value |
| --- | ---: |
| Accuracy on all 5030 files (percent; the paper states 96 with Fig. 3.5) | 96.5805 |
| Accuracy on its 1457-file validation part | 95.8819 |

Row order is assumed to be the sorted folder names (0-9, then a-z). In the conditional run the two weakest rows by F1 are 6 (0.732) and 32 (0.821); the paper's two weakest are the same rows (0.80 and 0.89).

## How to repeat this

From the repository root, with the Modal `repro-sign` profile authenticated:

```bash
# Data gate: both dataset paths and the shared cache
.agents/skills/reproduce-paper/scripts/check_modal_dataset.sh sign-language-mnist manifest.json
.agents/skills/reproduce-paper/scripts/check_modal_dataset.sh asl-dataset manifest.json
# Only if asl-dataset is absent (idempotent)
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/kemkar-2023-sign-language-to-text/modal_app.py::populate_asl_dataset
# Supplementary: evaluate the checkpoints committed upstream and in the tutorial repository
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/kemkar-2023-sign-language-to-text/modal_app.py::mnist_released --run-name NEW_NAME
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/kemkar-2023-sign-language-to-text/modal_app.py::mnist_article_checkpoint --run-name NEW_NAME
# Table 3.1 / Fig. 3.4: execute the upstream notebook, then evaluate its final model
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/kemkar-2023-sign-language-to-text/modal_app.py::mnist_notebook --run-name NEW_NAME
# Conditional Table 3.2 / Fig. 3.5 experiment
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach papers/kemkar-2023-sign-language-to-text/modal_app.py::asl_conditional --run-name NEW_NAME --seed 42
```

Each function refuses to reuse an existing run name and writes `metrics.json`, `predictions.json`, `run.json`, `pip-freeze.txt` and `log.txt` to `modal://repro-sign/volumes/kemkar-2023-sign-language-to-text-results/RUN_NAME/`. `metrics.json` holds the full classification report and confusion matrix. The notebook has no resume path; an interrupted run is restarted under a new name. Retraining is unseeded upstream, so a repeat will not give identical numbers.

## Data provenance and permissions

| Dataset | Source | License | Modal path | Identifier | Counts |
| --- | --- | --- | --- | --- | --- |
| Sign Language MNIST | https://www.kaggle.com/datasets/datamunge/sign-language-mnist, version 1 | CC0: Public Domain | `datasets:/sign-language-mnist` | manifest `3e7d79238e611ff78a13000bef2fc2570567fce6c4cd769d806e36e68dd5d0e0` | 27455 train, 7172 test |
| American Sign Language Dataset | https://www.kaggle.com/datasets/ayuraj/asl-dataset, version 1 | CC0: Public Domain | `datasets:/asl-dataset` | manifest `740aa32cdb44642ef3f304086f275742dadc39d11d616597b9812d9ab204bdf0`, tree `eac4c06624e1fd8c91f7f500839002024662ecb104186cd205648ede38400c84` | 5030 files, 2515 unique |

Both licenses were read from Kaggle's dataset metadata on 2026-10-02. Sign Language MNIST was already on the shared Volume. The ASL dataset was added by this attempt with `data.sh`. The `datasets` Volume is mounted read-only in every experiment function.

The queue record flags potential ethical concerns. The reviewer's comment ties this to the paper's framing of sign languages. Both datasets are public CC0 collections of cropped hand images; no new ethics gate was opened.

## Environment and compute

- Image: Modal `debian_slim` with a Python 3.9.13 virtualenv (the notebook's kernel version) holding the upstream pins: TensorFlow 2.9.1, scikit-learn 1.1.1, NumPy 1.23.1, pandas 1.4.3, Pillow 9.2.0. Modal image `im-NYozsuRdDXxjQlcyFpQgx6` for the first launches; the retained notebook runs used the rebuilt image recorded in each `run.json`. Full `pip freeze` is stored with every run.
- Hardware: Modal CPU containers, 8 cores and 16 GiB requested, no GPU. The container did not report a CPU model.
- Wall time: primary notebook run 17 min 50 s; repeats 17 min 44 s and 19 min 16 s; released-checkpoint evaluation 22 s; conditional ASL run 11 min 30 s. GPU-hours: 0. Cost was not metered per run.
- Modal app IDs, task IDs, timestamps and dashboard links are in each run entry.

## Guesses

1. The upstream repository is the paper's code for the first dataset. The paper does not link it.
2. Table 3.1 comes from the notebook's final model (`experiment-dropout-0`, dropout 0.3) trained as in the notebook: Adam, batch 32, up to 10 epochs, early stopping, best validation loss, 19500 / 7955 train / validation rows. The paper states only the architecture.
3. Test pixels are divided by 255, the notebook's final test protocol. The raw-pixel variant is computed and stored but not used for targets.
4. The tables were made with scikit-learn's `classification_report` on one-hot labels, the only input form that prints "micro avg" and "samples avg" rows. Predictions are the argmax class.
5. Table 3.1 rows are the label-binarizer order (confirmed by the Support column). Table 3.2 rows are assumed to be sorted folder names.
6. The accuracy stated beside each figure stands in for the figure, whose cells are illegible.
7. Conditional ASL run only: 28x28 grayscale, pixels / 255, PIL default resize, seeded shuffle with the notebook's 71/29 proportion over all 5030 files (duplicates not separated), 36-way output, the notebook's optimiser and callbacks, seed 42.

## Deviations

1. The notebook is executed headlessly with `jupyter nbconvert`. Its source is unmodified. The CSV paths are symlinks to the shared Volume, and the committed `models/` directory is removed first so only freshly trained checkpoints can be loaded.
2. matplotlib 3.5.2, matplotlib-inline 0.1.3, protobuf 3.19.6, nbconvert 7.2.10 and ipykernel 6.15.1 are pinned here; upstream does not list them. streamlit is not installed.
3. The paper used a 4 GB laptop and Google Colab; these runs used Modal CPU containers.

No upstream file is patched.

## Failed attempts and dead ends

| Run or step | What happened | Response |
| --- | --- | --- |
| Image build on `python:3.9.13-slim` | apt could not install git (the Debian release's security repository returns 404), and Modal's in-container client needs Python 3.10 or newer. | Python 3.9.13 virtualenv inside a current image; upstream commit fetched as a GitHub archive. |
| TensorFlow 2.9.1 in the container's main Python | Its protobuf pin broke Modal's client. | Same virtualenv. |
| `mnist-notebook-1`, `-2`, `-3` (attempt 1) | `nbconvert` failed in the first matplotlib cell: unpinned matplotlib-inline is incompatible with matplotlib 3.5.2. No training ran. | Pinned matplotlib-inline 0.1.3 and relaunched. |
| `mnist-notebook-3-attempt-2` | Container interrupted mid-training; Modal's automatic re-run hit the no-overwrite guard. | Relaunched once as `mnist-notebook-3-attempt-3`, which completed. |

## Gates

| Gate | Status | Note |
| --- | --- | --- |
| `paper-full-text-access` | resolved | The operator supplied the PDF. |
| `modal-reauthentication` | resolved | The operator ran `modal setup`. The preflight kept failing because the installed Modal client (1.2.6 on Python 3.9) lacks the `token info` command the wrapper calls; it passed with Modal 1.6.0 under a newer Python. |
| `asl-dataset-protocol` | open | A reviewer decides whether any protocol for the second dataset can count as the paper's, for example after asking the authors. |

## Author contact

None.
