# Sivamohan et al. 2024: ISL hand-gesture CNN (not reproduced)

**Paper ID:** `6eeabb6bf1ef16b3c7185ce9198f4b4964dc790e`

**Citation:** S. Sivamohan, S. Anslam Sibi, T. R. Divakar, and S. Jagan, "Hand Gesture Recognition and Translation for International Sign Language Communication using Convolutional Neural Networks," in *2024 2nd International Conference on Advancement in Computation & Computer Technologies (InCACCT)*, 2024, pp. 635–640, doi: 10.1109/InCACCT61598.2024.10551000.

**Paper:** https://doi.org/10.1109/InCACCT61598.2024.10551000 · **Code/artifacts:** no paper-owned code found in a brief search.

**Preference level:** 3

**Pipeline status:** `insufficient_information`

**Numerical agreement:** `not_assessed` — no experiment was run. The requested Fig. 7 implies 99.97% accuracy, while the paper says it computes its 97.85% accuracy from that same figure. Any single reproduced value would contradict one of the two.

**Attempt date:** 2026-09-29

**Decision:** After discussion in `#repro-sign-team-r`, the team decided not to implement or run anything for this paper and to record it as not reproduced. This record documents the evidence behind that decision.

## Reproduction agents

| Agent ID | Model and version | Agent application | Contribution | Attribution evidence / unknowns |
| --- | --- | --- | --- | --- |
| `claude-code-opus-5-5` | Opus 5.5 (`claude-opus-5-5`) | Claude Code (Claude desktop app, Code tab) | Checked the target numbers against Fig. 7, inspected the dataset directory read-only, ran a brief code search, and wrote this record. No implementation, training, or evaluation. | Session system metadata, 2026-09-29. The application version was not exposed. |

No runs exist, so no run carries `agent_ids`.

## Scope and target contract

The queue record asks for **"Fig. 7"**, and its `textual_conclusion` quotes the paper's claim of **97.85%** accuracy. The metric is accuracy (queue metric ID `wzc6kxyuhls6lut`).

- **Fig. 7** (p. 639) is a confusion matrix of the 26-letter CNN classifier on the test set. The paper does not print an accuracy for it. Transcribed at 400 dpi, its diagonal holds 12,376 correct predictions. There are four off-diagonal errors, one each: true F → predicted W, J → L, T → D, V → B. The extra `0` row and column are all zeros. **Accuracy = 12,376 / 12,380 = 99.97%.**
- **Sec. VI-B** (p. 638) says accuracy "reaches an impressive 97.85%". It also says this accuracy is calculated from the Fig. 7 confusion matrix. The abstract and the Conclusion repeat 97.85%.
- **Sec. VII** (p. 639) separately reports 94.10% training accuracy and 99.25% validation accuracy, with losses 0.1834 and 0.0297. Neither accuracy equals 97.85% or 99.97%.

The paper names only one evaluation, and that evaluation gives two different accuracies: 99.97% from the figure and 97.85% from the text. A reproduction targeting Fig. 7 would contradict the headline number, and one targeting 97.85% would contradict Fig. 7. The Conclusion's train and validation numbers match neither. The target identity therefore cannot be resolved from the paper. Gate `target-fig7-vs-97-85` records the alternatives; the team resolved it by deciding not to attempt a reproduction.

## Source provenance

| Artifact | Canonical source | Pinned revision / SHA-256 | Role |
| --- | --- | --- | --- |
| Paper PDF (`6eeabb6bf1ef16b3c7185ce9198f4b4964dc790e.pdf`, 6 pp.) | https://doi.org/10.1109/InCACCT61598.2024.10551000 | `7cf482dd0717ed6de9a94adb5d0451342108d743d319c0ae1e7eb04de7a6dc22` | Targets and protocol |
| Queue export (`reproductions-6eeabb6bf1ef16b3c7185ce9198f4b4964dc790e-2026-09-29.json`) | REPRO-SIGN queue | `a7f4510ce9d8a827055a99f916e17fcca75d64bd7e185c97df52e88f8ee7fbfc` | Assignment provenance |

**Search (2026-09-29):**

- The PDF has no code, data, or supplement links, and the queue lists `code_repos: N/A`.
- A web search for the exact title plus "github" found no paper-owned repository.
- A GitHub repository and code search for the title found no repository attributed to the paper's authors.
- Not checked, because target identity had already been ruled irreconcilable: author and institution pages, Zenodo/OSF, and IEEE Xplore supplements.

The preference level is 3 only provisionally, since no paper-owned code was found. No implementation was done.

## Results

| Target ID | Paper location | System | Dataset/split | Metric + version | Original | Reproduced | Difference | Terminal reason / evidence |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `fig7-confusion-matrix-accuracy` | Fig. 7 (p. 639) | Proposed CNN, 60 epochs | Author ISL A–Z landmark images; test split (ratio unspecified) | Accuracy (trace/total), unspecified implementation | 99.9677 (derived from counts) | not produced | — | `target_ambiguous`: contradicts the paper's stated 97.85%, which the paper says comes from this figure |
| `text-accuracy-97-85` | Abstract; Sec. VI-B; Sec. VII | Proposed CNN, 60 epochs | Same; "based on" Fig. 7 | Accuracy, unspecified implementation | 97.85 | not produced | — | `target_ambiguous`: the only evaluation the paper ties to it yields 99.97%; the train (94.10%) and validation (99.25%) accuracies also differ |

No score is copied from earlier work (`copied_scores: no`, confirmed by reading the paper).

**Blocker:** `status.blocker.reason_code = target_ambiguous`. The requested number itself is ill-defined, so this blocker comes before any data, code, or compute question and on its own prevents every target. The data concern below is recorded separately.

## How to repeat this

No experiment was run, so there is nothing to execute. The Fig. 7 accuracy can be re-derived from the transcribed counts:

```bash
python3 -c "d=[468,473,469,466,468,497,462,495,471,499,492,465,495,490,501,465,464,473,465,466,462,494,471,461,474,470]; c=sum(d); print(c, c+4, 100*c/(c+4))"
```

Expected output: `12376 12380 99.96768982229402`.

## Data provenance and permissions

| Dataset | Version/subset/splits | Source and access date | License/permission and cloud-use basis | Path in Volume `datasets` | Counts / manifest / checksum | Deviations |
| --- | --- | --- | --- | --- | --- | --- |
| Author-supplied ISL fingerspelling landmark images (queue dataset `unnamed-1z5f0h0ibdphpl0`) | `AtoZ_3.1/{A..Z}`; no split files supplied | Authors' Google Drive folder, imported to Modal 2026-09-10; inspected read-only 2026-09-29 | Queue: permission to use for reproduction experiments and to publish resulting weights (authors replied). The import manifest records a user-requested transfer to project Modal storage. | `drive-unnamed-1z5f0h0ibdphpl0` | 4,681 images (180 per class; C=185, Q=178, R=178), plus 27 reference images in `alphabets/`. Manifest `_drive_imports/6f02a3bf59a27749/manifest.json` SHA-256 `4d0a4f400d026e38a8000c19494bbf83b00d7254b6d48225ce6ce1eeadbb9b8b`; import ZIP `c1f78369…0ddb4` | Not used |

The supplied pool of 4,681 images is smaller than Fig. 7's evaluation set alone: 12,380 samples, 461–501 per class. So the supplied files cannot be the exact data behind Fig. 7 unless an unreported transformation, such as augmentation, expanded them. This is an extra data-identity concern and was not pursued further.

## Environment and patches

None. No container was built and no code was patched.

## Execution evidence

No runs. The only Modal operations were read-only: listing the `datasets` Volume and downloading two small metadata files (`manifest.json` and the import verification record) with `.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh` under profile `repro-sign`.

## Guesses and deviations

| Detail | Paper/evidence says | This attempt used | Rationale | Effect on interpretation |
| --- | --- | --- | --- | --- |
| Fig. 7 accuracy | Not printed | Trace/total of the transcribed matrix = 99.97% | Standard accuracy, as Sec. VI-B says accuracy is computed from Fig. 7 | Establishes the inconsistency; no result depends on it |

## Attempts, failures, and dead ends

No implementation attempts. The team made the decision from the paper alone.

## Candidate flags, ethics, and human evaluation

- `potential_ethical_concerns: no` and `includes_human_evaluation: no` in the queue. The paper's translation-quality section (VI-D) is qualitative and not a target.
- `copied_scores: no`, confirmed.
- The queue dataset comments list author emails and repeat the paper's Kaggle-origin sentence. The dataset record states the authors replied (`contacted_got_reply`).
- No participants or new sensitive data were involved.

## Author and team contact

- **Team:** the target inconsistency was discussed in `#repro-sign-team-r`, and the team decided to mark the paper as not reproduced without experiments.
- **Authors:** not contacted for this attempt. The earlier dataset-access correspondence is recorded in the queue's dataset record. Author contact about the target discrepancy would come only after an independent attempt, and none was made.
