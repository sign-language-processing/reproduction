# Sign2Pose reproduction

**Paper ID:** `559de08af0b4679a571ac2bd26cacd15728811aa`

**Citation:** Eunice, J.; J, A.; Sei, Y.; Hemanth, D.J. Sign2Pose: A Pose-Based Approach for Gloss Prediction Using a Transformer Model. *Sensors* 2023, 23(5), 2853. https://doi.org/10.3390/s23052853

**Paper:** https://doi.org/10.3390/s23052853 (PMC10007493) · **Code/artifacts:** none from the authors after a full search; cited basis SPOTER at `maty-bohacek/spoter@0f909bf`

**Preference level:** 3 (planned; no published code)

**Pipeline status:** not yet assigned; work in progress. The user authorised a conditional SPOTER-based reconstruction at gate `sign2pose-private-split` (numbers are evidence only; no target can be marked produced). Gate `wlasl-missing-videos` remains open, and the runs use the ~72% of instances with stored videos.

**Numerical agreement:** not yet assessed. The WLASL1000 and WLASL2000 runs are still training; this README will be completed once they finish. `reproduction.json` is the current record.

**Attempt date:** 2026-09-27

## Reproduction agents

| Agent ID | Model and version | Agent application | Contribution | Attribution evidence / unknowns |
| --- | --- | --- | --- | --- |
| `opus-5-5-claude-code` | Opus 5.5 (`claude-opus-5-5`) | Claude Code 2.1.283 | Branch and directory setup, assignment provenance, paper acquisition, target ledger, source search, SPOTER and dataset inspection, gates. No run executed. | Session environment declares the model; `claude --version` printed `2.1.283 (Claude Code)`. |

## Scope and target contract

The queue asks for **"Table 5, bottom line"**. That is the row labelled "Our's" in Table 5 ("Performance analysis on top 1% macro recognition accuracy…"), which gives four numbers. The abstract, contribution 4 and Section 6 repeat the same values as the authors' own results, so none of them is a copied baseline. The four rows above it (Pose-GRU, Pose-TGCN, GCN-BERT, SPOTER) are copied baselines and fall outside the assignment.

| Target ID | Dataset | Published top-1 (%) |
| --- | --- | ---: |
| `table5-ours-wlasl100` | WLASL100 | 80.9 |
| `table5-ours-wlasl300` | WLASL300 | 64.21 |
| `table5-ours-wlasl1000` | WLASL1000 | 49.46 |
| `table5-ours-wlasl2000` | WLASL2000 | 38.65 |

What the paper specifies about the protocol, with every gap marked:

- **Split: not the official WLASL split.** Section 5 describes a random 85:15 split and, in the same sentence, a 70/15/15 split. It publishes no split files, seed or stratification rule.
- **Pipeline:** key-frame extraction (histogram difference, threshold μ+σ, Euclidean distance; Algorithm 1 never defines what the distance compares). Then Apple Vision API poses (54 joints, 108 dimensions), rotation/squeeze/perspective/arm-joint augmentation, YOLOv3 signing-space and hand anchor-box normalisation (no weights or detector training data given), and horizontal flip with p = 0.5.
- **Model and training (Table 3 and Section 5):** 6 encoder and 6 decoder layers, 9 heads, hidden size 108, feed-forward 2048, one class query. SGD with lr 0.001, weight decay 1e-4 and momentum 0, uniform (0, 1) initialisation, cross-entropy loss, 300 epochs. The paper says it was implemented in TensorFlow.
- **Metric:** top-1 accuracy. The caption says "macro", while the text says "top 1 class accuracy". The copied baseline rows are per-instance. The paper states no checkpoint rule; Figure 7 shows a plateau after epoch 240. It gives no seeds or variance.

## Source provenance

| Artifact | Canonical source | Pinned revision / SHA-256 | Role |
| --- | --- | --- | --- |
| Paper full text | Europe PMC JATS XML, PMC10007493 | `cf6fc433b7cf8584d71b4f062af25bbc598be0905df204f6a32b28dea3470c67` | Targets and protocol |
| Published code | none found | — | — |
| SPOTER code (paper ref. 48) | https://github.com/maty-bohacek/spoter | `0f909bf92690772f43f0062be41860ed85b461ad` (Apache-2.0) | Cited basis; candidate base for a reimplementation |
| SPOTER WLASL100 poses | spoter release `supplementary-data` | sizes recorded; SHA-256 on download (CC BY-NC 4.0) | Apple Vision poses, WLASL100 only |

No PDF bytes could be obtained. MDPI and Europe PMC return HTTP 403 to automated clients, and PMC returns a proof-of-work page. The hashed JATS XML is the same article, and its tables match the text the user pasted.

**Search performed (2026-09-27):** the paper's links (only the WLASL Data Availability link exists, and there are no supplements); the MDPI landing page (403); GitHub repo and code search for "Sign2Pose" and variants; the first author's apparent GitHub account; web searches on title, method and author; Zenodo, Hugging Face and OSF.

Two GitHub code hits turned up, and neither is the authors' artifact. One is a thesis bibliography. The other is `Chinh-de/Sign_Language_Recognition`, a 2025 third-party MediaPipe model with a different architecture (input 171, d_model 256, 3+3 layers), so it was rejected. The full list is in `reproduction.json.source_search`.

Table 3 and Sections 3.4–5 match SPOTER's code almost parameter for parameter (`nn.Transformer(108, 9, 6, 6)`, FF 2048, SGD, the same augmentations and head-unit normalisation). The paper's additions are key-frame extraction, the YOLOv3 normalisation and the private split, and none of them has code.

## Results

Interim conditional results (test top-1 %, MediaPipe poses, official WLASL split, stored videos only, key frames, seed 379). These are **not** reproduced targets:

| Target | Paper | SPOTER's pick (max over test) | Validation-selected | Macro, validation-selected | Run |
| --- | ---: | ---: | ---: | ---: | --- |
| `table5-ours-wlasl100` | 80.9 | 47.50 | 47.50 | 49.31 | `full-wlasl100` |
| `table5-ours-wlasl300` | 64.21 | 27.36 | 23.58 | 24.09 | `full-wlasl300` |
| `table5-ours-wlasl1000` | 49.46 | running | running | running | `full-wlasl1000` |
| `table5-ours-wlasl2000` | 38.65 | running | running | running | `full-wlasl2000` |

## How to repeat this

Entry points are in `modal_app.py` (see its docstring): `extract_all` (MediaPipe poses), `export_csvs`, and `main` (SPOTER training and test). `poses.py` holds pose mapping, key-frame selection and CSV export. Full instructions follow once the runs finish.

## Data provenance and permissions

| Dataset | Version/subset/splits | Source and access date | License/permission and cloud-use basis | Path in Volume `datasets` | Counts / manifest / checksum | Deviations |
| --- | --- | --- | --- | --- | --- | --- |
| WLASL | `WLASL_v0.3.json`; WLASL100/300/1000/2000 are the first K glosses | https://dxli94.github.io/WLASL/, checked 2026-09-27 | Custom research-only terms; a project copy already exists (queue `on_modal: yes`). Missing videos require the authors' terms-of-use request form. | `WLASL/` | JSON `31ba5a5c…`, index `fba4a2c0…`. Stored videos: 1443/2038, 3671/5117, 9633/13168, 15146/21083 (~72%) | The paper's Table 2 counts match the full JSON; about 28% of videos are missing from the store |

## Environment and patches

None yet. No container has been built and no patches exist.

## Execution evidence

No runs.

## Guesses and deviations

None yet. The substitutions any conditional run would need are listed in gate `sign2pose-private-split`, alternative `conditional-spoter-reconstruction`.

## Attempts, failures, and dead ends

- `ingest_candidate.py` rejected the queue export because it is a single object rather than an array. The record is preserved verbatim, not reshaped.
- The queue record has `status: final` but no `confirmation` field. The assigned reproducer confirmed approval on 2026-09-27. The record stays verbatim, so the validator still flags the missing field.
- Every attempt to get the paper as a PDF was blocked (see Source provenance). The JATS XML was used instead.

## Candidate flags, ethics, and human evaluation

- `copied_scores: yes`: this applies to the four comparison rows of Table 5, not the target row.
- `code_repos: N/A`: confirmed by the independent search.
- `compute_requirements: N/A`: the paper states no hardware. The SPOTER-sized model (about 6 M parameters) fits a single GPU.
- `includes_human_evaluation: no` and `potential_ethical_concerns: no`: consistent with the paper. The work uses existing public sign videos only and adds no participants.

## Author and team contact

None. The paper names a corresponding author (D. J. Hemanth). Contacting the authors is listed as a gated option, not taken. The request for the missing WLASL videos is routed to Team S through gate `wlasl-missing-videos`.
