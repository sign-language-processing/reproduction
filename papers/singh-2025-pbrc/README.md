# PBRC sign language recognition reproduction

**Paper ID:** `7a6ad02a9fd4d6c6ce82a2c6b90132fd437db083`

**Citation:** N. K. Singh, A. R. Syulistyo, Y. Tanaka, and H. Tamukoh, "Sign Language Recognition using Parallel Bidirectional Reservoir Computing," *Nonlinear Theory and Its Applications, IEICE*, 17(1):79–92, 2026. DOI [10.1587/nolta.17.79](https://doi.org/10.1587/nolta.17.79). arXiv:2512.19451.

**Paper:** https://arxiv.org/abs/2512.19451 · **Code/artifacts:** no PBRC code exists after a full search; the protocol basis is the same group's MRC repository, [tamukohlaboratory/MultipleReservoirComputing-MRC@810643c](https://github.com/tamukohlaboratory/MultipleReservoirComputing-MRC/tree/810643ca14ce338b5099bab12a91b7d1e8480cd8)

**Preference level:** 3

**Pipeline status:** `blocked_on_data`

**Numerical agreement:** `not_assessed` — no target has been produced; nothing has been run, because 570 of the 2,038 WLASL100 videos the paper uses are missing from the shared dataset and the user asked to wait until data completeness is settled.

**Attempt date:** 2026-10-02 (in progress, stopped at the data gate)

## Reproduction agents

| Agent ID | Model and version | Agent application | Contribution | Attribution evidence / unknowns |
| --- | --- | --- | --- | --- |
| `claude-opus-5-5-claude-code` | Claude Opus 5.5 (`claude-opus-5-5`) | Claude Code 2.1.267 (Claude desktop app, Code tab) | Assignment retrieval, target ledger, source search (including a delegated subagent of the same model), MRC code analysis, WLASL completeness audit, data gate, and this report. No experiments have been run. | Session metadata from 2026-10-02 (session `a4310b07-bdb5-4126-9b3e-deab5fbd76f4`). |

## Scope and target contract

The assignment says "Table 1 and Table 2". The queue record says code is `N/A`, scores are partly copied (`copied_scores: yes`), and the metric is Accuracy. All 28 numbers in the two tables are listed in `reproduction.json.targets`:

- **Table 1** (p. 12). PBRC, BRC, standard ESN and Bi-GRU on WLASL100. For each system the table gives top-1 accuracy as the mean of 5 runs, its SD, and the mean training time. That is 12 targets, all in scope. Training time depends on hardware (the paper used an Intel Core i7 with 32 GB RAM). It will be measured and reported, but it is excluded from the numerical-agreement judgment.
- **Table 2** (p. 13). PBRC Top-1/5/10 and training time: 4 in-scope targets. PBRC Top-1 is the same quantity as the Table 1 mean. The paper does not say whether Top-5 and Top-10 are also 5-run means; this attempt assumes they are.
- **Copied rows in Table 2.** Pose-TGCN, I3D and MRC, 12 targets, all out of scope with `copied_baseline: true`:
  - The Pose-TGCN and I3D accuracies come from Li et al. (WACV 2020), Table 3.
  - Their training times and the whole MRC row come from Syulistyo et al., PLOS ONE 2025, Table 6. PBRC cites that paper as [27].

**Split.** The paper reports 1,780 training, 258 validation and 258 test videos. The MRC code before commit `2ab7ad3` explains this exactly:
- `nslt_100.json` has 2,038 videos: 1,442 train, 338 val and 258 test.
- `LoadData.py` puts both `train` and `val` into "training": 1,442 + 338 = 1,780.
- Every other split, including "validation", resolves to the 258 `test` videos.

So the paper's validation and test sets are the same videos. Its hyperparameters were chosen by "highest validation accuracy", so that selection was presumably made on the test videos. This is an inference from the sibling code, not something the paper states.

## Source provenance

| Artifact | Canonical source | Pinned revision / SHA-256 | Role |
| --- | --- | --- | --- |
| Paper PDF (arXiv v1) | https://arxiv.org/pdf/2512.19451v1 | `79f7d309ed6c7ab02a5a011b1883fdd512b9cd91dc5b65db24a41a73cc9e9380` | Targets and protocol |
| Version of record | https://www.jstage.jst.go.jp/article/nolta/17/1/17_79/_pdf/-char/en | `cd090a00d1a2dc6def4a3c2557ca882ec18a366da0731038180ccda7684aed42` | Same values; no supplementary material |
| MRC code (sibling, not PBRC) | https://github.com/tamukohlaboratory/MultipleReservoirComputing-MRC | `810643ca14ce338b5099bab12a91b7d1e8480cd8` | Feature extraction, split, ridge readout, Bi-GRU; no LICENSE file |
| Companion BRC paper | https://arxiv.org/abs/2512.00777 | `76ae8b7a8b7eb5b3aab580310f3ef5efb5f82ecccba5f7931a1fbcb2a5867a7b` | Same split; no code |
| MRC paper (ref [27]) | https://doi.org/10.1371/journal.pone.0322717 | journal HTML, read 2026-10-02 | Source of copied rows; code link |
| WLASL paper | https://arxiv.org/abs/1910.11006 | `9848ac59ceaea801ec4934b2ced5329074c071fc25596a0b46c0759d677f56b2` | Source of the Pose-TGCN and I3D accuracies |
| Survey record export | survey tool `/export?collection=papers&id=7a6ad02a…` | `0fc0e75c211ca6cfb1eb2fc3f7abf2e8b7ce2b15bf83b7eb66be927c01f219ee` | Assignment; preserved in `reproduction.json.assignment.record` with reviewer emails redacted |

**Search for PBRC code (2026-10-02).** None of the following turned up PBRC or BRC code:
- The arXiv abstract and HTML pages, and J-STAGE (which lists "Supplementary material (0)").
- Both YouTube video abstracts; their descriptions only link the arXiv DOI.
- GitHub repository, code and commit searches: exact title, "bidirectional reservoir" combined with sign language or WLASL, PBRC, and the authors' e-mail addresses.
- GitHub accounts: the `tamukohlaboratory` organization (10 repositories), `Tamukoh`, `TamukohLab`, `Hibikino-Toms`, `kyutech-kct`, and the second author's `ArieRS` (68 repositories).
- Zenodo, OSF, Hugging Face and Kaggle.

The only related code is the MRC repository. It is pinned to `810643c`, the last commit before `2ab7ad3` (2026-02-06) changed the split logic to 1,442/338/258.

The Kaggle WLASL100 dataset listed in the survey record was rejected for three reasons:
- It is a third-party upload, last updated on 2026-06-20, after the paper.
- Its split is 748/165/100.
- It is not the data the paper used.

**Assignment provenance.** The survey export has `status: final` but no `confirmation` field. The ingest script and the queue-record contract require `confirmation: confirmed`. The record is therefore filed as a direct user request. The export is kept in full except that reviewer email addresses are redacted; the SHA-256 identifies the original file. No confirmation value was invented.

## Results

No in-scope target has been produced. Every in-scope target is `not_produced` with reason `data_unavailable` (gate `wlasl100-missing-videos`).

| Target ID | Paper location | System | Dataset/split | Metric | Original | Reproduced | Difference | Terminal reason / evidence |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| t1-pbrc-top1-mean | Table 1 | PBRC | WLASL100 test (258) | top-1 %, mean of 5 | 60.85 | — | — | data_unavailable |
| t1-pbrc-top1-sd | Table 1 | PBRC | WLASL100 test | SD, 5 runs | 1.38 | — | — | data_unavailable |
| t1-pbrc-train-time | Table 1 | PBRC | train+val (1,780) | seconds | 18.67 | — | — | data_unavailable |
| t1-brc-top1-mean | Table 1 | BRC (140) | WLASL100 test | top-1 % | 58.11 | — | — | data_unavailable |
| t1-brc-top1-sd | Table 1 | BRC | WLASL100 test | SD | 1.45 | — | — | data_unavailable |
| t1-brc-train-time | Table 1 | BRC | train+val | seconds | 12.33 | — | — | data_unavailable |
| t1-esn-top1-mean | Table 1 | ESN (280) | WLASL100 test | top-1 % | 56.90 | — | — | data_unavailable |
| t1-esn-top1-sd | Table 1 | ESN | WLASL100 test | SD | 1.34 | — | — | data_unavailable |
| t1-esn-train-time | Table 1 | ESN | train+val | seconds | 21.10 | — | — | data_unavailable |
| t1-bigru-top1-mean | Table 1 | Bi-GRU | WLASL100 test | top-1 % | 50.01 | — | — | data_unavailable |
| t1-bigru-top1-sd | Table 1 | Bi-GRU | WLASL100 test | SD | 2.58 | — | — | data_unavailable |
| t1-bigru-train-time | Table 1 | Bi-GRU | train+val | seconds | 3341.50 | — | — | data_unavailable |
| t2-pbrc-top1 | Table 2 | PBRC | WLASL100 test | top-1 % | 60.85 | — | — | data_unavailable |
| t2-pbrc-top5 | Table 2 | PBRC | WLASL100 test | top-5 % | 85.86 | — | — | data_unavailable |
| t2-pbrc-top10 | Table 2 | PBRC | WLASL100 test | top-10 % | 91.74 | — | — | data_unavailable |
| t2-pbrc-train-time | Table 2 | PBRC | train+val | seconds | 18.67 | — | — | data_unavailable |
| t2-pose-tgcn-top1 / top5 / top10 / train-time | Table 2 | Pose-TGCN | WLASL100 | top-k %, s | 55.43 / 78.68 / 87.60 / 2298.9 | — | — | copied_baseline (out of scope) |
| t2-i3d-top1 / top5 / top10 / train-time | Table 2 | I3D | WLASL100 | top-k %, s | 65.89 / 84.11 / 89.92 / 72822.5 | — | — | copied_baseline (out of scope) |
| t2-mrc-top1 / top5 / top10 / train-time | Table 2 | MRC | WLASL100 | top-k %, s | 60.35 / 84.65 / 91.51 / 52.7 | — | — | copied_baseline (out of scope) |

**Selected blocker: `data_unavailable`.** It is the earliest blocker, and on its own it prevents every target. The paper's numbers depend on the exact 2,038-video split, and 56 of the 258 test videos are missing.

No other blocker is open. The protocol gaps (`pbrc-unstated-protocol`) were resolved by the user's decision to use the MRC-derived configuration.

## How to repeat this

No entry points exist yet; they will be added once the data gate is resolved. The completeness audit can be repeated from this repository:

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh volume get datasets /WLASL/index.csv index.csv
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh volume get datasets /WLASL/WLASL_v0.3.json WLASL_v0.3.json
```

Then, for the top 100 glosses of `WLASL_v0.3.json`, map each instance URL to its file name with `create_index.py`'s `generate_name_from_url` and check whether that file appears in `index.csv`.

## Data provenance and permissions

| Dataset | Version/subset/splits | Source and access date | License/permission and cloud-use basis | Path in Volume `datasets` | Counts / manifest / checksum | Deviations |
| --- | --- | --- | --- | --- | --- | --- |
| WLASL100 | WLASL v0.3, `nslt_100.json` (2,038 videos: 1,442 / 338 / 258) | https://dxli94.github.io/WLASL/; shared volume populated 2026-07-09 and re-populated 2026-10-01 | C-UDA 1.0, academic/computational use only; survey record says "Custom (research only)" | `WLASL/` | Present: 1,023 / 243 / 202 = 1,468 (72.0%). `index.csv` `5c271dd9…`, `WLASL_v0.3.json` `31ba5a5c…`, `nslt_100.json` `aaa4bb43…` | 570 videos missing; see `wlasl100-missing-videos.tsv` |

**Is WLASL itself incomplete?** Yes. WLASL is distributed as source URLs, not video files, and many of those links have died:
- The shared copy holds 15,601 of the 21,083 WLASL2000 instances (74.0%) and 1,468 of the 2,038 WLASL100 instances (72.0%).
- Missing WLASL100 videos by host: handspeak.com 174 (HTTP 404 on the 2026-10-01 retry), aslpro.com 162 (HTTP 404/406 on a 2026-10-02 probe), YouTube 232 (removed videos), aslsignbank 2.
- The WLASL authors provide missing videos only through a request form that requires accepting their terms. That is a human/Team S action, recorded as open gate `wlasl100-missing-videos`.
- The paper uses only WLASL100, so the request needs to cover only the 570 WLASL100 videos in `wlasl100-missing-videos.tsv`. The full-WLASL counts above are context, not a requirement of this reproduction.

The PBRC authors' counts (1,780 + 258) match the complete `nslt_100` split, so they had all 2,038 videos.

## Environment and patches

None yet; no container has been built and no code has run. The MRC code will be used at `810643c`. The patches needed to add PBRC's bidirectional and parallel reservoirs will be listed here when written.

## Execution evidence

No runs. Modal access was verified on 2026-10-02 through `modal_repro_sign.sh` (profile `repro-sign`). The Modal 1.2.6 client under Python 3.9 has no `modal token info`, so a Modal 1.6.0 client was installed in a local Python 3.12 virtualenv to satisfy the wrapper.

## Guesses and deviations

These are planned and approved by the user (gate `pbrc-unstated-protocol`), and are not yet exercised.

| Detail | Paper/evidence says | This attempt will use | Rationale | Effect on interpretation |
| --- | --- | --- | --- | --- |
| Per-video readout | Ridge "describe[s] the bidirectional reservoir states"; "second stage of ridge regression" (§2.7) | MRC model-space representation: per-video Ridge(α=15) of x(t+1) on x(t), then Ridge(α=3) readout | Same group's code; the two-stage wording matches | Major; behavior-defining |
| Input scaling / connectivity / noise / transient drop | Not stated | 0.3 / 0.2 / 0.01 / 5 | MRC code | Moderate |
| Features | "Hand joint coordinates"; Fig. 5 shows pose and hands | MediaPipe Holistic pose 33 + both hands 21 each, x,y, nose-centred, per-frame z-score (150 dims) | MRC `ExtractTheKeypoint` option 2 | Moderate |
| Flipped copies | Not mentioned; 258 test videos | Evaluate on the 258 original test videos; use of flipped training copies still to be decided | MRC extracts flips for every split | Unknown |
| Leak equation | α·tanh(·) term (Eqs. 7–10) | Follow the paper's equation | MRC code omits α on tanh; the paper is explicit | Small to moderate |
| BRC/ESN ρ and α | Only node counts | PBRC's ρ=0.3, α=0.6 | Only values given | Moderate for Table 1 baselines |

## Attempts, failures, and dead ends

- **Survey record export.** The export endpoint requires an authenticated session. It was fetched through the user's logged-in browser session and its bytes were verified by SHA-256.
- **Modal client.** The first preflight failed: the installed Modal 1.2.6 client has no `token info` command. Upgrading under Python 3.9 was impossible (1.2.6 was the newest release offered there), so Modal 1.6.0 was installed in a Python 3.12 virtualenv. The wrapper preflight then passed.
- **Kaggle WLASL100 copy.** Rejected; see Source provenance.
- **`datasets/WLASL/siformer` CSVs.** These are augmented skeleton data for a different paper (32 samples per class), not MediaPipe features, so they are unusable here.

## Candidate flags, ethics, and human evaluation

- **Ethics flag.** The survey sets `potential_ethical_concerns: yes`. WLASL consists of publicly posted videos of identifiable signers, licensed under C-UDA for research use. This attempt processes the existing shared copy for computation only and never redistributes videos. No new participants are involved.
- **Human evaluation:** none.
- **Copied scores.** `copied_scores: yes` is confirmed for the Pose-TGCN, I3D and MRC rows of Table 2.

## Author and team contact

No author contact. Team S action is pending: request the 570 missing WLASL100 videos only (gate `wlasl100-missing-videos`).
