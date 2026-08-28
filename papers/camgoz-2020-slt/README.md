---
license: cc-by-nc-sa-4.0
library_name: pytorch
tags:
  - sign-language-translation
  - sign-language-recognition
  - reproduction
datasets:
  - rwth-phoenix-weather-2014t
---

# Sign Language Transformers reproduction

**Paper ID:** `camgoz-2020-slt`

**Preference level:** 1

**Status:** `complete`

**Numerical agreement:** not fully reproduced

**Attempt date:** 2026-08-20

## Scope and target contract

This reproduces the authors' supplied joint Sign2(Gloss+Text) configuration, with recognition and translation loss weights both equal to 1. It corresponds to the lambda_R=1, lambda_T=1 row of Table 4: dev WER/BLEU-4 35.13/21.73 and test WER/BLEU-4 33.75/21.22. The published config's seed 42 and development-set checkpoint/search selection are used.

All four target metrics were produced, so the pipeline status is `complete`. The values remain below the paper and are therefore not a full numerical reproduction; the approximately 19 BLEU result is accepted as done.

## Source provenance

| Artifact | Pinned source | Role |
| --- | --- | --- |
| Paper | [CVF PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Camgoz_Sign_Language_Transformers_Joint_End-to-End_Sign_Language_Recognition_and_Translation_CVPR_2020_paper.pdf), SHA-256 `ae08194f...25a1` | Targets and protocol |
| Code | [neccam/slt](https://github.com/neccam/slt/tree/90588825f6229474bc19ac7a6b30ea3116635ba3) at `90588825f6229474bc19ac7a6b30ea3116635ba3` | Model, training, search, metrics, and config |
| Features | [Author-restored pami0 files](https://github.com/neccam/nslt/issues/38#issuecomment-4330442225), exact hashes in `reproduction.json` | Train/dev/test inputs |
| Weights | [repro-sign/neccam-slt at `157663d`](https://huggingface.co/repro-sign/neccam-slt/tree/157663da7cd6390cb8c999e848e55645ec4336ea), `model.ckpt` SHA-256 `e5c34ece...fa8b8` | Reproduced seed-42 checkpoint |

## Results

| Split | Metric | Paper | Reproduced | Difference |
| --- | --- | ---: | ---: | ---: |
| dev | WER | 35.13 | 53.6696 | +18.5396 |
| dev | BLEU-4 | 21.73 | 19.1460 | -2.5840 |
| test | WER | 33.75 | 54.0033 | +20.2533 |
| test | BLEU-4 | 21.22 | 18.4889 | -2.7311 |

Development selected recognition beam 10 and translation beam 2 with alpha 2. Exact values, run metadata, and native-result hashes are in `reproduction.json`.

## How to repeat this

From the repository root, authenticated to the Modal profile/workspace `repro-sign`:

```bash
./setup.sh
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run \
  papers/camgoz-2020-slt/scripts/modal_app.py::train
papers/camgoz-2020-slt/scripts/publish.sh
```

The launcher copies the pinned upstream `configs/sign.yaml`, changes only `data_path` and `model_dir`, and invokes the upstream `python -m signjoey train` entrypoint. The `datasets` Volume is read-only, `huggingface-cache` is mounted at `/cache/huggingface`, and outputs use the v2 Volume `neccam-slt-results`.

## Data provenance and permissions

The run uses the authors' restored pami0 1024-D features: 7,095 train, 519 dev, and 642 test records. The three exact archives and checksums live at `rwth-phoenix-2014-t/features/author/PHOENIX2014T/` on Modal Volume `datasets`; see `reproduction.json`. Processing is non-commercial research under CC BY-NC-SA 4.0 on the project cloud.

The experiment consumes precomputed features and does not decode video, so `simple-video-utils` is not invoked.

## Environment and patches

The image pins the published Python 3.7, PyTorch 1.4.0, TorchText 0.5.0 stack on CUDA 11.4.3 and runs on one T4. No source or config-value patch is applied. For packaging, the Dockerfile omits two conda self-management pins, replaces three unavailable patch releases (TensorBoard 2.1.2 to 2.1.1, tensorflow-estimator 2.1.2 to 2.1.0, and warmup-scheduler 0.1.1 to 0.3), and pins `typing-extensions` for the Python 3.7 child.

## Execution evidence

The structured run policy was backfilled from the pre-migration report (whose
SHA-256 is retained in `reproduction.json`). The committed 24-hour launcher
ceiling is preserved; declaration time, retry maximum, GPU-hour ceiling, and
cost ceiling were not recorded and remain explicit unknowns.

| Run | Modal IDs | Hardware | Seed | Time | Terminal state | Evidence |
| --- | --- | --- | ---: | ---: | --- | --- |
| `full-seed-42` | app `ap-7aW0BiKItDEIHlMKv525Y7`; call `fc-01M0FSJMYJPDHG8GP8MM794CN3` | Tesla T4 15 GiB | 42 | 5,574.76 s | exit 0 | `reproduction.json` run/artifact IDs |

The authors' plateau rule stopped training normally at step 4,400 and selected checkpoint step 2,600. The full development search and fixed-parameter test evaluation then completed in the same upstream command without retry.

## Guesses and deviations

- The paper does not state the Table 4 seed; seed 42 comes from the published config.
- The repository config searches translation alphas -1 through 5 and uses `learning_rate_min: 1e-7`; the paper says alphas 0 through 2 and `1e-6`. This reproduction treats the published code/config as the executable recipe.
- The paper does not identify the original hardware. The legacy framework ran without modification on an available T4.

## Attempts, failures, and dead ends

The discarded 11-BLEU attempt used a third-party feature mirror whose archive sizes and vocabulary differed from the authors' data. Once the authors' restored files were found, that mirror path and its bespoke tooling were removed. The retained run used the official restored artifacts directly and reached 19.15 dev / 18.49 test BLEU-4.

## Candidate flags, ethics, and human evaluation

This was a direct assignment rather than a queue candidate. It introduced no participant interaction or human evaluation and processed an existing licensed benchmark only within project infrastructure. No dataset content is committed or uploaded with the model.

## Author and team contact

No new author or Team S/R contact was required. The public author response supplied the previously missing feature files, and the run stayed within the declared single-GPU compute gate.

## Use and limitations

The published checkpoint is an independent reproduction, not an author checkpoint. Use it only with the pinned `neccam/slt` implementation and included upstream config. It expects precomputed pami0 features, is licensed conservatively under CC BY-NC-SA 4.0, and is not production-ready or evidence that the paper's numerical claim succeeds or fails.

Selected checkpoint: step 2,600; 333,937,813 bytes; SHA-256 `e5c34ece5e41039bc0997281352eb4c8a0d10da53ba7da23af40c7a67d3fa8b8`.
