---
license: cc-by-nc-sa-3.0
tags:
  - sign-language-translation
  - reproduction
datasets:
  - rwth-phoenix-weather-2014t
---

# Neural Sign Language Translation reproduction

This is a faithful run of the authors' published `neccam/nslt` Luong recipe for Camgoz et al., *Neural Sign Language Translation* (CVPR 2018). It targets the Luong row of Table 5, not the separate Bahdanau system in Table 6.

**Preference level:** 2

**Status:** `complete`

**Numerical agreement:** fully reproduced.

The authors' full 150,000-step run completed. Its terminal best-model loop selected checkpoint 118,000 on development BLEU-4 and evaluated that fixed checkpoint on both splits.

## Target

| Split | Metric | Paper | Reproduced | Difference |
| --- | --- | ---: | ---: | ---: |
| dev | BLEU-4 | 10.00 | 10.260990 | +0.260990 |
| test | BLEU-4 | 9.00 | 9.603572 | +0.603572 |

The pinned repository's unsmoothed, case-sensitive, whitespace-tokenized corpus BLEU implementation is used. It prints BLEU-1 through BLEU-4 and returns BLEU-4.

These are the terminal upstream evaluations of checkpoint 118,000, not the final-step checkpoint. Exact values, raw prediction hashes, run IDs, and provenance are in `reproduction.json`.

## Pinned sources

| Item | Source | Role |
| --- | --- | --- |
| Paper | [CVF PDF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Camgoz_Neural_Sign_Language_CVPR_2018_paper.pdf) (SHA-256 `642e088ee56eeaa6aa641bc53438f695a6b5f4942bd0b6bd87e9ae53db41205b`) | Targets and protocol |
| Code | [neccam/nslt at `0695158`](https://github.com/neccam/nslt/tree/06951580b58f04b9cd64efcf61aeca36011031d3) | Authors' training, evaluation, and metric implementation |
| AlexNet initialization | [bvlc_alexnet.npy](https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy) (SHA-256 `1294ef51496c52f2e4879ee3b5e1da22c35d61f3240937491b4469403d5428cc`) | Paper-required ImageNet initialization |

## Data and execution

The run uses RWTH-PHOENIX-Weather 2014T v3 on the shared Modal `datasets` v2 Volume: 7,096 train, 519 dev, and 642 test videos. It uses the shared `huggingface-cache` v2 Volume at `/cache/huggingface`.

The upstream manifests expect MATLAB-resized frame directories. With approval for this reproduction, the retained one-concern patch reads the supplied MP4s using `simple-video-utils`, resizes frames to 227x227 with OpenCV cubic interpolation, and preserves the upstream BGR mean-subtraction convention. This is a documented preprocessing deviation: direct MP4 decode/OpenCV resize is not pixel-identical to the authors' MATLAB resize and intermediate image files.

Run from the repository root with authenticated `repro-sign` Modal access:

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach \
  papers/camgoz-2018-nslt/scripts/modal_app.py::launch_train
```

The launcher invokes the authors' module directly with its README recipe: four residual 1,000-unit GRU layers, Luong attention, reversed source, batch size 1, Adam 1e-5, seed 285, and 150,000 training steps. The results volume is `camgoz-nslt-results`; a 24-hour Modal segment resumes from the verified upstream checkpoint rather than restarting.

## Retained patches

- `0001-compatibility.patch`: TensorFlow/Python compatibility required to execute the pinned source.
- `0002-alexnet-initialization.patch`: preserves the paper-required ImageNet initialization, which upstream otherwise overwrites with global variable initialization on a fresh run.
- `0003-simple-video-utils.patch`: approved direct-video input path described above.
- `0004-checkpoint-resume.patch`: saves paired resume state with every checkpoint and retains the upstream evaluation cadence over controlled Modal segments.

No datasets, raw videos, outputs, or checkpoints are committed to this repository. The permitted selected checkpoint bundle, hparams, and exact predictions are preserved at [repro-sign/camgoz-nslt revision `7851726`](https://huggingface.co/repro-sign/camgoz-nslt/tree/7851726a660a9441fa8e1817a8818695dee1df8b); the repository's latest revision carries this model card and evidence record. The 503,848,900-byte model data file has SHA-256 `bbace325a51197db547c18493870c5c5a87b289939b7505bd7ce73ed79831a05`.

## Execution evidence

The structured run policy was backfilled from the pre-migration report (whose
SHA-256 is retained in `reproduction.json`). The launcher’s 24-hour wall ceiling
and the documented 80 GPU-hour ceiling are preserved; declaration time, retry
maximum, and cost ceiling were not recorded and remain explicit unknowns.

The first single-A100 segment reached a verified checkpoint at step 91,000 before Modal's 24-hour function limit. The second segment resumed it without restarting, skipped the 6,132 already-seen examples, trained through step 150,000, and completed the upstream terminal best-model evaluation. The selected checkpoint is step 118,000.

| Segment | Modal app | Terminal state | Duration |
| --- | --- | --- | ---: |
| Initial | `ap-YIhshIz2SxWpZ8ga3jTb6i` | controlled timeout at verified checkpoint | 24 h |
| Resumed | `ap-ndaVyaD00TOsiDttMuXGvX` | exit 0, all target scores produced | 16.17 h |

The source checkpoint, prediction files, and configuration are recorded with immutable Hugging Face revision URLs and SHA-256 hashes in `reproduction.json`.
