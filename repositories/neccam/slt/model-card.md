---
license: cc-by-nc-sa-4.0
library_name: pytorch
tags:
  - sign-language-translation
  - sign-language-recognition
  - reproduction
  - rwth-phoenix-weather-2014t
datasets:
  - rwth-phoenix-weather-2014t
---

# REPRO-SIGN reproduction: neccam/slt

This is the seed-42 checkpoint from an independent reproduction of Camgoz et al., “Sign Language Transformers: Joint End-to-End Sign Language Recognition and Translation” (CVPR 2020). It is not an author-released checkpoint and does not fully match the paper numerically.

| Split | Metric | Paper | Reproduction |
| --- | --- | ---: | ---: |
| dev | WER | 35.13 | 53.6696 |
| dev | BLEU-4 | 21.73 | 19.1460 |
| test | WER | 33.75 | 54.0033 |
| test | BLEU-4 | 21.22 | 18.4889 |

Development selected recognition beam 10 and translation beam 2 with alpha 2. Training stopped normally at step 4,400 and selected checkpoint step 2,600.

## Provenance

- Code: `neccam/slt` commit `90588825f6229474bc19ac7a6b30ea3116635ba3`
- Config: the upstream `configs/sign.yaml`, with only data/output paths replaced; SHA-256 `89242e5195403c5721cf2430236d70817a5de17c8d648bbc9d5e3a214fd07183`
- Checkpoint: `model.ckpt`, 333,937,813 bytes; SHA-256 `e5c34ece5e41039bc0997281352eb4c8a0d10da53ba7da23af40c7a67d3fa8b8`
- Data: authors' restored RWTH-PHOENIX-Weather 2014T pami0 features, 7,095/519/642 train/dev/test records
- Environment: Python 3.7, PyTorch 1.4.0, fp32, one Tesla T4

Use this checkpoint with the pinned implementation and included `config.yaml`. It expects precomputed pami0 features and does not decode video. It is provided for non-commercial reproducibility research under the conservative CC BY-NC-SA 4.0 terms inherited from the training data; it is not production-ready and is not evidence that the paper's numerical claim succeeds or fails.
