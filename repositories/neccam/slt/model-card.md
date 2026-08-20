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

This repository contains the seed-42 checkpoint from an independent reproduction of Camgoz et al., “Sign Language Transformers: Joint End-to-End Sign Language Recognition and Translation” (CVPR 2020). It is not an author-released checkpoint and does not numerically match the paper.

## Results

| Split | Metric | Paper | Reproduction |
| --- | --- | ---: | ---: |
| dev | WER | 35.13 | 80.5444 |
| dev | BLEU-4 | 21.73 | 11.3566 |
| test | WER | 33.75 | 77.7413 |
| test | BLEU-4 | 21.22 | 11.1158 |

Development selected recognition beam 10 and translation beam 4 with alpha -1. The full raw metrics are in `modal-result.json`.

## Provenance

- Published code: `neccam/slt` commit `90588825f6229474bc19ac7a6b30ea3116635ba3`
- Reproduction config: `config.yaml`, SHA-256 `4c3781a1cd1de7236adae132aa19b43d003ed942bf5293226f14cad79b467023`
- Checkpoint: `model.ckpt`, 334,956,987 bytes, SHA-256 `72f6cac1723463f9c7781051ab9bcd77c34ed3675637389bf09f0f6a4bc7d576`
- Selected training step: 6,200
- Training data: RWTH-PHOENIX-Weather 2014T pami0 precomputed 1024-D features, 7,096/519/642 train/dev/test examples
- Environment: Python 3.7.13, PyTorch 1.7.1+cu110, fp32, one A100

The unavailable author-hosted features were replaced by the public `lavinal712/slt` mirror pinned at revision `d5c32f2cd1cf27a26083671532a32e75c98dbae3`. Mirror identity with the original files cannot be cryptographically proven. The reproduction applies one PyTorch integer-division compatibility patch and otherwise preserves the published architecture and supplied default configuration.

## Use and limitations

Load this checkpoint only with the pinned `neccam/slt` implementation and reproduction config. It expects precomputed pami0 features; it does not decode video. The checkpoint is provided for non-commercial reproducibility research under the conservative CC BY-NC-SA 4.0 terms inherited from the training data. Do not treat it as the paper's weights, a production model, or evidence that the paper's numerical claim succeeds or fails.
