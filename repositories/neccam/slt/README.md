# Sign Language Transformers (CVPR 2020)

This directory reproduces the default joint recognition/translation experiment from [neccam/slt](https://github.com/neccam/slt), pinned at commit `90588825f6229474bc19ac7a6b30ea3116635ba3`. The target contract is the `lambda_R=1.0`, `lambda_T=1.0` row of Table 4 in the paper.

The published implementation trains from three gzip-pickled files of precomputed, frame-level 1024-D CNN features. It does not decode the PHOENIX14T videos during training, so this recipe intentionally does not add a video decoding path. The raw videos and annotations remain available under `/datasets/rwth-phoenix-2014-t`; the pinned feature artifacts live under `/datasets/rwth-phoenix-2014-t/features`.

The authors' download links are no longer available. `data.sh` therefore verifies a public mirror at pinned Hugging Face dataset revision `d5c32f2cd1cf27a26083671532a32e75c98dbae3`. That mirror declares pickle protocol 5 even though it uses no protocol-5 opcodes, while the published environment uses Python 3.7. The data entry point preserves each downloaded archive and creates a checksum-pinned `.protocol4` derivative by changing only the two-byte protocol declaration; decompressed payload bytes are otherwise identical. PHOENIX14T is used under its non-commercial research terms and is never copied to the repository or Hugging Face model repository.

## Result

The seed-42 pipeline completed all four targets. Pipeline completeness and numerical agreement are separate; the reproduced scores differ substantially from Table 4.

| Split | Metric | Paper | Reproduced | Difference |
| --- | --- | ---: | ---: | ---: |
| dev | WER | 35.13 | 80.5444 | +45.4144 |
| dev | BLEU-4 | 21.73 | 11.3566 | -10.3734 |
| test | WER | 33.75 | 77.7413 | +43.9913 |
| test | BLEU-4 | 21.22 | 11.1158 | -10.1042 |

Development selected recognition beam 10 and translation beam 4 with alpha -1. The selected step-6200 checkpoint has SHA-256 `72f6cac1723463f9c7781051ab9bcd77c34ed3675637389bf09f0f6a4bc7d576` and is published at immutable [repro-sign/neccam-slt revision 6ae275a](https://huggingface.co/repro-sign/neccam-slt/tree/6ae275aec44b59d129f22fab36a7120f05f94eb3). See `report.md` for the complete evidence and deviations.

## Reproduce

All Modal operations use the repository wrapper and therefore fail closed unless the active workspace is `repro-sign`.

```bash
# Populate/verify the three checksum-pinned feature files once.
repositories/neccam/slt/scripts/data.sh

# Train/evaluate on small real subsets, including checkpoint save/reload.
repositories/neccam/slt/scripts/dry_run.sh

# Launch the seed-42 full run on one A100 and monitor it in Modal.
repositories/neccam/slt/scripts/train.sh

# Re-evaluate the persisted best full checkpoint and persist exact metrics.
repositories/neccam/slt/scripts/eval.sh

# Upload the permitted checkpoint, config, metrics, and checksums.
repositories/neccam/slt/scripts/publish.sh
```

`datasets` is mounted read-only for dry/full experiments. `huggingface-cache` is mounted read-write at `/cache/huggingface` with `HF_HOME` and `HF_HUB_CACHE` set. Checkpoints, logs, predictions, and metrics are written to the v2 Modal Volume `neccam-slt-results`, never to the shared cache.

After terminal evaluation, `scripts/publish.sh` copies only the checkpoint, config, metrics, raw result JSON, model card, and checksum manifest to a temporary directory and uploads them to the `repro-sign` Hugging Face organization. The published model checkpoint is licensed conservatively as CC BY-NC-SA 4.0; no dataset content is uploaded.

## Citation

```bibtex
@inproceedings{camgoz2020sign,
  author = {Necati Cihan Camgoz and Oscar Koller and Simon Hadfield and Richard Bowden},
  title = {Sign Language Transformers: Joint End-to-End Sign Language Recognition and Translation},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
  year = {2020}
}
```
