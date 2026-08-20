# Sign Language Transformers (CVPR 2020)

This directory reproduces the default joint recognition/translation experiment from [neccam/slt](https://github.com/neccam/slt), pinned at commit `90588825f6229474bc19ac7a6b30ea3116635ba3`. The target contract is the `lambda_R=1.0`, `lambda_T=1.0` row of Table 4 in the paper.

The published implementation trains from three gzip-pickled files of precomputed, frame-level 1024-D CNN features. It does not decode the PHOENIX14T videos during training, so this recipe intentionally does not add a video decoding path. The raw videos and annotations remain available under `/datasets/rwth-phoenix-2014-t`; the pinned feature artifacts live under `/datasets/rwth-phoenix-2014-t/features`.

The authors' download links are no longer available. `data.sh` therefore verifies a public mirror at pinned Hugging Face dataset revision `d5c32f2cd1cf27a26083671532a32e75c98dbae3`. That mirror declares pickle protocol 5 even though it uses no protocol-5 opcodes, while the published environment uses Python 3.7. The data entry point preserves each downloaded archive and creates a checksum-pinned `.protocol4` derivative by changing only the two-byte protocol declaration; decompressed payload bytes are otherwise identical. PHOENIX14T is used under its non-commercial research terms and is never copied to the repository or Hugging Face model repository.

## Reproduce

All Modal operations use the repository wrapper and therefore fail closed unless the active workspace is `repro-sign`.

```bash
# Populate/verify the three checksum-pinned feature files once.
repositories/neccam/slt/scripts/data.sh

# Train/evaluate on small real subsets, including checkpoint save/reload.
repositories/neccam/slt/scripts/dry_run.sh

# Launch the seed-42 full run on one A100 and monitor it in Modal.
repositories/neccam/slt/scripts/train.sh

# Re-evaluate the persisted best full checkpoint if needed.
repositories/neccam/slt/scripts/eval.sh
```

`datasets` is mounted read-only for dry/full experiments. `huggingface-cache` is mounted read-write at `/cache/huggingface` with `HF_HOME` and `HF_HUB_CACHE` set. Checkpoints, logs, predictions, and metrics are written to the v2 Modal Volume `neccam-slt-results`, never to the shared cache.

After terminal evaluation, publish the permitted model artifacts with `scripts/publish.sh`; the script copies only the checkpoint/config/metrics to a temporary directory and uploads them to the `repro-sign` Hugging Face organization.

## Citation

```bibtex
@inproceedings{camgoz2020sign,
  author = {Necati Cihan Camgoz and Oscar Koller and Simon Hadfield and Richard Bowden},
  title = {Sign Language Transformers: Joint End-to-End Sign Language Recognition and Translation},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
  year = {2020}
}
```
