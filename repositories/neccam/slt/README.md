# Sign Language Transformers (CVPR 2020)

This is a thin launcher for the authors' [neccam/slt](https://github.com/neccam/slt) implementation at commit `90588825f6229474bc19ac7a6b30ea3116635ba3`. It reproduces the joint `lambda_R=1`, `lambda_T=1` row of Table 4. There is no local model implementation or copied experiment config: `modal_app.py` substitutes only the shared data and output paths into the upstream config, then runs the upstream `python -m signjoey train` entry point.

## Data

In April 2026 the author restored the three original pami0 files in [neccam/nslt#38](https://github.com/neccam/nslt/issues/38#issuecomment-4330442225). This recipe uses those files directly, without annotation repair, feature conversion, or local data code. Their exact sizes and hashes are recorded in `evidence/data/features.json`.

Training consumes the authors' precomputed features and does not decode video.

## Run

All Modal calls must use the repository wrapper so they fail closed outside the `repro-sign` workspace. The function mounts `datasets` read-only, uses `huggingface-cache` as the shared Hugging Face cache, and writes checkpoints/results to the v2 Volume `neccam-slt-results`.

```bash
./setup.sh
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run \
  repositories/neccam/slt/modal_app.py::train
```

The container uses Python 3.7, PyTorch 1.4.0, and TorchText 0.5.0. A T4 is intentional: it executes that legacy CUDA build directly, so no source patch is needed. The small packaging-only adjustments are documented inline in the Dockerfile.

After terminal evaluation, publish the selected checkpoint, runtime upstream config, metrics, run metadata, and checksums:

```bash
repositories/neccam/slt/scripts/publish.sh
```

## Result

The terminal scores and immutable checkpoint revision are recorded in `metrics.json` and `report.md`.

## Citation

```bibtex
@inproceedings{camgoz2020sign,
  author = {Necati Cihan Camgoz and Oscar Koller and Simon Hadfield and Richard Bowden},
  title = {Sign Language Transformers: Joint End-to-End Sign Language Recognition and Translation},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
  year = {2020}
}
```
