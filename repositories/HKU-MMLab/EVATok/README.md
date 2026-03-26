# EVATok

GitHub: https://github.com/HKU-MMLab/EVATok
Models: https://huggingface.co/YuuTennYi/EVATok

## Build

```bash
docker build -t evatok:latest -f repositories/HKU-MMLab/EVATok/Dockerfile .
```

## Run (upstream reconstruction — GIF output)

Applies EVATok's built-in `reconstruction_video_ddp.py` to sign language test videos.
Downloads ~2 GB of model files on first run (cached in `/shared/.cache`).

```bash
docker run --rm --gpus all \
  -v /mnt/rylo-tnas/users/rotem/sign/video-tokenizer/cosmos-predict1/cosmos_predict1/tokenizer/test_data/cropped:/input_videos:ro \
  -v /shared/.cache:/root/.cache \
  -v $(pwd)/repositories/HKU-MMLab/EVATok/output:/workspace/results \
  -e PYTHONPATH=/workspace \
  evatok:latest \
  bash -c "
    mkdir -p qualitative_test_set_gifs_select/signs &&
    for f in /input_videos/*.mp4; do ln -sf \"\$f\" qualitative_test_set_gifs_select/signs/; done &&
    mkdir -p ckpts &&
    python -c \"
from huggingface_hub import hf_hub_download
hf_hub_download('YuuTennYi/EVATok', 'VQ_SB_final_with_router_ucf_k600_1000k.pt', local_dir='ckpts')
hf_hub_download('YuuTennYi/EVATok', 'router_w_lpips_1.2_50k.pt', local_dir='ckpts')
\" &&
    torchrun --standalone --nnodes=1 --nproc_per_node=1 \
      tokenizer/reconstruction_video_ddp.py \
      --model-config configs/vq/VQ_SB_final_with_router_w_lpips_1.2.yaml \
      --vq-ckpt ckpts/VQ_SB_final_with_router_ucf_k600_1000k.pt \
      --qualitative
  "
```

Output GIFs are written to `repositories/HKU-MMLab/EVATok/output/reconstructions/vq/ViDTok-ucf&k600-.../`.
Each video produces `NNNNNN.gif` (reconstruction) and `NNNNNN_gt.gif` (ground truth).

## Run (full-video side-by-side — MP4 output)

`run_full_video.py` is a custom script that processes full-length videos (not just 16-frame samples).
It chunks each video into 16-frame windows, runs encode→decode on each chunk, and writes a side-by-side
GT|Reconstruction MP4.

```bash
docker run --rm --gpus all \
  -v /mnt/rylo-tnas/users/rotem/sign/video-tokenizer/cosmos-predict1/cosmos_predict1/tokenizer/test_data/cropped:/input_videos:ro \
  -v /shared/.cache:/root/.cache \
  -v $(pwd)/repositories/HKU-MMLab/EVATok/output:/workspace/results \
  -e PYTHONPATH=/workspace \
  evatok:latest \
  bash -c "
    mkdir -p ckpts &&
    python -c \"
from huggingface_hub import hf_hub_download
hf_hub_download('YuuTennYi/EVATok', 'VQ_SB_final_with_router_ucf_k600_1000k.pt', local_dir='ckpts')
hf_hub_download('YuuTennYi/EVATok', 'router_w_lpips_1.2_50k.pt', local_dir='ckpts')
\" &&
    python run_full_video.py \
      --input-dir /input_videos \
      --output-dir results/full_video \
      --model-config configs/vq/VQ_SB_final_with_router_w_lpips_1.2.yaml \
      --vq-ckpt ckpts/VQ_SB_final_with_router_ucf_k600_1000k.pt \
      --display-size 512
  "
```

Options:
- `--display-size N` — upscale both GT and reconstruction to NxN for the output video (default: 128, same as model input)
- `--fps F` — override output FPS (default: source video FPS)
- `--videos stem1 stem2 ...` — process only specific videos by filename stem (default: all `*.mp4` in input dir)

Output MP4s are written to `results/full_video/<stem>_sbs.mp4`.

**Note:** `run_full_video.py` must be copied into the container. Add to the Dockerfile or bind-mount it:
```bash
-v $(pwd)/repositories/HKU-MMLab/EVATok/run_full_video.py:/workspace/run_full_video.py:ro
```

## Notes

- `--qualitative` mode (upstream script) reads videos from `qualitative_test_set_gifs_select/` (hardcoded), structured as `class_dir/video.mp4`. Test videos are symlinked into a `signs/` subdir.
- Model config: `VQ_SB_final_with_router_w_lpips_1.2.yaml` (UCF+K600 trained, with adaptive router).
- The router checkpoint path is resolved relative to `WORKDIR` as `ckpts/router_w_lpips_1.2_50k.pt` (hardcoded in the YAML).
- `PYTHONPATH=/workspace` is required so the `tokenizer`, `datasets`, `utils` packages resolve correctly.
- `patch_torch_load.py` is applied at build time: EVATok checkpoints use numpy scalars, which PyTorch >=2.6 rejects unless `weights_only=False`.