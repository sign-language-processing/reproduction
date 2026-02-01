# VFRTok Docker Reproduction

## Original Repository
https://github.com/KlingTeam/VFRTok

## Description
VFRTok is a variable frame rate video tokenizer that uses Vision Transformers (ViT) with rotary position embeddings. It's part of a NeurIPS 2025 paper on video tokenization.

## Requirements
- GPU with CUDA support (uses DeepSpeed, flash-attention, bfloat16)
- ~1.5GB for model weights
- Built on NVIDIA PyTorch container with FFmpeg 4.4 compiled for decord compatibility

## Model Weights
Pretrained weights are downloaded from HuggingFace: `KwaiVGI/VFRTok`
- `vfrtok-l.bin` (718 MB) - Large model

## Build Command
```bash
docker build \
  -t vfrtok:latest \
  -f repositories/KlingTeam/VFRTok/Dockerfile \
  .
```

## Run Commands

### Test inference with sample video (verified working)
```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  vfrtok:latest \
  deepspeed inference.py -i /workspace/VFRTok/test.csv -o /workspace/VFRTok/outputs \
    --config configs/vfrtok-l.yaml --ckpt vfrtok-l.bin --enc_fps 24
```

### Interactive shell
```bash
docker run --rm -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  vfrtok:latest bash
```

### Inference with your own videos
1. Prepare a CSV file with a `video_path` column containing paths to your videos
2. Mount the directory containing your videos
3. Run inference:
```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /path/to/your/videos:/data \
  vfrtok:latest \
  deepspeed inference.py -i /data/your_videos.csv -o /data/outputs \
    --config configs/vfrtok-l.yaml --ckpt vfrtok-l.bin --enc_fps 24
```

## Notes
- The Dockerfile builds FFmpeg 4.4 from source for decord compatibility (newer FFmpeg has breaking API changes)
- Uses a test video generated with ffmpeg testsrc
- Model weights are downloaded during build from HuggingFace
- The `--ipc=host --ulimit memlock=-1 --ulimit stack=67108864` flags are recommended by NVIDIA for PyTorch containers
- Warnings about "vit_*" registry overwrites are normal and can be ignored

## Asymmetric Frame Rate Inference
For different input/output frame rates:
```bash
deepspeed inference.py -i /workspace/VFRTok/test.csv -o /workspace/VFRTok/outputs \
  --config configs/vfrtok-l.yaml --ckpt vfrtok-l.bin \
  --enc_fps 24 --dec_fps 60
```
