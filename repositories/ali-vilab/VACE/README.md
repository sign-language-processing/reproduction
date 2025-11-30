# VACE Reproduction

https://github.com/ali-vilab/VACE

## Build

```bash
docker build -t ali-vilab-vace:latest -f repositories/ali-vilab/VACE/Dockerfile .
```

## Run

All run commands use NVIDIA recommended flags for PyTorch containers:
- `--ipc=host` - Use host's shared memory (required for PyTorch multiprocessing)
- `--ulimit memlock=-1` - Unlimited locked memory
- `--ulimit stack=67108864` - 64MB stack size

```bash
docker run --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /shared/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/output:/workspace/VACE/processed \
  -it ali-vilab-vace:latest bash
```

### Wan

Run inference:

```bash
# Download the Wan2.1-VACE model
hf download Wan-AI/Wan2.1-VACE-1.3B
ln -s ~/.cache/huggingface/hub/models--Wan-AI--Wan2.1-VACE-1.3B/snapshots/*/. models/Wan2.1-VACE-1.3B

# Download only the models you need from "VACE-Annotators" using "include"
hf download ali-vilab/VACE-Annotators --include "depth/*"
ln -s ~/.cache/huggingface/hub/models--ali-vilab--VACE-Annotators/snapshots/*/. models/VACE-Annotators

python vace/vace_pipeline.py \
  --base wan \
  --task depth \
  --video assets/videos/test.mp4 \
  --prompt 'xxx' \
```

### Swap Anything

#### Known Issue: Device Parameter Bug

The `vace_preproccess.py` script has a bug where it passes an invalid `device` parameter to the SAM2 model. This must be fixed before running swap_anything tasks.

**Fix:** Remove the invalid device parameter from the SAM2 initialization:

```bash
sed -i 's/cfg=task_cfg, device=f'"'"'cuda:{os.getenv("RANK", 0)}'"'"'/cfg=task_cfg/' /workspace/VACE/vace/vace_preproccess.py
```

#### Known Issue: Missing --mode Parameter

The `--mode` parameter is **required** for swap_anything but not documented. The mode is comma-separated: `<video_mode>,<image_mode>`.

Valid modes for video (inpainting): `salient`, `mask`, `bbox`, `masktrack`, `bboxtrack`, `label`, `caption`
Valid modes for image (reference): `salient`, `mask`, `bbox`, `salientmasktrack`, `salientbboxtrack`, `masktrack`

#### Running Swap Anything

```bash
# Download required annotator models
hf download ali-vilab/VACE-Annotators --include "salient/*"
hf download ali-vilab/VACE-Annotators --include "sam2/*"
ln -sf ~/.cache/huggingface/hub/models--ali-vilab--VACE-Annotators/snapshots/*/. models/VACE-Annotators

# Download a test video
wget -O your_name_what.mp4 https://media.spreadthesign.com/video/mp4/13/93875.mp4

# Apply the device parameter fix (required!)
sed -i 's/cfg=task_cfg, device=f'"'"'cuda:{os.getenv("RANK", 0)}'"'"'/cfg=task_cfg/' /workspace/VACE/vace/vace_preproccess.py

# Run preprocessing (note: --mode is required!)
python vace/vace_preproccess.py \
   --task "swap_anything" \
   --video "your_name_what.mp4" \
   --image "assets/images/girl.png" \
   --mode "salient,salient" \
   --pre_save_dir "./processed"
```

**Output:** Creates files in `processed/swap_anything/<timestamp>/`:
- `src_video-swap_anything.mp4` - Processed video
- `src_mask-swap_anything.mp4` - Mask video
- `src_ref_image_0-swap_anything.png` - Reference image

### LTX

Run inference:

```bash
python vace/vace_ltx_inference.py \
  --ckpt_path ./VACE-LTX-Video-0.9/ltx-video-2b-v0.9.safetensors \
  --src_video examples/videos/girl.mp4 \
  --prompt "A girl is walking"
```
