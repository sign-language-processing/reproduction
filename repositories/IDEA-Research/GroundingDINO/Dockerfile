# GroundingDINO - Open-set Object Detection
FROM nvcr.io/nvidia/pytorch:25.11-py3

# Install libgl1 for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Clone GroundingDINO repository
RUN rm -rf /workspace/* && \
    git clone https://github.com/IDEA-Research/GroundingDINO.git /workspace

WORKDIR /workspace

# Patch CUDA code for PyTorch 2.10+ compatibility
# Replace deprecated value.type() with value.scalar_type()
RUN sed -i 's/AT_DISPATCH_FLOATING_TYPES(value\.type()/AT_DISPATCH_FLOATING_TYPES(value.scalar_type()/g' \
    groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu

# Install requirements, skipping packages already in base image
RUN grep -v -E "^(torch|torchvision|numpy|pycocotools)$" requirements.txt > requirements_filtered.txt && \
    pip install --no-cache-dir -r requirements_filtered.txt && \
    rm requirements_filtered.txt

# Build and install GroundingDINO with CUDA extensions
# Use --no-build-isolation to use pre-installed torch from base image
# Set TORCH_CUDA_ARCH_LIST for CUDA 13 (sm_70/75 dropped)
RUN TORCH_CUDA_ARCH_LIST="8.0 8.6 8.7 8.9 9.0 12.1" pip install --no-cache-dir --no-build-isolation -e .

# Install huggingface_hub for model downloads
RUN pip install --no-cache-dir huggingface_hub

CMD ["bash"]
