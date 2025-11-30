#!/bin/bash
set -euo pipefail

# Accept install directory as parameter, default to /workspace
INSTALL_DIR="${1:-/workspace}"

echo "Installing GroundingDINO from source to ${INSTALL_DIR}..."

# Install system dependencies
apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Clone GroundingDINO repository
echo "Cloning GroundingDINO..."
rm -rf "${INSTALL_DIR}/GroundingDINO"
git clone https://github.com/IDEA-Research/GroundingDINO.git "${INSTALL_DIR}/GroundingDINO"
cd "${INSTALL_DIR}/GroundingDINO"

# Patch CUDA code for PyTorch 2.10+ compatibility
# Replace deprecated value.type() with value.scalar_type()
echo "Applying CUDA compatibility patch..."
sed -i 's/AT_DISPATCH_FLOATING_TYPES(value\.type()/AT_DISPATCH_FLOATING_TYPES(value.scalar_type()/g' \
    groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu

# Install requirements, skipping packages already in base image
echo "Installing Python requirements..."
grep -v -E "^(torch|torchvision|numpy|pycocotools)$" requirements.txt > requirements_filtered.txt
pip install --no-cache-dir -r requirements_filtered.txt
rm requirements_filtered.txt

# Build and install GroundingDINO with CUDA extensions
# Use --no-build-isolation to use pre-installed torch from base image
# Set TORCH_CUDA_ARCH_LIST for CUDA 13 (sm_70/75 dropped)
echo "Building and installing GroundingDINO..."
TORCH_CUDA_ARCH_LIST="8.0 8.6 8.7 8.9 9.0 12.1" pip install --no-cache-dir --no-build-isolation -e .

# Install huggingface_hub for model downloads
echo "Installing huggingface_hub..."
pip install --no-cache-dir huggingface_hub

echo "GroundingDINO installation complete!"
echo "Installed at: ${INSTALL_DIR}/GroundingDINO"
