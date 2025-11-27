#!/bin/bash
set -euo pipefail

echo "Installing Decord from source..."

# Verify ffmpeg 4.x is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg not found. Install ffmpeg 4.x first."
    exit 1
fi

# Check ffmpeg version - look for version 4.x in version string or library versions
FFMPEG_VERSION=$(ffmpeg -version 2>&1)
if echo "$FFMPEG_VERSION" | grep -q "version [45]\."; then
    echo "Found ffmpeg 4.x/5.x - proceeding with Decord installation..."
elif echo "$FFMPEG_VERSION" | grep -q "libavcodec.*58\." || echo "$FFMPEG_VERSION" | grep -q "libavcodec.*59\."; then
    # FFmpeg 4.x has libavcodec 58.x, FFmpeg 5.x has 59.x
    echo "Found compatible ffmpeg (libavcodec 58.x/59.x) - proceeding with Decord installation..."
else
    echo "Warning: Could not verify ffmpeg version. Found:"
    echo "$FFMPEG_VERSION" | head -3
    echo "Proceeding anyway..."
fi

# Install build dependencies
apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Clone and build decord
echo "Cloning Decord repository..."
rm -rf /tmp/decord
git clone --recursive https://github.com/dmlc/decord /tmp/decord
cd /tmp/decord

echo "Building Decord..."
mkdir -p build && cd build
cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"

echo "Installing Decord Python package..."
cd ../python
pip install --no-cache-dir .

# Cleanup
cd /
rm -rf /tmp/decord

echo "Decord installation complete!"
python3 -c "import decord; print(f'Decord version: {decord.__version__}')"
