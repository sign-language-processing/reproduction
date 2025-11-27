#!/bin/bash
set -euo pipefail

echo "Installing FFmpeg from source..."

# Remove any existing ffmpeg installation
apt-get remove -y ffmpeg 2>/dev/null || true

# Install build dependencies
apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    pkg-config \
    yasm \
    nasm \
    git \
    libx264-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone and build FFmpeg
git clone --branch release/4.3 --depth 1 https://github.com/FFmpeg/FFmpeg.git /tmp/ffmpeg
cd /tmp/ffmpeg

./configure \
    --prefix=/usr \
    --disable-everything \
    --enable-demuxer=mov \
    --enable-muxer=mp4 \
    --enable-decoder=h264 \
    --enable-encoder=libx264 \
    --enable-pic \
    --enable-shared

make -j"$(nproc)"
make install

# Cleanup
cd /
rm -rf /tmp/ffmpeg

echo "FFmpeg installation complete!"
ffmpeg -version
