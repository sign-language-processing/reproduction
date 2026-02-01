FROM nvcr.io/nvidia/pytorch:25.11-py3

# Set non-interactive frontend for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including ffmpeg dev libraries for building decord
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    build-essential \
    cmake \
    libavcodec-dev \
    libavfilter-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# Build FFmpeg 4.4 from source for decord compatibility (newer FFmpeg has breaking API changes)
WORKDIR /tmp/ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    nasm \
    yasm \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* && \
    git clone --depth 1 -b n4.4.5 https://github.com/FFmpeg/FFmpeg.git . && \
    ./configure --prefix=/usr/local/ffmpeg4 \
        --enable-shared \
        --disable-static \
        --disable-doc \
        --disable-programs \
        --enable-gpl && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# Build and install decord from source using FFmpeg 4.4
WORKDIR /tmp/decord
ENV PKG_CONFIG_PATH=/usr/local/ffmpeg4/lib/pkgconfig:$PKG_CONFIG_PATH
ENV LD_LIBRARY_PATH=/usr/local/ffmpeg4/lib:$LD_LIBRARY_PATH
RUN git clone --recursive https://github.com/dmlc/decord.git . && \
    mkdir build && \
    cd build && \
    cmake .. -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release \
        -DFFMPEG_DIR=/usr/local/ffmpeg4 && \
    make -j$(nproc) && \
    cd ../python && \
    pip install --no-cache-dir . && \
    echo "/usr/local/ffmpeg4/lib" > /etc/ld.so.conf.d/ffmpeg4.conf && \
    ldconfig

# Set working directory
WORKDIR /workspace/VFRTok

# Clone the VFRTok repository
RUN git clone https://github.com/KlingTeam/VFRTok.git /workspace/VFRTok && \
    git checkout main

# Install ninja for faster flash-attn compilation
RUN pip install --no-cache-dir ninja

# Install flash-attn with memory-limited compilation
RUN MAX_JOBS=4 pip install --no-cache-dir flash-attn --no-build-isolation

# Install Python dependencies from requirements.txt
# Note: Some packages may already be in the base image, skip torch/torchvision/torchaudio
RUN pip install --no-cache-dir \
    accelerate==1.2.1 \
    diffusers==0.31.0 \
    einops==0.8.0 \
    huggingface-hub==0.27.0 \
    matplotlib==3.10.0 \
    peft==0.14.0 \
    ruamel.yaml==0.18.6 \
    safetensors==0.4.5 \
    timm==1.0.12 \
    transformers==4.47.1 \
    scikit-image \
    torchmetrics \
    tensorboard \
    tensorboardX \
    pandas \
    deepspeed \
    loguru \
    opencv-python \
    redis

# Download model weights from HuggingFace
RUN wget -q --show-progress \
    "https://huggingface.co/KwaiVGI/VFRTok/resolve/main/vfrtok-l.bin" \
    -O vfrtok-l.bin

# Download a sample video for testing using ffmpeg testsrc (most reliable)
RUN mkdir -p /workspace/VFRTok/sample_data && \
    ffmpeg -f lavfi -i testsrc=duration=2:size=256x256:rate=24 -c:v libx264 -pix_fmt yuv420p /workspace/VFRTok/sample_data/sample.mp4

# Create test CSV file
RUN echo "video_path" > /workspace/VFRTok/test.csv && \
    echo "/workspace/VFRTok/sample_data/sample.mp4" >> /workspace/VFRTok/test.csv

# Add metadata to CSV
RUN python scripts/add_metadata_to_csv.py -i /workspace/VFRTok/test.csv -o /workspace/VFRTok/test.csv --data_column video_path || true

# Set default command
CMD ["bash"]
