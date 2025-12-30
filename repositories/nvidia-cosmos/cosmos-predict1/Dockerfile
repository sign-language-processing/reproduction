# Cosmos-Predict1 World Foundation Models
FROM nvcr.io/nvidia/pytorch:25.12-py3

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install ffmpeg 4 from source (required for decord)
COPY libraries/ffmpeg/install_from_source.sh /tmp/install_ffmpeg.sh
RUN bash /tmp/install_ffmpeg.sh

# Install decord from source
COPY libraries/decord/install_from_source.sh /tmp/install_decord.sh
RUN bash /tmp/install_decord.sh

# Clone the cosmos-predict1 repository
RUN rm -rf /workspace/* && \
    git clone https://github.com/nvidia-cosmos/cosmos-predict1.git /workspace

WORKDIR /workspace

# Install requirements, skipping packages already in base image
# Base image includes: torch, torchvision, torchaudio, flash-attn, apex
RUN grep -v -E "^(torch|torchvision|torchaudio|apex)" requirements.txt > requirements_temp.txt && \
    pip install --no-cache-dir -r requirements_temp.txt && \
    rm requirements_temp.txt


CMD ["/bin/bash"]
