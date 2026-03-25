# VFRTok - Variable Frame Rate Video Tokenizer
FROM gogamza/unsloth-vllm-gb10:latest

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install ffmpeg 4 from source (required for decord)
COPY libraries/ffmpeg/install_from_source.sh /tmp/install_ffmpeg.sh
RUN bash /tmp/install_ffmpeg.sh

# Install decord from source
COPY libraries/decord/install_from_source.sh /tmp/install_decord.sh
RUN bash /tmp/install_decord.sh

# Clone VFRTok repository
RUN git clone https://github.com/KlingTeam/VFRTok.git /workspace/VFRTok

WORKDIR /workspace/VFRTok

# Install requirements without packages that conflict with base image
RUN pip install --no-cache-dir \
    diffusers einops timm peft hjson ruamel.yaml \
    scikit-image torchmetrics tensorboard tensorboardX pandas deepspeed redis \
    mediapy simple-video-utils

CMD ["/bin/bash"]
