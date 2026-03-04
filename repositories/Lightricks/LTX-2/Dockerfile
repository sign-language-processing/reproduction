# LTX-2 Video VAE Tokenizer
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

WORKDIR /workspace

# Install dependencies
RUN pip install --no-cache-dir \
    mediapy \
    simple-video-utils \
    git+https://github.com/huggingface/diffusers.git

CMD ["/bin/bash"]
