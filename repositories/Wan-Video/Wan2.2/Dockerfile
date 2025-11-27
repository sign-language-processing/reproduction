# Wan2.2 Video Generation Model
FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV DEBIAN_FRONTEND=noninteractive

# Install libgl1 for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install ffmpeg 4 from source (required for decord)
COPY libraries/ffmpeg/install_from_source.sh /tmp/install_ffmpeg.sh
RUN bash /tmp/install_ffmpeg.sh

# Install decord from source
COPY libraries/decord/install_from_source.sh /tmp/install_decord.sh
RUN bash /tmp/install_decord.sh

# Clone the Wan2.2 repository
RUN rm -rf /workspace/* && \
    git clone https://github.com/Wan-Video/Wan2.2.git /workspace

WORKDIR /workspace

# Install requirements, skipping packages already in base image
RUN grep -v -E "(torch|torchvision|torchaudio|flash_attn|diffusers)" requirements.txt > requirements_temp.txt && \
    pip install --no-cache-dir -r requirements_temp.txt && \
    rm requirements_temp.txt

# Install compatible diffusers, peft, and huggingface_hub CLI
RUN pip install --no-cache-dir diffusers==0.31.0 peft==0.14.0 "huggingface_hub[cli]"

CMD ["/bin/bash"]
