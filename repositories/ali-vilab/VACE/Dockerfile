# VACE - All-in-One Video Creation and Editing
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

WORKDIR /workspace

# Clone the VACE repository
RUN git clone https://github.com/ali-vilab/VACE.git

WORKDIR /workspace/VACE

# Install requirements, skipping packages already in base image or unavailable
# - torch, torchvision: use base image versions (PyTorch 2.10)
# - flash_attn: pre-installed in base image
# - decord: installed from source above
# - onnxruntime-gpu: no arm64 wheels, use onnxruntime instead
RUN grep -v -E "^(torch|torchvision|flash_attn|decord|onnxruntime-gpu)" requirements/framework.txt > requirements_filtered.txt && \
    pip install --no-cache-dir -r requirements_filtered.txt onnxruntime && \
    rm requirements_filtered.txt

CMD ["python", "vace/vace_pipeline.py", "--help"]
