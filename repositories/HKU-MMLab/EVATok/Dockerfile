FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ffmpeg < 5 required for decord
COPY libraries/ffmpeg/install_from_source.sh /tmp/install_ffmpeg.sh
RUN bash /tmp/install_ffmpeg.sh

COPY libraries/decord/install_from_source.sh /tmp/install_decord.sh
RUN bash /tmp/install_decord.sh

RUN rm -rf /workspace/* && git clone https://github.com/HKU-MMLab/EVATok.git /workspace

WORKDIR /workspace

# Skip torch/torchvision/torchaudio — already in base image
RUN grep -v -E "^(torch==|torchvision==|torchaudio==)" requirements.txt > /tmp/req.txt && \
    pip install --no-cache-dir -r /tmp/req.txt && \
    rm /tmp/req.txt

# Patch torch.load calls: PyTorch >=2.6 changed default weights_only to True
# but EVATok checkpoints use numpy scalars which require weights_only=False
COPY repositories/HKU-MMLab/EVATok/patch_torch_load.py /tmp/patch_torch_load.py
RUN python3 /tmp/patch_torch_load.py

CMD ["/bin/bash"]
