FROM nvcr.io/nvidia/pytorch:26.02-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY libraries/ffmpeg/install_from_source.sh /tmp/install_ffmpeg.sh
RUN bash /tmp/install_ffmpeg.sh

COPY libraries/decord/install_from_source.sh /tmp/install_decord.sh
RUN bash /tmp/install_decord.sh

WORKDIR /workspace

ENV INSTALLED_STABLE_PACKAGES="torch|torchvision|torchaudio|apex|flash-attn|decord"

RUN pip list --format=freeze \
    | grep -E "^($INSTALLED_STABLE_PACKAGES)==" \
    > /tmp/image-versions.txt
