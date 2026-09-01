# Separate environment for the Transformer-tiny/+Back-translation baseline
# (ref [33]'s kayoyin/transformer-slt, vendored OpenNMT-py 1.0.0 pinned to
# torchtext==0.4.0). Incompatible with the modern-Transformers image used
# for the mBART/mT5 systems, so this paper keeps a second, older-stack
# Dockerfile rather than forcing one environment to serve both. Pattern
# follows the sibling camgoz-2020-slt reproduction in this repo (modern CUDA
# base + conda py37, letting pip resolve a CUDA-11-compatible old torch).
FROM nvidia/cuda:11.4.3-devel-ubuntu20.04@sha256:5d81539e4f3fab923fac7599baaf44ac055a0e69a47d49aed7c9da1e499c9cba

ARG DEBIAN_FRONTEND=noninteractive
ARG TSLT_REVISION=d119fbb642d653a987a2e1b2cd1541c88df7f2ef
ARG MINICONDA_SHA256=4dc4214839c60b2f5eb3efbdee1ef5d9b45e74f2c09fcae6c8934a13f36ffc3e

RUN apt-get update \
    && apt-get install -y --no-install-recommends git wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh \
    && echo "${MINICONDA_SHA256}  Miniconda3-py37_4.12.0-Linux-x86_64.sh" | sha256sum -c - \
    && bash Miniconda3-py37_4.12.0-Linux-x86_64.sh -b \
    && rm Miniconda3-py37_4.12.0-Linux-x86_64.sh

COPY patches/ /tmp/patches/
RUN git clone https://github.com/kayoyin/transformer-slt.git /opt/transformer-slt \
    && git -C /opt/transformer-slt checkout "${TSLT_REVISION}" \
    && git -C /opt/transformer-slt -c user.email=repro@example.invalid -c user.name=repro am /tmp/patches/*.patch \
    # tools/rouge.py resolves its wordnet DB files relative to its own
    # directory (pkg_resources), but the repo ships them at the repo root;
    # same pre-existing repo-layout quirk hit by the mBART/mT5 image.
    && cp /opt/transformer-slt/wordnet_key_value.txt /opt/transformer-slt/wordnet_key_value_special_cases.txt /opt/transformer-slt/tools/

WORKDIR /opt/transformer-slt

RUN /root/miniconda3/bin/pip install --no-cache-dir -r requirements.txt \
    && /root/miniconda3/bin/pip install --no-cache-dir -e . \
    && /root/miniconda3/bin/python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet')"

ENV HF_HOME=/cache/huggingface \
    HF_HUB_CACHE=/cache/huggingface/hub
