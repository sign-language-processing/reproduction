# SignXmDA is a fork of neccam/slt and needs its legacy torchtext 0.5 / torch 1.4
# stack, so this follows papers/camgoz-2020-slt rather than the shared NGC image.
# torch 1.4 (CUDA 10.1) has no kernels for sm_86+, so jobs run on T4.
FROM nvidia/cuda:11.4.3-devel-ubuntu20.04@sha256:5d81539e4f3fab923fac7599baaf44ac055a0e69a47d49aed7c9da1e499c9cba

ARG DEBIAN_FRONTEND=noninteractive
ARG SIGNXMDA_REVISION=b6e2ca4ea81cab41cf858e1d0ce948a778cb09a5
ARG SLT_REVISION=90588825f6229474bc19ac7a6b30ea3116635ba3
ARG PGEN_REVISION=2b8a2329d196aead731e4dc7a1449cd57247708c
ARG MINICONDA_SHA256=4dc4214839c60b2f5eb3efbdee1ef5d9b45e74f2c09fcae6c8934a13f36ffc3e

RUN apt-get update \
    && apt-get install -y --no-install-recommends git wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh \
    && echo "${MINICONDA_SHA256}  Miniconda3-py37_4.12.0-Linux-x86_64.sh" | sha256sum -c - \
    && bash Miniconda3-py37_4.12.0-Linux-x86_64.sh -b \
    && rm Miniconda3-py37_4.12.0-Linux-x86_64.sh

RUN git clone https://github.com/Atrewin/SignXmDA.git /SignXmDA \
    && git -C /SignXmDA checkout "${SIGNXMDA_REVISION}"

# PGen's released gloss-text pseudo-pairs, used only by the conditional teachers.
RUN git clone https://github.com/Atrewin/PGen.git /PGen \
    && git -C /PGen checkout "${PGEN_REVISION}"

# SignXmDA ships no requirements file; its README points to neccam/slt's.
RUN wget -q -O /tmp/requirements.txt \
        "https://raw.githubusercontent.com/neccam/slt/${SLT_REVISION}/requirements.txt" \
    && sed -i '/^conda==4.8.0$/d' /tmp/requirements.txt \
    && sed -i '/^conda-package-handling==1.6.0$/d' /tmp/requirements.txt \
    && sed -i 's/^tensorboard==2.1.2$/tensorboard==2.1.1/' /tmp/requirements.txt \
    && sed -i 's/^tensorflow-estimator==2.1.2$/tensorflow-estimator==2.1.0/' /tmp/requirements.txt \
    && sed -i 's/^warmup-scheduler==0.1.1$/warmup-scheduler==0.3/' /tmp/requirements.txt \
    && /root/miniconda3/bin/pip install --no-cache-dir \
        -r /tmp/requirements.txt typing-extensions==4.7.1 \
    && /root/miniconda3/bin/pip install --no-cache-dir transformers==3.0.2 \
    && /root/miniconda3/bin/pip freeze > /SignXmDA.freeze.txt

# Gloss-model validation/decoding fixes for the conditional teachers; see each header.
COPY patches/ /patches/
RUN git -C /SignXmDA apply /patches/0001-gloss-decoding-mirror-training.patch \
    && git -C /SignXmDA apply /patches/0002-test-pass-forward-type.patch

WORKDIR /SignXmDA

ENV HF_HOME=/cache/huggingface \
    HF_HUB_CACHE=/cache/huggingface/hub
