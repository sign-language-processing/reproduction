FROM nvidia/cuda:11.4.3-devel-ubuntu20.04@sha256:5d81539e4f3fab923fac7599baaf44ac055a0e69a47d49aed7c9da1e499c9cba

ARG DEBIAN_FRONTEND=noninteractive
ARG SLT_REVISION=90588825f6229474bc19ac7a6b30ea3116635ba3
ARG MINICONDA_SHA256=4dc4214839c60b2f5eb3efbdee1ef5d9b45e74f2c09fcae6c8934a13f36ffc3e

RUN apt-get update \
    && apt-get install -y --no-install-recommends git wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh \
    && echo "${MINICONDA_SHA256}  Miniconda3-py37_4.12.0-Linux-x86_64.sh" | sha256sum -c - \
    && bash Miniconda3-py37_4.12.0-Linux-x86_64.sh -b \
    && rm Miniconda3-py37_4.12.0-Linux-x86_64.sh

RUN git clone https://github.com/neccam/slt.git /slt \
    && git -C /slt checkout "${SLT_REVISION}"

WORKDIR /slt

# Three exact requirements disappeared from PyPI; only their patch releases change.
RUN sed -i '/^conda==4.8.0$/d' requirements.txt \
    && sed -i '/^conda-package-handling==1.6.0$/d' requirements.txt \
    && sed -i 's/^tensorboard==2.1.2$/tensorboard==2.1.1/' requirements.txt \
    && sed -i 's/^tensorflow-estimator==2.1.2$/tensorflow-estimator==2.1.0/' requirements.txt \
    && sed -i 's/^warmup-scheduler==0.1.1$/warmup-scheduler==0.3/' requirements.txt \
    && /root/miniconda3/bin/pip install --no-cache-dir \
        -r requirements.txt typing-extensions==4.7.1

ENV HF_HOME=/cache/huggingface \
    HF_HUB_CACHE=/cache/huggingface/hub
