FROM ghcr.io/sign-language-processing/reproduction:latest

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/cache/huggingface \
    HF_HUB_CACHE=/cache/huggingface/hub \
    TORCH_HOME=/cache/huggingface/torch \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# main.py reloads its own checkpoints (which carry numpy RNG state) with a bare
# torch.load; PyTorch >= 2.6 defaults to weights_only=True and refuses them.

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git zlib1g-dev libbz2-dev liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# sclite, the scorer evaluation/slr_eval/wer_calculation.py calls (python_evaluate: False).
ARG SCTK_COMMIT=9688a26882a688132a5e414cadcb4c19b6fffaba
RUN git clone https://github.com/usnistgov/SCTK.git /opt/SCTK \
    && cd /opt/SCTK && git checkout ${SCTK_COMMIT} \
    && make config && make all && make install && test -x bin/sclite

# ctcdecode, the beam-search decoder utils/decode.py imports.
ARG CTCDECODE_COMMIT=c90ad94a0b19554f80804fb7812f2a1447a34a70
COPY patches/ctcdecode-cxx17.patch /tmp/ctcdecode-cxx17.patch
RUN git clone https://github.com/parlance/ctcdecode.git /opt/ctcdecode \
    && cd /opt/ctcdecode && git checkout ${CTCDECODE_COMMIT} \
    && git submodule update --init --recursive \
    && patch -p1 < /tmp/ctcdecode-cxx17.patch \
    && pip install --no-cache-dir --no-build-isolation . \
    && cd / && python -c "import ctcdecode"

# The repo has no requirements file. These are its remaining imports; opencv and
# scipy are the oldest Python 3.12 releases (scipy.misc is still imported).
RUN pip install --no-cache-dir \
    opencv-python-headless==4.10.0.84 \
    scipy==1.13.1 \
    pandas==2.2.3 \
    lmdb==1.5.1 \
    six==1.16.0 \
    matplotlib==3.9.2 \
    tqdm==4.66.5

ARG SC2SLR_COMMIT=65c95fb03fa0c170dc3f1136265f4b97ec26d7fa
COPY patches/sc2slr-*.patch /tmp/patches/
RUN git clone https://github.com/christianlsl/Skeleton-based-Contrast-for-Sign-Language-Recognition.git /workspace/SC2SLR \
    && cd /workspace/SC2SLR && git checkout ${SC2SLR_COMMIT} \
    && for p in /tmp/patches/sc2slr-*.patch; do git apply "$p"; done

# Runtime paths the upstream configs hardcode: ./software/sclite and
# ./dataset/phoenix2014{,-T} (the committed phoenix2014 symlink points at the
# author's disk); frames are staged under /data at container start.
RUN cd /workspace/SC2SLR \
    && mkdir -p software && ln -s /opt/SCTK/bin/sclite software/sclite \
    && rm -f dataset/phoenix2014 \
    && ln -s /data/phoenix2014 dataset/phoenix2014 \
    && ln -s /data/phoenix2014-T dataset/phoenix2014-T \
    && python -c "import scipy.misc, cv2, ctcdecode, lmdb, pandas"

WORKDIR /workspace/SC2SLR
