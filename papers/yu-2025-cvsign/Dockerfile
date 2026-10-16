FROM ghcr.io/sign-language-processing/reproduction:latest

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/cache/huggingface \
    HF_HUB_CACHE=/cache/huggingface/hub \
    TORCH_HOME=/cache/huggingface/torch \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# Upstream main.py loads its own checkpoints (which carry numpy RNG state) with
# a bare torch.load; PyTorch >= 2.6 defaults to weights_only=True and refuses
# them. TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD restores the pre-2.6 behaviour
# without patching upstream code.

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git zlib1g-dev libbz2-dev liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# sclite, the scorer CorrNet's configs/baseline.yaml selects (evaluate_tool: sclite).
ARG SCTK_COMMIT=9688a26882a688132a5e414cadcb4c19b6fffaba
RUN git clone https://github.com/usnistgov/SCTK.git /opt/SCTK \
    && cd /opt/SCTK && git checkout ${SCTK_COMMIT} \
    && make config && make all && make install && test -x bin/sclite

# ctcdecode, the beam-search decoder CorrNet's utils/decode.py imports.
ARG CTCDECODE_COMMIT=c90ad94a0b19554f80804fb7812f2a1447a34a70
COPY patches/ctcdecode-cxx17.patch /tmp/ctcdecode-cxx17.patch
RUN git clone https://github.com/parlance/ctcdecode.git /opt/ctcdecode \
    && cd /opt/ctcdecode && git checkout ${CTCDECODE_COMMIT} \
    && git submodule update --init --recursive \
    && patch -p1 < /tmp/ctcdecode-cxx17.patch \
    && pip install --no-cache-dir --no-build-isolation . \
    && cd / && python -c "import ctcdecode"

# CorrNet's requirements.txt pins (opencv 4.5.5, scipy 1.7, numpy 1.20) have no
# Python 3.12 wheels. These are the oldest releases that do and that still
# ship scipy.misc, which CorrNet's utils/video_augmentation.py imports.
RUN pip install --no-cache-dir \
    opencv-python-headless==4.10.0.84 \
    scipy==1.13.1 \
    gdown==5.2.0

# Pristine CorrNet: evaluates the authors' released checkpoint (harness check).
ARG CORRNET_COMMIT=5a361cc709ae8f697f4c9e23760e64c2daa8f744
RUN git clone https://github.com/hulianyuyy/CorrNet.git /opt/CorrNet \
    && git -C /opt/CorrNet checkout ${CORRNET_COMMIT}

# CVSign = CorrNet harness + TLP's temporal decoder + the paper's CCA/CVA.
# The CTC compatibility patch goes first; the pristine CorrNet above only
# evaluates, which never runs CTC backward.
ARG TLP_COMMIT=21be215c6ba749d98ce6a51f4b84cc9d5029bfe6
COPY patches/corrnet-ctc-without-cudnn.patch patches/cvsign-*.patch /tmp/patches/
RUN git clone https://github.com/hulianyuyy/Temporal-Lift-Pooling.git /opt/TLP \
    && git -C /opt/TLP checkout ${TLP_COMMIT} \
    && git clone /opt/CorrNet /workspace/CVSign \
    && cd /workspace/CVSign && git checkout ${CORRNET_COMMIT} \
    && cp /opt/TLP/modules/tconv.py modules/tconv.py \
    && git apply /tmp/patches/corrnet-ctc-without-cudnn.patch \
    && for p in /tmp/patches/cvsign-*.patch; do git apply "$p"; done

# Runtime paths the upstream configs hardcode: ./software/sclite and
# ./dataset/phoenix2014-T (populated at container start under /data).
RUN for d in /opt/CorrNet /workspace/CVSign; do \
      mkdir -p $d/software $d/dataset \
      && ln -s /opt/SCTK/bin/sclite $d/software/sclite \
      && ln -s /data/phoenix2014-T $d/dataset/phoenix2014-T; \
    done \
    && python -c "import scipy.misc, cv2, ctcdecode"

WORKDIR /workspace/CVSign
