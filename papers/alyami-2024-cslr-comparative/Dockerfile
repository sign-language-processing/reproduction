FROM ghcr.io/sign-language-processing/reproduction:latest

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/cache/huggingface \
    HF_HUB_CACHE=/cache/huggingface/hub \
    TORCH_HOME=/cache/huggingface/torch \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# Upstream main.py resumes from its own checkpoints (which carry numpy RNG
# state) with a bare torch.load; PyTorch >= 2.6 defaults to weights_only=True
# and refuses them. TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD restores the pre-2.6
# behaviour without patching upstream code.

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git zlib1g-dev libbz2-dev liblzma-dev p7zip-full unzip \
    && rm -rf /var/lib/apt/lists/*

# sclite, the scorer every baseline.yaml selects (evaluate_tool: sclite, or
# python_evaluate: False in SEN).
ARG SCTK_COMMIT=9688a26882a688132a5e414cadcb4c19b6fffaba
RUN git clone https://github.com/usnistgov/SCTK.git /opt/SCTK \
    && cd /opt/SCTK && git checkout ${SCTK_COMMIT} \
    && make config && make all && make install && test -x bin/sclite

# ctcdecode, the beam-search decoder all four repos import in utils/decode.py.
# Unpatched it does not compile against current PyTorch (see the patch header).
ARG CTCDECODE_COMMIT=c90ad94a0b19554f80804fb7812f2a1447a34a70
COPY ctcdecode-cxx17.patch /tmp/ctcdecode-cxx17.patch
RUN git clone https://github.com/parlance/ctcdecode.git /opt/ctcdecode \
    && cd /opt/ctcdecode && git checkout ${CTCDECODE_COMMIT} \
    && git submodule update --init --recursive \
    && patch -p1 < /tmp/ctcdecode-cxx17.patch \
    && pip install --no-cache-dir --no-build-isolation . \
    && cd / && python -c "import ctcdecode"

# The repos' pins (opencv 4.5.5, scipy 1.7, numpy 1.20, pandas 1.3) have no
# Python 3.12 wheels. These are the oldest releases that do and that still
# ship scipy.misc, which every utils/video_augmentation.py imports.
RUN pip install --no-cache-dir \
    opencv-python-headless==4.10.0.84 \
    scipy==1.13.1 \
    pandas==2.2.3 \
    matplotlib==3.9.2 \
    six==1.16.0 \
    gdown==5.2.0

# ArabSign is the only video source (Phoenix and GSL ship frames). The root
# Dockerfile pins simple-video-utils 0.7.4, but the published
# ghcr.io/.../reproduction:latest image does not contain it (build
# im-L4hfuloTR54towJj5WD1nD: ModuleNotFoundError), so the same pin is installed here.
# It requires av>=18.0.0, which resolved to av 19.0.2; av 19 rejects the
# metadata_errors argument simple-video-utils passes to av.open (run
# ap-OMIRjJ8IzhzSco6yzVHx7u: "open() got an unexpected keyword argument
# 'metadata_errors'"), so av is pinned to the newest 18.x release.
RUN pip install --no-cache-dir simple-video-utils==0.7.4 av==18.1.0

# The four published implementations, pristine. Each run works on its own copy.
ARG VAC_COMMIT=71f3e0334fbc8cecc7ce9816ec69781068abaac0
ARG TLP_COMMIT=21be215c6ba749d98ce6a51f4b84cc9d5029bfe6
ARG SEN_COMMIT=c9a4c7ebd0489e22496c00f9f31d930c0d968ad8
ARG CORRNET_COMMIT=a812814f4deefda0b75ab1dcc90b106c62912263
RUN git clone https://github.com/VIPL-SLP/VAC_CSLR.git /opt/repos/vac \
    && git -C /opt/repos/vac checkout ${VAC_COMMIT} \
    && git clone https://github.com/hulianyuyy/Temporal-Lift-Pooling.git /opt/repos/tlp \
    && git -C /opt/repos/tlp checkout ${TLP_COMMIT} \
    && git clone https://github.com/hulianyuyy/SEN_CSLR.git /opt/repos/sen \
    && git -C /opt/repos/sen checkout ${SEN_COMMIT} \
    && git clone https://github.com/hulianyuyy/CorrNet.git /opt/repos/corrnet \
    && git -C /opt/repos/corrnet checkout ${CORRNET_COMMIT} \
    && python -c "import scipy.misc, cv2, pandas, ctcdecode, simple_video_utils"

# ./software/sclite is the path every evaluation/slr_eval/wer_calculation.py hardcodes.
RUN for r in vac tlp sen corrnet; do \
      mkdir -p /opt/repos/$r/software && ln -s /opt/SCTK/bin/sclite /opt/repos/$r/software/sclite; \
    done

WORKDIR /workspace
