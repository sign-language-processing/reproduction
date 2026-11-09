# AlphaPose for the one-off Halpe-136 keypoint extraction (Sec. 3.2). Its CUDA
# extensions target PyTorch 1.11+ (docs/INSTALL.md), so it gets its own image
# instead of the NGC PyTorch 2.x training image. A full CUDA toolkit base is
# needed: pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel ships a partial conda
# nvcc without cusolverDn.h, and AlphaPose's nms_cuda build fails there.
# Ubuntu 22.04 for Python 3.10, the oldest Python Modal's image builder runs.
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.0;8.6" \
    FORCE_CUDA=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential python3.10 python3.10-dev python3-pip libyaml-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

RUN python -m pip install --no-cache-dir --upgrade "pip==23.3.2" \
    && python -m pip install --no-cache-dir torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
       --extra-index-url https://download.pytorch.org/whl/cu117 \
    && python -m pip install --no-cache-dir "numpy==1.23.5" "cython==0.29.36" "setuptools==65.5.0" gdown==5.2.0

# AlphaPose's setup.py install_requires, pinned and installed with pip first:
# left to setup.py, easy_install picks munkres 3.0.0, whose sdist has no setup
# script, and the build fails.
RUN python -m pip install --no-cache-dir --no-build-isolation \
    six==1.16.0 terminaltables==3.1.10 scipy==1.10.1 opencv-python==4.8.1.78 \
    matplotlib==3.7.5 visdom==0.2.4 tqdm==4.66.5 tensorboardx==2.6.2.2 easydict==1.13 \
    pyyaml==6.0.2 halpecocotools==0.0.0 munkres==1.1.4 timm==0.1.20 natsort==8.4.0 \
    pycocotools==2.0.7

# AlphaPose (2023) uses np.float and friends, removed in numpy 1.24 (pinned above).
ARG ALPHAPOSE_COMMIT=c60106d19afb443e964df6f06ed1842962f5f1f7
RUN git clone https://github.com/MVIG-SJTU/AlphaPose.git /opt/AlphaPose \
    && cd /opt/AlphaPose && git checkout ${ALPHAPOSE_COMMIT} \
    && python setup.py build develop \
    && python -c "import alphapose, detector"

# alphapose/datasets/coco_wholebody.py imports tkinter at import time.
RUN apt-get update && apt-get install -y --no-install-recommends python3-tk \
    && rm -rf /var/lib/apt/lists/* && python -c "import tkinter"

WORKDIR /opt/AlphaPose
