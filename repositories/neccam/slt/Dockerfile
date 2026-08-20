FROM nvidia/cuda:11.4.3-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG SLT_REVISION=90588825f6229474bc19ac7a6b30ea3116635ba3

RUN apt-get update \
    && apt-get install -y --no-install-recommends git wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py37_4.12.0-Linux-x86_64.sh -b \
    && rm Miniconda3-py37_4.12.0-Linux-x86_64.sh

RUN git clone https://github.com/neccam/slt.git /slt \
    && git -C /slt checkout "${SLT_REVISION}"

WORKDIR /slt

# These exact pins are unavailable for Python 3.7. The nearest patch releases
# preserve the published framework/API versions and match the recipe that was
# previously proven in this reproduction repository.
RUN sed -i '/^conda==4.8.0$/d' requirements.txt \
    && sed -i '/^conda-package-handling==1.6.0$/d' requirements.txt \
    && sed -i 's/^tensorboard==2.1.2$/tensorboard==2.1.1/' requirements.txt \
    && sed -i 's/^tensorflow-estimator==2.1.2$/tensorflow-estimator==2.1.0/' requirements.txt \
    && sed -i 's/^warmup-scheduler==0.1.1$/warmup-scheduler==0.3/' requirements.txt \
    && sed -i 's/^torch==1.4.0$/torch==1.7.1+cu110/' requirements.txt \
    && sed -i 's/^torchtext==0.5.0$/torchtext==0.8.1/' requirements.txt \
    && sed -i 's/^torchvision==0.5.0$/torchvision==0.8.2+cu110/' requirements.txt \
    && /root/miniconda3/bin/pip install --no-cache-dir \
        -f https://download.pytorch.org/whl/torch_stable.html \
        -r requirements.txt \
        typing-extensions==4.7.1

ENV HF_HOME=/cache/huggingface \
    HF_HUB_CACHE=/cache/huggingface/hub
