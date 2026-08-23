FROM nvcr.io/nvidia/tensorflow:20.12-tf1-py3@sha256:312147204a65c65ba25a75944333105493772121c11c33e769919ea10e9a256f

ARG DEBIAN_FRONTEND=noninteractive
ARG NSLT_REVISION=06951580b58f04b9cd64efcf61aeca36011031d3
ARG ALEXNET_SHA256=1294ef51496c52f2e4879ee3b5e1da22c35d61f3240937491b4469403d5428cc
ARG VIDEO_UTILS_SHA256=55560a77c47b1326821335244b710e671cfd43d2fc3ef181cc14ef8e4cc4f01d

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates git wget pkg-config \
      libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev \
      libavutil-dev libswresample-dev libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/neccam/nslt.git /nslt \
    && git -C /nslt checkout "${NSLT_REVISION}"

RUN pip install --no-cache-dir \
      opencv-python-headless==4.5.1.48 \
      Cython==0.29.37 \
    && pip install --no-cache-dir --no-build-isolation av==8.0.3 \
    && wget -q -O /tmp/simple_video_utils-0.0.4-py3-none-any.whl \
      https://files.pythonhosted.org/packages/4e/27/0adcd6dac521fbce6ca63294d4e0403c49e403e494df6036e84ab85a137e/simple_video_utils-0.0.4-py3-none-any.whl \
    && echo "${VIDEO_UTILS_SHA256}  /tmp/simple_video_utils-0.0.4-py3-none-any.whl" | sha256sum -c - \
    && pip install --no-cache-dir --no-deps \
      /tmp/simple_video_utils-0.0.4-py3-none-any.whl \
    && rm /tmp/simple_video_utils-0.0.4-py3-none-any.whl \
    && wget -q -O /nslt/BaseModel/bvlc_alexnet.npy \
      https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy \
    && echo "${ALEXNET_SHA256}  /nslt/BaseModel/bvlc_alexnet.npy" | sha256sum -c -

COPY patches /tmp/patches
RUN for patch in /tmp/patches/*.patch; do git -C /nslt apply "$patch"; done \
    && rm -rf /tmp/patches

ENV HF_HOME=/cache/huggingface \
    HF_HUB_CACHE=/cache/huggingface/hub

WORKDIR /nslt/nslt
