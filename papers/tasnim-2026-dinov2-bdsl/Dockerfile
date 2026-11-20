FROM ghcr.io/sign-language-processing/reproduction@sha256:679f06c1abe55eda74213fe883e24db8802e8aeeac1b6d94619055b08029d2ea
RUN git clone https://github.com/facebookresearch/dinov2.git /opt/dinov2 && \
    git -C /opt/dinov2 checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8
ENV XFORMERS_DISABLED=1
