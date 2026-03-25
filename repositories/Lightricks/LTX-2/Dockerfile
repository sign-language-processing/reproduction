FROM ghcr.io/sign-language-processing/reproduction:latest

# Clone the LTX-2 repository
RUN rm -rf /workspace/* && \
    git clone https://github.com/Lightricks/LTX-2.git /workspace/LTX-2

# Install ltx-core from the repo, skipping packages already in base image
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install /workspace/LTX-2/packages/ltx-core --upgrade-strategy only-if-needed

# Install additional dependencies for our VAE round-trip script
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install imageio[pyav] Pillow huggingface-hub --upgrade-strategy only-if-needed

# Copy our scripts and source modules
COPY src /workspace/src
COPY scripts /workspace/scripts

WORKDIR /workspace

CMD ["/bin/bash"]
