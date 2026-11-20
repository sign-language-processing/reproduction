FROM ghcr.io/sign-language-processing/reproduction@sha256:305b6165d306192996358ca312d9a751fa409f43063a76dc7758880a8f905291

# Clone the Wan2.2 repository
RUN rm -rf /workspace/* && \
    git clone https://github.com/Wan-Video/Wan2.2.git /workspace

# Install requirements, skipping packages already in base image
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -v -E "^($INSTALLED_STABLE_PACKAGES|diffusers)" requirements.txt > requirements_temp.txt && \
    pip install -r requirements_temp.txt --upgrade-strategy only-if-needed

# Install compatible diffusers, peft, and huggingface_hub CLI
RUN pip install --no-cache-dir diffusers==0.31.0 peft==0.14.0 huggingface_hub

CMD ["/bin/bash"]
