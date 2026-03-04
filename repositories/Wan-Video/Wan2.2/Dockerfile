FROM ghcr.io/sign-language-processing/reproduction:latest

# Clone the Wan2.2 repository
RUN rm -rf /workspace/* && \
    git clone https://github.com/Wan-Video/Wan2.2.git /workspace

WORKDIR /workspace

# Install requirements, skipping packages already in base image
RUN grep -v -E "(torch|torchvision|torchaudio|flash_attn|diffusers)" requirements.txt > requirements_temp.txt && \
    pip install --no-cache-dir -r requirements_temp.txt && \
    rm requirements_temp.txt

# Install compatible diffusers, peft, and huggingface_hub CLI
RUN pip install --no-cache-dir diffusers==0.31.0 peft==0.14.0 huggingface_hub

CMD ["/bin/bash"]
