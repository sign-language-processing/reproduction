FROM ghcr.io/sign-language-processing/reproduction:latest

# Clone the cosmos-predict1 repository
RUN rm -rf /workspace/* && \
    git clone https://github.com/nvidia-cosmos/cosmos-predict1.git /workspace

WORKDIR /workspace

# Install requirements, skipping packages already in base image
# Base image includes: torch, torchvision, torchaudio, flash-attn, apex
RUN grep -v -E "^(torch|torchvision|torchaudio|apex|decord)" requirements.txt > requirements_temp.txt && \
    pip install --no-cache-dir --ignore-installed -r requirements_temp.txt && \
    rm requirements_temp.txt

# Copy other training setup files
COPY repositories/nvidia-cosmos/cosmos-predict1/training /tmp/training
RUN cp -a /tmp/training/. /workspace/cosmos_predict1/tokenizer/training/
# Apply patches for compatibility and performance
RUN bash /tmp/training/patch.sh

CMD ["/bin/bash"]
