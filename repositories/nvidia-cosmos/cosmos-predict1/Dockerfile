FROM ghcr.io/sign-language-processing/reproduction:latest

# Clone the cosmos-predict1 repository
RUN rm -rf /workspace/* && \
    git clone https://github.com/nvidia-cosmos/cosmos-predict1.git /workspace

# Install requirements, skipping packages already in base image
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -v -E "^($INSTALLED_STABLE_PACKAGES|pyyaml|PyYAML)" requirements.txt > requirements_temp.txt  && \
    pip install -r requirements_temp.txt --upgrade-strategy only-if-needed

# Install additional dependencies
RUN pip install huggingface-hub

# Clean up files that we will mount
RUN rm -rf /workspace/checkpoints

# Copy other training setup files
COPY training /tmp/training
RUN cp -a /tmp/training/. /workspace/cosmos_predict1/tokenizer/training/
# Apply patches for compatibility and performance
RUN bash /tmp/training/patch.sh

CMD ["/bin/bash"]
