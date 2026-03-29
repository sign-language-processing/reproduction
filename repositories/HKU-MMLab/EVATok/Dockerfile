FROM ghcr.io/sign-language-processing/reproduction:latest

RUN rm -rf /workspace/* && git clone https://github.com/HKU-MMLab/EVATok.git /workspace

WORKDIR /workspace

# Install requirements, skipping packages already in base image
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -v -E "^($INSTALLED_STABLE_PACKAGES)" requirements.txt > /tmp/req.txt && \
    pip install -r /tmp/req.txt --upgrade-strategy only-if-needed && \
    rm /tmp/req.txt

# Patch torch.load calls: PyTorch >=2.6 changed default weights_only to True
# but EVATok checkpoints use numpy scalars which require weights_only=False
COPY repositories/HKU-MMLab/EVATok/patch_torch_load.py /tmp/patch_torch_load.py
RUN python3 /tmp/patch_torch_load.py

CMD ["/bin/bash"]
