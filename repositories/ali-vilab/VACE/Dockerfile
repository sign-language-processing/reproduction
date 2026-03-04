FROM ghcr.io/sign-language-processing/reproduction:latest

WORKDIR /workspace

# Clone the VACE repository
RUN git clone https://github.com/ali-vilab/VACE.git

WORKDIR /workspace/VACE

# Install requirements, skipping packages already in base image or unavailable
# - torch, torchvision: use base image versions (PyTorch 2.10)
# - flash_attn: pre-installed in base image
# - decord: installed from source above
# - onnxruntime-gpu: no arm64 wheels, use onnxruntime instead
RUN grep -v -E "^(torch|torchvision|flash_attn|decord|onnxruntime-gpu)" requirements/framework.txt > requirements_filtered.txt && \
    pip install --no-cache-dir -r requirements_filtered.txt onnxruntime

# Install Wan2.1 support
RUN pip install wan@git+https://github.com/Wan-Video/Wan2.1

# Install LTX-Video support (as per VACE warning message)
RUN pip install ltx-video@git+https://github.com/Lightricks/LTX-Video@ltx-video-0.9.1 sentencepiece --no-deps

# Install preprocessing tools, skipping problematic dependencies
# - GroundingDino: does not play well with torch 2.10, requires install from source
RUN grep -v -E "GroundingDINO" requirements/annotator.txt > requirements_annotator_filtered.txt && \
    pip install --no-cache-dir -r requirements_annotator_filtered.txt

# Install GroundingDINO from source
COPY repositories/IDEA-Research/GroundingDINO/install_from_source.sh /tmp/install_groundingdino.sh
RUN bash /tmp/install_groundingdino.sh /tmp


CMD ["python", "vace/vace_pipeline.py", "--help"]
