FROM ghcr.io/sign-language-processing/reproduction:latest

# Clone the VACE repository
RUN git clone https://github.com/ali-vilab/VACE.git

WORKDIR /workspace/VACE

# Install requirements, skipping packages already in base image
# - onnxruntime-gpu: no arm64 wheels, use onnxruntime instead
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -v -E "^($INSTALLED_STABLE_PACKAGES|onnxruntime-gpu)" requirements/framework.txt > requirements_filtered.txt && \
    pip install -r requirements_filtered.txt onnxruntime --upgrade-strategy only-if-needed

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
