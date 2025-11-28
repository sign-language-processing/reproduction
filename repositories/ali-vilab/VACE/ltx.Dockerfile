# VACE with LTX-Video support
# This extends the base VACE image with LTX-Video model support

FROM ali-vilab-vace:latest

# Install LTX-Video support (as per VACE warning message)
RUN pip install ltx-video@git+https://github.com/Lightricks/LTX-Video@ltx-video-0.9.1 sentencepiece --no-deps

# Default command: show help for LTX inference script
CMD ["python", "vace/vace_ltx_inference.py", "--help"]
