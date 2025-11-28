# VACE with Wan2.1 support
# This extends the base VACE image with Wan2.1 model support

FROM ali-vilab-vace:latest

# Install Wan2.1 support
RUN pip install wan@git+https://github.com/Wan-Video/Wan2.1

# Default command: show help for Wan inference script
CMD ["python", "vace/vace_wan_inference.py", "--help"]
