# GroundingDINO - Open-set Object Detection
FROM nvcr.io/nvidia/pytorch:25.11-py3

# Copy installation script
COPY repositories/IDEA-Research/GroundingDINO/install_from_source.sh /tmp/install_groundingdino.sh
RUN bash /tmp/install_groundingdino.sh /workspace

WORKDIR /workspace/GroundingDINO

CMD ["bash"]
