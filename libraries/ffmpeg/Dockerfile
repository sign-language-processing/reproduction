FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# Copy the installation script
COPY install_from_source.sh /tmp/install_from_source.sh

# Run the installation
RUN bash /tmp/install_from_source.sh

# Verify ffmpeg works
CMD ["ffmpeg", "-version"]
