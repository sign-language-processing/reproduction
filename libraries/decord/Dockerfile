FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# First install ffmpeg 4 from source
COPY ffmpeg/install_from_source.sh /tmp/install_ffmpeg.sh
RUN bash /tmp/install_ffmpeg.sh

# Then install decord
COPY decord/install_from_source.sh /tmp/install_decord.sh
RUN bash /tmp/install_decord.sh

# Verify both work
CMD python3 -c "import decord; print('Decord version:', decord.__version__); print('Decord successfully imported!')"
