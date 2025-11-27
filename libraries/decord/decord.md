# Decord

Installing `decord` requires a very specific environment.

First, you must have `ffmpeg` lower than version 5 (see libraries/ffmpeg/ffmpeg.md).

Then, you must install `decord` from source.

```shell
# Install ffmpeg 4 from source (required for decord)
COPY libraries/ffmpeg/install_from_source.sh /tmp/install_ffmpeg.sh
RUN bash /tmp/install_ffmpeg.sh

# Install decord from source
COPY libraries/decord/install_from_source.sh /tmp/install_decord.sh
RUN bash /tmp/install_decord.sh
```
