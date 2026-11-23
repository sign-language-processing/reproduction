# The paper's stack is Keras/TensorFlow (DenseNet/ResNet/EfficientNet applications,
# ImageDataGenerator-style augmentation), not the root study image's PyTorch/NGC
# stack. No video decoding is involved (static 64x64 images), so simple-video-utils/
# decord/ffmpeg are not needed here.
FROM tensorflow/tensorflow:2.17.0-gpu@sha256:ebf7ad13136740adeee241fcfe6b14d646e431c6bc5c151d528e31ddafde4623

ENV DEBIAN_FRONTEND=noninteractive

RUN pip install --no-cache-dir \
    pandas==2.2.3 \
    pyarrow==17.0.0 \
    scikit-learn==1.5.2 \
    pillow==10.4.0

WORKDIR /workspace

ENV INSTALLED_STABLE_PACKAGES="tensorflow|pandas|pyarrow|scikit-learn|pillow"

RUN pip list --format=freeze \
    | grep -iE "^($INSTALLED_STABLE_PACKAGES)==" \
    > /tmp/image-versions.txt
