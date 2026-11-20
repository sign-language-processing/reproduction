FROM ghcr.io/sign-language-processing/reproduction@sha256:679f06c1abe55eda74213fe883e24db8802e8aeeac1b6d94619055b08029d2ea
# Keep the study GPU image and isolate TensorFlow's CUDA dependencies from torch.
RUN python -m venv /opt/taqiyya && /opt/taqiyya/bin/pip install --no-cache-dir \
    'tensorflow[and-cuda]==2.18.0' keras==3.8.0 numpy==1.26.4 \
    scikit-learn==1.5.2 scipy==1.14.1 pillow==10.4.0
