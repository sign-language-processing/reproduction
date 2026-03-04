FROM ghcr.io/sign-language-processing/reproduction:latest

ENV PYTHONUNBUFFERED=1

# Set working directory for installation
WORKDIR /workspace/ml-atoken-train

# Copy only pyproject.toml first for better Docker layer caching
COPY repositories/apple/ml-atoken-train/pyproject.toml /workspace/ml-atoken-train/

# Install Python dependencies (this layer will be cached unless pyproject.toml changes)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Now copy the rest of the code (changes here won't invalidate dependency installation)
COPY repositories/apple/ml-atoken-train /workspace/ml-atoken-train

# Create directories for data and checkpoints
RUN mkdir -p /workspace/data /workspace/checkpoints /workspace/output

# Set default command
CMD ["/bin/bash"]
