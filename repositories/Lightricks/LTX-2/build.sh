#!/bin/bash
# Build the LTX-2 video tokenizer Docker image
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building LTX-2 Video Tokenizer Docker image..."

# Build using docker-compose
docker compose build

echo ""
echo "Build complete! Image: ltx2-video-tokenizer:latest"
echo ""
echo "Usage:"
echo "  ./run.sh encode <input.mp4> <output.pt>    # Encode video to latents"
echo "  ./run.sh decode <input.pt> <output.mp4>    # Decode latents to video"
echo "  ./run.sh reconstruct <input.mp4> <output.mp4>  # Encode + decode"
echo "  ./run.sh shell                             # Interactive shell"
