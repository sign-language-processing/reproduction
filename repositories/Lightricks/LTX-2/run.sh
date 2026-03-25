#!/bin/bash
# Run the LTX-2 video tokenizer
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
DOCKER_IMAGE="ltx2-video-tokenizer:latest"
CONTAINER_NAME="ltx2-tokenizer"
VIDEO_TOKENIZER_PATH="/mnt/rylo-tnas/users/rotem/sign/video-tokenizer"
CACHE_DIR="/mnt/rylo-tnas/users/rotem/.cache"

# Function to display usage
usage() {
    echo "LTX-2 Video Tokenizer"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  encode <input> <output>      Encode video(s) to latents"
    echo "  decode <input> <output>      Decode latents to video"
    echo "  reconstruct <input> <output> Encode and decode (roundtrip)"
    echo "  shell                        Start interactive shell"
    echo "  help                         Show this help message"
    echo ""
    echo "Options (pass after input/output):"
    echo "  --resolution WxH     Resize input to WxH (e.g., 640x480)"
    echo "  --dtype TYPE         Model dtype: bfloat16, float32, float16"
    echo "  --tiling             Enable tiled VAE for large videos"
    echo "  --start-time SEC     Start time in seconds"
    echo "  --end-time SEC       End time in seconds"
    echo "  --overwrite          Overwrite existing output files"
    echo ""
    echo "Examples:"
    echo "  $0 encode /path/to/video.mp4 /path/to/latents.pt"
    echo "  $0 decode /path/to/latents.pt /path/to/output.mp4"
    echo "  $0 reconstruct /path/to/video.mp4 /path/to/reconstructed.mp4"
    echo "  $0 encode /path/to/videos/ /path/to/latents/ --resolution 512x512"
    echo "  $0 shell"
    exit 1
}

# Function to run docker command
run_docker() {
    docker run --rm -it \
        --gpus all \
        --name "$CONTAINER_NAME-$$" \
        -e HF_HOME=/workspace/.cache/huggingface \
        -e TORCH_HOME=/workspace/.cache/torch \
        -v "$VIDEO_TOKENIZER_PATH":/workspace/video-tokenizer \
        -v "$CACHE_DIR/huggingface":/workspace/.cache/huggingface \
        -v "$CACHE_DIR/torch":/workspace/.cache/torch \
        -v "$INPUT_MOUNT" \
        -v "$OUTPUT_MOUNT" \
        -w /workspace/video-tokenizer \
        "$DOCKER_IMAGE" \
        "$@"
}

# Parse command
if [ $# -lt 1 ]; then
    usage
fi

COMMAND="$1"
shift

case "$COMMAND" in
    encode)
        if [ $# -lt 2 ]; then
            echo "Error: encode requires input and output paths"
            usage
        fi
        INPUT_PATH="$1"
        OUTPUT_PATH="$2"
        shift 2

        # Convert to absolute paths
        INPUT_PATH="$(realpath "$INPUT_PATH")"
        OUTPUT_PATH="$(realpath "$OUTPUT_PATH" 2>/dev/null || echo "$OUTPUT_PATH")"
        OUTPUT_DIR="$(dirname "$OUTPUT_PATH")"
        mkdir -p "$OUTPUT_DIR"
        OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"
        OUTPUT_FILE="$(basename "$OUTPUT_PATH")"

        INPUT_DIR="$(dirname "$INPUT_PATH")"
        INPUT_FILE="$(basename "$INPUT_PATH")"

        INPUT_MOUNT="$INPUT_DIR:/workspace/input:ro"
        OUTPUT_MOUNT="$OUTPUT_DIR:/workspace/output"

        run_docker python -m video_tokenizer.bin \
            --tokenizer ltx2 \
            --encode "/workspace/input/$INPUT_FILE" \
            --output "/workspace/output/$OUTPUT_FILE" \
            "$@"
        ;;

    decode)
        if [ $# -lt 2 ]; then
            echo "Error: decode requires input and output paths"
            usage
        fi
        INPUT_PATH="$1"
        OUTPUT_PATH="$2"
        shift 2

        INPUT_PATH="$(realpath "$INPUT_PATH")"
        OUTPUT_PATH="$(realpath "$OUTPUT_PATH" 2>/dev/null || echo "$OUTPUT_PATH")"
        OUTPUT_DIR="$(dirname "$OUTPUT_PATH")"
        mkdir -p "$OUTPUT_DIR"
        OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"
        OUTPUT_FILE="$(basename "$OUTPUT_PATH")"

        INPUT_DIR="$(dirname "$INPUT_PATH")"
        INPUT_FILE="$(basename "$INPUT_PATH")"

        INPUT_MOUNT="$INPUT_DIR:/workspace/input:ro"
        OUTPUT_MOUNT="$OUTPUT_DIR:/workspace/output"

        run_docker python -m video_tokenizer.bin \
            --tokenizer ltx2 \
            --decode "/workspace/input/$INPUT_FILE" \
            --output "/workspace/output/$OUTPUT_FILE" \
            "$@"
        ;;

    reconstruct)
        if [ $# -lt 2 ]; then
            echo "Error: reconstruct requires input and output paths"
            usage
        fi
        INPUT_PATH="$1"
        OUTPUT_PATH="$2"
        shift 2

        INPUT_PATH="$(realpath "$INPUT_PATH")"
        OUTPUT_PATH="$(realpath "$OUTPUT_PATH" 2>/dev/null || echo "$OUTPUT_PATH")"
        OUTPUT_DIR="$(dirname "$OUTPUT_PATH")"
        mkdir -p "$OUTPUT_DIR"
        OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"
        OUTPUT_FILE="$(basename "$OUTPUT_PATH")"

        INPUT_DIR="$(dirname "$INPUT_PATH")"
        INPUT_FILE="$(basename "$INPUT_PATH")"

        INPUT_MOUNT="$INPUT_DIR:/workspace/input:ro"
        OUTPUT_MOUNT="$OUTPUT_DIR:/workspace/output"

        run_docker python -m video_tokenizer.bin \
            --tokenizer ltx2 \
            --reconstruct "/workspace/input/$INPUT_FILE" \
            --output "/workspace/output/$OUTPUT_FILE" \
            "$@"
        ;;

    shell)
        INPUT_MOUNT="/tmp:/workspace/input"
        OUTPUT_MOUNT="/tmp:/workspace/output"
        run_docker /bin/bash
        ;;

    help|--help|-h)
        usage
        ;;

    *)
        echo "Error: Unknown command '$COMMAND'"
        usage
        ;;
esac
