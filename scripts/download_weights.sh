#!/bin/bash
# Download HyperSIGMA pretrained weights from HuggingFace

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( dirname "$SCRIPT_DIR" )"
PRETRAINED_DIR="$REPO_DIR/pretrained"

mkdir -p "$PRETRAINED_DIR"

echo "Downloading HyperSIGMA pretrained weights..."

# Check if huggingface-cli is available
if command -v huggingface-cli &> /dev/null; then
    echo "Using huggingface-cli..."
    huggingface-cli download WHU-Sigma/HyperSIGMA \
        spat-vit-base-ultra-checkpoint-1599.pth \
        spec-vit-base-ultra-checkpoint-1599.pth \
        --local-dir "$PRETRAINED_DIR" \
        --local-dir-use-symlinks False
else
    echo "huggingface-cli not found, using wget..."

    BASE_URL="https://huggingface.co/WHU-Sigma/HyperSIGMA/resolve/main"

    if [ ! -f "$PRETRAINED_DIR/spat-vit-base-ultra-checkpoint-1599.pth" ]; then
        echo "Downloading spatial encoder weights..."
        wget -O "$PRETRAINED_DIR/spat-vit-base-ultra-checkpoint-1599.pth" \
            "$BASE_URL/spat-vit-base-ultra-checkpoint-1599.pth"
    else
        echo "Spatial encoder weights already exist."
    fi

    if [ ! -f "$PRETRAINED_DIR/spec-vit-base-ultra-checkpoint-1599.pth" ]; then
        echo "Downloading spectral encoder weights..."
        wget -O "$PRETRAINED_DIR/spec-vit-base-ultra-checkpoint-1599.pth" \
            "$BASE_URL/spec-vit-base-ultra-checkpoint-1599.pth"
    else
        echo "Spectral encoder weights already exist."
    fi
fi

echo ""
echo "Weights downloaded to: $PRETRAINED_DIR"
ls -la "$PRETRAINED_DIR"
