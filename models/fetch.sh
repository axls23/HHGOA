#!/usr/bin/env bash
# Downloads YuNet + SFace ONNX weights from the OpenCV Zoo (Apache-2.0) into
# the local model cache. Never committed to git — see PRD §5.4 NVMe layout.
set -euo pipefail

CACHE_DIR="${FACEANCHOR_MODEL_CACHE:-$HOME/.cache/faceanchor/models}"
mkdir -p "$CACHE_DIR"

# opencv_zoo tracks .onnx weights via git-lfs; media.githubusercontent.com
# resolves LFS pointers to the actual binary (raw.githubusercontent.com does not).
BASE_URL="https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"

YUNET_PATH="face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_PATH="face_recognition_sface/face_recognition_sface_2021dec.onnx"

fetch() {
    local rel_path="$1"
    local out_file="$CACHE_DIR/$(basename "$rel_path")"
    if [[ -f "$out_file" ]]; then
        echo "already have $out_file"
        return
    fi
    echo "fetching $rel_path -> $out_file"
    curl -fsSL "$BASE_URL/$rel_path" -o "$out_file.tmp"
    if head -c 32 "$out_file.tmp" | grep -q "git-lfs"; then
        echo "ERROR: got an LFS pointer instead of binary content for $rel_path" >&2
        rm -f "$out_file.tmp"
        exit 1
    fi
    mv "$out_file.tmp" "$out_file"
}

fetch "$YUNET_PATH"
fetch "$SFACE_PATH"

echo "models ready in $CACHE_DIR"
ls -la "$CACHE_DIR"
