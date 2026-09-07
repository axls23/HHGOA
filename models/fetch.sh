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

# --- capture-stage models (Stage 1) -----------------------------------------
# Optional by design: the pipeline runs without them, on the heuristic gate,
# and says so. So a download failure here warns rather than aborting a setup
# that is otherwise complete.
#
#   MiniFASNetV2 / V1SE  presentation-attack detection, Apache-2.0
#                        (minivision-ai/Silent-Face-Anti-Spoofing)
#   ediffiqa_t           learned face image quality, MIT
#                        (LSIbabnikz/eDifFIQA, ONNX export by yakhyo)
#
# Both licences are permissive, which is why these two and not the better-known
# alternatives: CR-FIQA is CC BY-NC, and the InsightFace detector/recogniser
# zoo is non-commercial — the same trap PRD §G5 keeps buffalo_l out for.
ANTISPOOF_URL="https://github.com/yakhyo/face-anti-spoofing/releases/download/weights"
FIQA_URL="https://github.com/yakhyo/face-image-quality-assessment/releases/download/weights"

fetch_optional() {
    local url="$1" out_file="$CACHE_DIR/$2"
    if [[ -f "$out_file" ]]; then
        echo "already have $out_file"
        return 0
    fi
    echo "fetching $2 -> $out_file"
    if curl -fsSL "$url" -o "$out_file.tmp"; then
        mv "$out_file.tmp" "$out_file"
    else
        rm -f "$out_file.tmp"
        echo "WARNING: could not fetch $2 — the capture-stage check it powers will be reported unavailable, and the pipeline will run without it" >&2
    fi
}

fetch_optional "$ANTISPOOF_URL/MiniFASNetV2.onnx"   MiniFASNetV2.onnx
fetch_optional "$ANTISPOOF_URL/MiniFASNetV1SE.onnx" MiniFASNetV1SE.onnx
fetch_optional "$FIQA_URL/ediffiqa_t.onnx"          ediffiqa_t.onnx

# --- alternative detector (FACEANCHOR_DETECTOR=yolo) -------------------------
# YOLOv8-face, 5 keypoints, same row layout as YuNet so nothing downstream
# changes. NOTE THE LICENCE: this one is YOLOv8-derived and therefore
# GPL/AGPL, unlike every other weight above (Apache-2.0/MIT). A detector is on
# the default path and cannot be scoped to an opt-in feature the way the GPL
# Groth16Verifier is, so selecting it is a licensing decision about the whole
# project. Left optional for exactly that reason.
YOLO_FACE_URL="https://github.com/akanametov/yolo-face/releases/download/1.0.0"
fetch_optional "$YOLO_FACE_URL/yolov8n-face.onnx"   yolov8n-face.onnx

echo "models ready in $CACHE_DIR"
ls -la "$CACHE_DIR"
