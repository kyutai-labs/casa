#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 --model_id MODEL_ID [--batch_size N] [--img_size N]"
    echo ""
    echo "  --model_id    Required. e.g. CASA-Helium1-VL-2B, CASA-Qwen2_5-VL-3B"
    echo "  --batch_size  Batch size (default: 16)"
    echo "  --img_size    Image size (default: 896)"
    exit 1
}

MODEL_ID=""
BATCH_SIZE=16
IMG_SIZE=896

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_id)   MODEL_ID="$2";   shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --img_size)   IMG_SIZE="$2";   shift 2 ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

if [[ -z "$MODEL_ID" ]]; then
    echo "Error: --model_id is required"
    usage
fi

DATASETS=(chartqa textvqa realworldqa ai2d mme ocrbench docvqa infographic_vqa gqa)

echo "model_id:   $MODEL_ID"
echo "batch_size: $BATCH_SIZE"
echo "img_size:   $IMG_SIZE"
echo "datasets:   ${DATASETS[*]}"
echo ""

for dataset in "${DATASETS[@]}"; do
    echo "=== $dataset ==="
    TRITON_CACHE_DIR=/tmp/triton_cache_dir_$USER uv run scripts/eval.py \
        "$MODEL_ID" \
        --dataset_name "$dataset" \
        --batch_size "$BATCH_SIZE" \
        --image_size "$IMG_SIZE"
    echo ""
done
