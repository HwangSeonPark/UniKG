#!/bin/bash
# 특정 데이터셋과 모델에 대한 디노이즈 및 트리플 추출 스크립트

set -euo pipefail

cd "$(dirname "$0")"

# 1. KELM-sub의 gpt-5 모델로 트리플 추출 (denoise.txt와 entities.txt가 이미 있음)
echo "=== KELM-sub 트리플 추출 ==="
python3 main.py \
    --model gpt-5-mini \
    --dataset webnlg20 \
    --task tri \
    --src dnz \
    --max_concurrent 8

echo ""
