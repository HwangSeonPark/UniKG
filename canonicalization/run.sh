#!/bin/bash

MODEL=$1
DATASET=$2
API_BASE=$3

if [ $# -lt 2 ]; then
  echo "Usage: bash run.sh <gpt|qwen|mistral> <dataset> [api_base]"
  echo ""
  echo "GPT:"
  echo "  export OPENAI_API_KEY=sk-xxxx"
  echo "  bash run.sh gpt CaRB"
  echo ""
  echo "vLLM (Qwen/Mistral):"
  echo "  bash run.sh qwen CaRB http://localhost:8000/v1"
  exit 1
fi

python3 run.py \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --api_base "$API_BASE"
