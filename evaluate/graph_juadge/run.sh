#!/bin/bash
set -euo pipefail

mkdir -p logs

# 사용법:
# 1) 기존 데이터셋(GenWiki/SCIERC/...) 전체: ./run.sh dnztri all
# 2) webnlg20 단일 파일: ./run.sh all webnlg20 /ABS/PATH/articles.txt
#
# 인자:
# $1 MODE: dnztri | all | dnz | tri | ent
# $2 DSET: all | GenWiki | SCIERC | KELM-sub | CaRB | webnlg20
# $3 FILE: (DSET=webnlg20일 때) articles.txt 절대경로

MODE="${1:-dnztri}"
DSET="${2:-all}"
FILE="${3:-}"

wait_srv () {
  PORT="$1"
  MODL="$2"
  for _ in $(seq 1 240); do
    if curl -sS "http://127.0.0.1:${PORT}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer dummy" \
      -d "{\"model\":\"${MODL}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1,\"temperature\":0}" \
      > /dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

# webnlg20이면 main.py에 text_file/dset_name을 같이 넘긴다
mk_arg () {
  if [[ "${DSET}" == "webnlg20" ]]; then
    if [[ -z "${FILE}" ]]; then
      return 1
    fi
    # dset_name을 webnlg20으로 고정
    printf -- "--text_file %s --dset_name webnlg20" "${FILE}"
  else
    printf -- "--dataset %s" "${DSET}"
  fi
}

ARG="$(mk_arg)"

# GPT는 병렬(동시) 실행
# python3 main.py --model gpt-5.1 --task "${MODE}" ${ARG} > "logs/gpt-5.1_${DSET}_${MODE}.log" 2>&1 &
# P1=$!
# python3 main.py --model gpt-5-mini --task "${MODE}" ${ARG} > "logs/gpt-5-mini_${DSET}_${MODE}.log" 2>&1 &
# P2=$!

# 오픈소스는 vLLM을 모델별로 직렬 실행
CUDA_VISIBLE_DEVICES=3 vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
  --port 8100 --gpu-memory-utilization 0.8 > logs/mistral_server.log 2>&1 &
S1=$!
wait_srv 8100 "mistralai/Mistral-7B-Instruct-v0.3"
python3 main.py --model mistral --task "${MODE}" ${ARG} > "logs/mistral_${DSET}_${MODE}.log" 2>&1
kill "${S1}"
wait "${S1}" 2>/dev/null || true
sleep 20

CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 8101 --gpu-memory-utilization 0.8 > logs/qwen_server.log 2>&1 &
S2=$!
wait_srv 8101 "Qwen/Qwen2.5-7B-Instruct"
python3 main.py --model qwen --task "${MODE}" ${ARG} > "logs/qwen_${DSET}_${MODE}.log" 2>&1
kill "${S2}"
wait "${S2}" 2>/dev/null || true
sleep 20

CUDA_VISIBLE_DEVICES=3 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --port 8102 --gpu-memory-utilization 0.8 > logs/llama_server.log 2>&1 &
S3=$!
wait_srv 8102 "meta-llama/Llama-3.1-8B-Instruct"
python3 main.py --model llama-8b --task "${MODE}" ${ARG} > "logs/llama_${DSET}_${MODE}.log" 2>&1
kill "${S3}"
wait "${S3}" 2>/dev/null || true

wait "${P1}"
wait "${P2}"
