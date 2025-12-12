# run_all.sh
#!/bin/bash

echo "=== Starting Mistral ==="
CUDA_VISIBLE_DEVICES=2 vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --port 8100 --gpu-memory-utilization 0.8 > logs/mistral_server.log 2>&1 &
MISTRAL_PID=$!

sleep 90

python3 main.py --model mistral --dataset all

kill $MISTRAL_PID
sleep 10

echo "=== Starting Qwen ==="
CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8101 --gpu-memory-utilization 0.8 > logs/qwen_server.log 2>&1 &
QWEN_PID=$!

sleep 90

python3 main.py --model qwen --dataset all

kill $QWEN_PID
sleep 10

echo "=== Starting Llama ==="
CUDA_VISIBLE_DEVICES=2 vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8102 --gpu-memory-utilization 0.8 > logs/llama_server.log 2>&1 &
LLAMA_PID=$!

sleep 90

python3 main.py --model llama-8b --dataset all

kill $LLAMA_PID

echo "=== All Done ==="
