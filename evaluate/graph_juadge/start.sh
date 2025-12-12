# start_all.sh
#!/bin/bash
mkdir -p logs

CUDA_VISIBLE_DEVICES=2 nohup vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --port 8100 --gpu-memory-utilization 0.3 > logs/mistral.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8101 --gpu-memory-utilization 0.3 > logs/qwen.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8102 --gpu-memory-utilization 0.3 > logs/llama.log 2>&1 &

echo "Waiting 90s..."
sleep 90

nohup python3 main.py --model all --dataset all > logs/run.log 2>&1 &
echo "Done. Monitor: tail -f logs/run.log"
