# stop.sh
#!/bin/bash
pkill -f "python main.py"
pkill -f "vllm serve"
echo "Stopped."
