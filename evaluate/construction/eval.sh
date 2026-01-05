#!/bin/bash

# Single model evaluation script
# Usage: bash eval.sh <model_name> [--base-dir BASE_DIR] [--golden-dir GOLDEN_DIR] [--log-dir LOG_DIR] [--work-dir WORK_DIR]
# Example: bash eval.sh KGGEN_c --base-dir /path/to/dataset --golden-dir /path/to/golden

# Default paths (can be overridden by command-line arguments or environment variables)
DEFAULT_BASE_DIR="${BASE_DIR:-}"
DEFAULT_GOLDEN_DIR="${GOLDEN_DIR:-}"
DEFAULT_LOG_DIR="${LOG_DIR:-}"
DEFAULT_WORK_DIR="${WORK_DIR:-}"

# Parse command-line arguments
MODEL_NAME=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --base-dir)
            DEFAULT_BASE_DIR="$2"
            shift 2
            ;;
        --golden-dir)
            DEFAULT_GOLDEN_DIR="$2"
            shift 2
            ;;
        --log-dir)
            DEFAULT_LOG_DIR="$2"
            shift 2
            ;;
        --work-dir)
            DEFAULT_WORK_DIR="$2"
            shift 2
            ;;
        *)
            if [ -z "$MODEL_NAME" ]; then
                MODEL_NAME="$1"
            else
                echo "Error: Unknown argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if model name is provided
if [ -z "$MODEL_NAME" ]; then
    echo "Usage: bash $0 <model_name> [--base-dir BASE_DIR] [--golden-dir GOLDEN_DIR] [--log-dir LOG_DIR] [--work-dir WORK_DIR]"
    echo "Example: bash $0 KGGEN_c --base-dir /path/to/dataset --golden-dir /path/to/golden"
    echo ""
    echo "Alternatively, you can set environment variables:"
    echo "  BASE_DIR: Directory containing model prediction files"
    echo "  GOLDEN_DIR: Directory containing golden/ground truth files"
    echo "  LOG_DIR: Directory for log files (default: evaluate/construction/logs)"
    echo "  WORK_DIR: Working directory (default: current directory)"
    echo ""
    if [ -n "$DEFAULT_BASE_DIR" ] && [ -d "$DEFAULT_BASE_DIR" ]; then
        echo "Available models in $DEFAULT_BASE_DIR:"
        ls -1 "$DEFAULT_BASE_DIR"
    fi
    exit 1
fi

# Set working directory
if [ -n "$DEFAULT_WORK_DIR" ]; then
    WORK_DIR="$DEFAULT_WORK_DIR"
else
    # Try to detect KGC root directory
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    WORK_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi

cd "$WORK_DIR" || exit 1

# Set log directory
if [ -n "$DEFAULT_LOG_DIR" ]; then
    LOG_DIR="$DEFAULT_LOG_DIR"
else
    LOG_DIR="${WORK_DIR}/evaluate/construction/logs"
fi
mkdir -p "$LOG_DIR"

# Set base directory (must be provided)
if [ -z "$DEFAULT_BASE_DIR" ]; then
    echo "Error: BASE_DIR must be provided either as --base-dir argument or BASE_DIR environment variable"
    echo "Usage: bash $0 <model_name> --base-dir /path/to/dataset [--golden-dir /path/to/golden]"
    exit 1
fi
BASE_DIR="$DEFAULT_BASE_DIR"

# Set golden directory (must be provided)
if [ -z "$DEFAULT_GOLDEN_DIR" ]; then
    echo "Error: GOLDEN_DIR must be provided either as --golden-dir argument or GOLDEN_DIR environment variable"
    echo "Usage: bash $0 <model_name> --base-dir /path/to/dataset --golden-dir /path/to/golden"
    exit 1
fi
GOLDEN_DIR="$DEFAULT_GOLDEN_DIR"

MODEL_DIR="${BASE_DIR}/${MODEL_NAME}"

# Check if model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "[Error] Model directory not found: ${MODEL_DIR}"
    exit 1
fi

# Metrics to evaluate
METRICS="metrix"

# Dataset mapping (for GOLD files)
declare -A DS_MAP=(
    ["GenWiki"]="GenWiki-Hard"
    ["SCIERC"]="SCIERC"
    ["KELM-sub"]="kelm_sub"
    ["webnlg20"]="webnlg20"
    ["CaRB"]="CaRB-Expert"
)

# List of datasets to evaluate
DATASETS=("CaRB")

echo ""
echo "========================================================================"
echo "Starting evaluation for model: ${MODEL_NAME}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"

success_count=0
fail_count=0
skip_count=0

# Run evaluation for each dataset
for DS in "${DATASETS[@]}"; do
    # Map dataset directory name
    GOLD_DS="${DS_MAP[$DS]}"
    
    # Set file paths
    PRED="${MODEL_DIR}/${DS}/triples.txt"
    GOLD="${GOLDEN_DIR}/${GOLD_DS}/triples.txt"
    
    # Generate log file path (format: model_name_dataset_result)
    LOG_FILE="${LOG_DIR}/${MODEL_NAME}_${DS}_result"
    
    # Check if files exist
    if [ ! -f "$PRED" ]; then
        echo "[Skipped] ${DS}: Prediction file not found."
        ((skip_count++))
        continue
    fi
    
    if [ ! -f "$GOLD" ]; then
        echo "[Warning] ${DS}: Golden file not found (${GOLD})"
        ((skip_count++))
        continue
    fi
    
    # Run evaluation
    echo ""
    echo "----------------------------------------"
    echo "Dataset: ${DS}"
    echo "----------------------------------------"
    echo "Pred: ${PRED}"
    echo "Gold: ${GOLD}"
    echo "Log: ${LOG_FILE}"
    echo ""
    
    python3 -m evaluate.construction.main \
        --pred "$PRED" \
        --gold "$GOLD" \
        --models "$METRICS" \
        --log "$LOG_FILE"
    
    if [ $? -eq 0 ]; then
        echo "✓ Completed: ${DS}"
        ((success_count++))
    else
        echo "✗ Failed: ${DS}"
        ((fail_count++))
    fi
    echo ""
done

echo ""
echo "========================================================================"
echo "Evaluation completed for model: ${MODEL_NAME}"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"
echo "Success: ${success_count} | Failed: ${fail_count} | Skipped: ${skip_count}"
echo "Result CSV: evaluate/construction/${MODEL_NAME}_result.csv"
echo "========================================================================"
