#!/bin/bash

# Parse command line arguments
if [ $# -lt 2 ]; then
    echo "Usage: bash run.sh <model_name> <dataset_name> [mine_articles_dir]"
    echo "  model_name: qwen, mistral, gpt, etc."
    echo "  dataset_name: webnlg20, CaRB, KELM-sub, GenWiki, SCIERC, all, or mine"
    echo "  mine_articles_dir: (optional) path to mine dataset articles directory (required if dataset_name is mine)"
    exit 1
fi

mdl_nm="$1"
dset_nm="$2"
MINE_ARTICLES_DIR="$3"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Set paths
INPUT_DIR="$SCRIPT_DIR/input"
OUTPUT_DIR="$SCRIPT_DIR/output"

# =========================
# Port Configuration
# =========================
# vLLM server port settings
# IMPORTANT: These environment variables must be set before running this script
# Example: export QWEN_PORT=<port_number>
export VLLM_HOST="${VLLM_HOST:-localhost}"
if [ -z "$QWEN_PORT" ]; then
    echo "Error: QWEN_PORT environment variable is not set. Please set it before running this script." >&2
    echo "Example: export QWEN_PORT=<port_number>" >&2
    exit 1
fi
export QWEN_PORT="$QWEN_PORT"

if [ -z "$MISTRAL_PORT" ]; then
    echo "Error: MISTRAL_PORT environment variable is not set. Please set it before running this script." >&2
    echo "Example: export MISTRAL_PORT=<port_number>" >&2
    exit 1
fi
export MISTRAL_PORT="$MISTRAL_PORT"

export DEFAULT_VLLM_PORT="${DEFAULT_VLLM_PORT:-$QWEN_PORT}"

# Refiner server port settings
# IMPORTANT: These environment variables must be set before running this script
# Example: export REFINER_PORT=<port_number>
export REFINER_HOST="${REFINER_HOST:-localhost}"
if [ -z "$REFINER_PORT" ]; then
    echo "Error: REFINER_PORT environment variable is not set. Please set it before running this script." >&2
    echo "Example: export REFINER_PORT=<port_number>" >&2
    exit 1
fi
export REFINER_PORT="$REFINER_PORT"

export REFINER_MAX_WORKERS="${REFINER_MAX_WORKERS:-10}"
export REFINER_MODEL="${REFINER_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export REFINER_MAX_TOKENS="${REFINER_MAX_TOKENS:-10000}"

# vLLM API Key setting (optional)
export VLLM_API_KEY="${VLLM_API_KEY:-none}"

# Dataset mapping (case-insensitive)
declare -A DSET_MAP
DSET_MAP["webnlg20"]="webnlg20"
DSET_MAP["carb"]="CaRB-Expert"
DSET_MAP["CaRB"]="CaRB-Expert"
DSET_MAP["KELM-sub"]="kelm_sub"
DSET_MAP["GenWiki"]="GenWiki-Hard"
DSET_MAP["SCIERC"]="SCIERC"

# Determine if GPT model
IS_GPT=false
if [[ "$mdl_nm" == "gpt"* ]] || [[ "$mdl_nm" == "GPT"* ]]; then
    IS_GPT=true
fi

# Function to get original articles path
get_articles_path() {
    local dset="$1"
    if [ "$dset" == "mine" ]; then
        # For mine dataset, use provided directory or environment variable
        if [ -n "$MINE_ARTICLES_DIR" ]; then
            echo "$MINE_ARTICLES_DIR"
        elif [ -n "$MINE_ARTICLES_PATH" ]; then
            echo "$MINE_ARTICLES_PATH"
        else
            echo "Error: mine dataset requires articles directory path" >&2
            echo "  Provide as 3rd argument: bash run.sh <model> mine <articles_dir>" >&2
            echo "  Or set environment variable: export MINE_ARTICLES_PATH=<articles_dir>" >&2
            exit 1
        fi
    else
        # Case-insensitive lookup
        local dset_lower=$(echo "$dset" | tr '[:upper:]' '[:lower:]')
        local actual_dir="${DSET_MAP[$dset_lower]:-${DSET_MAP[$dset]:-$dset}}"
        echo "$BASE_DIR/datasets/construction/$actual_dir/articles.txt"
    fi
}

# Function to process single dataset
process_dataset() {
    local dset="$1"
    local articles_path=$(get_articles_path "$dset")
    
    echo "=========================================="
    echo "Processing dataset: $dset"
    echo "=========================================="
    
    # For mine dataset, split articles first
    if [ "$dset" == "mine" ]; then
        # Step 1: Split articles (mine dataset only)
        echo ""
        echo "Step 1: Split articles"
        echo "=========================================="
        python3 "$SCRIPT_DIR/split.py" "$articles_path" "$INPUT_DIR"
        
        if [ $? -ne 0 ]; then
            echo "Error: split.py failed"
            exit 1
        fi
    fi
    
    # Step 2: Extract triples
    echo ""
    echo "=========================================="
    echo "Step 2: Extract triples"
    echo "=========================================="
    if [ "$IS_GPT" = true ]; then
        if [ "$dset" == "mine" ]; then
            python3 "$SCRIPT_DIR/extract_gpt.py" "$dset" "$INPUT_DIR" "$OUTPUT_DIR"
        else
            python3 "$SCRIPT_DIR/extract_gpt.py" "$dset" "$articles_path" "$OUTPUT_DIR"
        fi
    else
        if [ "$dset" == "mine" ]; then
            python3 "$SCRIPT_DIR/extract.py" "$mdl_nm" "$dset" "$INPUT_DIR" "$OUTPUT_DIR"
        else
            python3 "$SCRIPT_DIR/extract.py" "$mdl_nm" "$dset" "$articles_path" "$OUTPUT_DIR"
        fi
    fi
    
    if [ $? -ne 0 ]; then
        echo "Error: extract failed"
        exit 1
    fi
    
    # Step 3: Refine triples (always use qwen model)
    echo ""
    echo "=========================================="
    echo "Step 3: Refine triples (using qwen model)"
    echo "=========================================="
    if [ "$dset" == "mine" ]; then
        python3 "$SCRIPT_DIR/run.py" "qwen" "$dset" "$INPUT_DIR" "$OUTPUT_DIR"
    else
        python3 "$SCRIPT_DIR/run.py" "qwen" "$dset" "$articles_path" "$OUTPUT_DIR"
    fi
    
    if [ $? -ne 0 ]; then
        echo "Error: refiner failed"
        exit 1
    fi
    
    # Step 4: Merge triples (mine dataset only) or copy refined to triples.txt (regular datasets)
    echo ""
    echo "=========================================="
    if [ "$dset" == "mine" ]; then
        echo "Step 4: Merge triples"
        echo "=========================================="
        python3 "$SCRIPT_DIR/merge_triples.py" "$INPUT_DIR" "$OUTPUT_DIR" "$dset"
        
        if [ $? -ne 0 ]; then
            echo "Error: merge_triples.py failed"
            exit 1
        fi
        
        # Step 5: Clean up split articles
        echo ""
        echo "=========================================="
        echo "Step 5: Clean up split articles"
        echo "=========================================="
        rm -f "$INPUT_DIR"/article_*.txt
        rm -f "$INPUT_DIR"/articles.txt
        echo "Cleaned up split articles"
    else
        echo "Step 4: Save final triples"
        echo "=========================================="
        # For regular datasets, refined_triples.txt is the final triples.txt
        cp "$OUTPUT_DIR/$dset/refined_triples.txt" "$OUTPUT_DIR/$dset/triples.txt"
        echo "Saved triples to: $OUTPUT_DIR/$dset/triples.txt"
        
    fi
    
    echo ""
    echo "=========================================="
    echo "Dataset $dset processing completed!"
    echo "=========================================="
}

# Main processing
cd "$SCRIPT_DIR"

# Create necessary directories
mkdir -p "$INPUT_DIR"
mkdir -p "$OUTPUT_DIR"

export PYTHONUNBUFFERED=1

# Process datasets
if [ "$dset_nm" == "all" ]; then
    # Process all datasets (excluding mine)
    for dset in "webnlg20" "CaRB" "KELM-sub" "GenWiki" "SCIERC"; do
        process_dataset "$dset"
    done
elif [ "$dset_nm" == "mine" ]; then
    # Process mine dataset
    process_dataset "mine"
else
    # Process single dataset
    process_dataset "$dset_nm"
fi

echo ""
echo "=========================================="
echo "All steps completed successfully!"
echo "=========================================="
