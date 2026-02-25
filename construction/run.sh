#!/bin/bash

if [ $# -lt 2 ]; then
    echo "Usage: bash run.sh <model_name> <dataset_name> [mine_articles_dir]"
    echo "  model_name: qwen, mistral, gpt, etc."
    echo "  dataset_name: webnlg20, carb-expert, kelm-sub, genwiki-hard, scierc, all, or mine"
    echo "  mine_articles_dir: path to mine dataset articles directory (required if dataset_name is mine)"
    exit 1
fi

mdl_nm="$1"
dset_nm="$2"
MINE_ARTICLES_DIR="$3"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
INPUT_DIR="$SCRIPT_DIR/input"
OUTPUT_DIR="$SCRIPT_DIR/output"

if [ -f "$BASE_DIR/.env" ]; then
    set -a
    source "$BASE_DIR/.env"
    set +a
fi

IS_GPT=false
if [[ "$mdl_nm" == "gpt"* ]] || [[ "$mdl_nm" == "GPT"* ]]; then
    IS_GPT=true
fi

if [ "$IS_GPT" != true ] && [ -z "$port" ]; then
    echo "Error: port is not set. Add port=<number> to .env"
    exit 1
fi

export VLLM_HOST="${VLLM_HOST:-localhost}"
export REFINER_HOST="${REFINER_HOST:-localhost}"

if [ "$IS_GPT" = true ]; then
    export REFINER_MODEL="gpt-5.1"
else
    export REFINER_MODEL="${REFINER_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
    export DEFAULT_VLLM_PORT="$port"
fi

IS_REF_GPT=false
if [[ "$REFINER_MODEL" == "gpt"* ]] || [[ "$REFINER_MODEL" == "GPT"* ]]; then
    IS_REF_GPT=true
fi

if [ "$IS_REF_GPT" != true ]; then
    export REFINER_PORT="$port"
fi

export REFINER_MAX_WORKERS="${REFINER_MAX_WORKERS:-10}"
export REFINER_MAX_TOKENS="${REFINER_MAX_TOKENS:-10000}"
export VLLM_API_KEY="${VLLM_API_KEY:-none}"

declare -A DSET_MAP
DSET_MAP["webnlg20"]="webnlg20"
DSET_MAP["carb"]="carb-expert"
DSET_MAP["carb-expert"]="carb-expert"
DSET_MAP["kelm-sub"]="kelm-sub"
DSET_MAP["genwiki"]="genwiki-hard"
DSET_MAP["genwiki-hard"]="genwiki-hard"
DSET_MAP["scierc"]="scierc"

get_articles_path() {
    local dset="$1"
    if [ "$dset" == "mine" ]; then
        if [ -n "$MINE_ARTICLES_DIR" ]; then
            echo "$MINE_ARTICLES_DIR"
        elif [ -n "$MINE_ARTICLES_PATH" ]; then
            echo "$MINE_ARTICLES_PATH"
        else
            echo "Error: mine dataset requires articles directory path" >&2
            echo "  Provide as argument: bash run.sh <model> mine <articles_dir>" >&2
            echo "  Or set environment variable: export MINE_ARTICLES_PATH=<articles_dir>" >&2
            exit 1
        fi
    else
        local dset_lower=$(echo "$dset" | tr '[:upper:]' '[:lower:]')
        local canon="${DSET_MAP[$dset_lower]:-$dset_lower}"
        echo "$BASE_DIR/datasets/$canon/articles.txt"
    fi
}

process_dataset() {
    local dset="$1"
    local articles_path=$(get_articles_path "$dset")
    local dset2="$dset"

    echo "Processing dataset: $dset"

    if [ "$dset" == "mine" ]; then
        echo "Step 1: Split articles"
        python3 "$SCRIPT_DIR/utils/split.py" "$articles_path" "$INPUT_DIR"
        if [ $? -ne 0 ]; then
            echo "Error: split.py failed"
            exit 1
        fi
    fi

    echo "Step 2: Extract triples"
    if [ "$dset" == "mine" ]; then
        python3 "$SCRIPT_DIR/extractor.py" "$mdl_nm" "$dset" "$INPUT_DIR" "$OUTPUT_DIR"
    else
        dset2=$(basename "$(dirname "$articles_path")")
        python3 "$SCRIPT_DIR/extractor.py" "$mdl_nm" "$dset2" "$articles_path" "$OUTPUT_DIR"
    fi
    if [ $? -ne 0 ]; then
        echo "Error: extract failed"
        exit 1
    fi

    echo "Step 3: Refine triples"
    if [ "$dset" == "mine" ]; then
        python3 "$SCRIPT_DIR/run.py" "$mdl_nm" "$dset" "$INPUT_DIR" "$OUTPUT_DIR"
    else
        python3 "$SCRIPT_DIR/run.py" "$mdl_nm" "$dset2" "$articles_path" "$OUTPUT_DIR"
    fi
    if [ $? -ne 0 ]; then
        echo "Error: verifier failed"
        exit 1
    fi

    if [ "$dset" == "mine" ]; then
        echo "Step 4: Merge triples"
        python3 "$SCRIPT_DIR/utils/merge_triples.py" "$INPUT_DIR" "$OUTPUT_DIR" "$dset"
        if [ $? -ne 0 ]; then
            echo "Error: merge_triples.py failed"
            exit 1
        fi
        echo "Step 5: Clean up split articles"
        rm -f "$INPUT_DIR"/article_*.txt
        rm -f "$INPUT_DIR"/articles.txt
    else
        echo "Saved triples to: $OUTPUT_DIR/$dset2/triples.txt"
    fi

    echo "Dataset $dset processing completed."
}

cd "$SCRIPT_DIR"
mkdir -p "$INPUT_DIR"
mkdir -p "$OUTPUT_DIR"
export PYTHONUNBUFFERED=1

if [ "$dset_nm" == "all" ]; then
    for dset in "webnlg20" "carb-expert" "kelm-sub" "genwiki-hard" "scierc"; do
        process_dataset "$dset"
    done
elif [ "$dset_nm" == "mine" ]; then
    process_dataset "mine"
else
    process_dataset "$dset_nm"
fi

echo "All steps completed successfully."
