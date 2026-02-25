
# Construction Pipeline

This pipeline processes knowledge graph triple extraction and verification/refinement from articles.

```

## Environment Setup

First, create and activate the conda environment in the KGC root directory:

```bash
conda create -n unikg python=3.10.12
conda activate unikg
```

Then, install all required packages from the root directory:

```bash
pip install -r requirements.txt
```

After the installation is complete, move to the `construction` directory to run the pipeline:

```bash
cd construction
```

Create a `.env` file in the KGC root directory:

```text
OPENAI_API_KEY=sk-...        # Required for GPT models
port=8000                    # Required for vLLM models (qwen, mistral)
```



## Start vLLM server (for vLLM models)

If you use vLLM-based models (e.g., qwen, mistral), start the vLLM OpenAI-compatible server **before** running this pipeline:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 3906 \
  --dtype float16 \
  --gpu-memory-utilization 0.7 \
  --max-model-len 30000 > vllm_server.log 2>&1 &

## Usage

```bash
# Process single dataset (excluding mine)
bash run.sh <model_name> <dataset_name>

# Process all datasets (excluding mine)
bash run.sh <model_name> all

# Process mine dataset
bash run.sh <model_name> mine <articles_dir>
# Or set environment variable:
export MINE_ARTICLES_PATH=/path/to/articles
bash run.sh <model_name> mine
```

### Examples

```bash
# Process webnlg20 dataset with qwen model
bash run.sh qwen webnlg20

# Process all datasets with mistral model
bash run.sh mistral all

# Process mine dataset with gpt model
bash run.sh gpt mine /path/to/wikiqa/articles
```

## Model Support

- **Extraction**: Uses `extractor.py` for both vLLM models (e.g., qwen, mistral) and OpenAI models (e.g., gpt, gpt-5.1)
- **Verification/refinement**: Uses `run.py` with the same model family as extraction. GPT extraction uses GPT for refinement; vLLM extraction uses the vLLM refiner.

## Pipeline Steps

1. **Split Articles**: If articles exceed 2048 characters, split them at sentence boundaries
2. **Extract Triples**: Extract knowledge graph triples from articles using specified model
3. **Verify/Refine Triples**: Verify/refine extracted triples using the same model family as extraction (GPT or vLLM)
4. **Merge Triples**: Merge split triples back to original article count (with deduplication)

## Dataset Names

- `webnlg20`
- `carb-expert`
- `kelm-sub`
- `genwiki-hard`
- `scierc`
- `mine` (requires articles directory path as 3rd argument or `MINE_ARTICLES_PATH` environment variable)
- `all` (processes all datasets except mine)

## Notes

- Articles longer than 2048 characters are automatically split at sentence boundaries
- Split articles are merged back after processing to maintain original article count
- All paths are passed as arguments - no hardcoded paths in Python files
- Refinement step uses the same model family as the extraction step (auto-detected from model name)


## Common Modules

The codebase keeps the pipeline modules minimal:

- `extractor.py`: prompt loading + extraction + postprocessing
- `verifier.py`: prompt loading + refinement/extraction helpers
- `utils/split.py`: article splitting
- `utils/merge_triples.py`: triple merging with deduplication
