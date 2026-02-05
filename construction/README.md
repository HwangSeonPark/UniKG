# Construction Pipeline

This pipeline processes knowledge graph triple extraction and verification/refinement from articles.

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
- **Verification/refinement**: Uses `run.py` (default: local refiner). If `REFINER_MODEL` starts with `gpt`, it will use OpenAI.

## Pipeline Steps

1. **Split Articles**: If articles exceed 2048 characters, split them at sentence boundaries
2. **Extract Triples**: Extract knowledge graph triples from articles using specified model
3. **Verify/Refine Triples**: Verify/refine extracted triples using qwen model (always)
4. **Merge Triples**: Merge split triples back to original article count
5. **Cleanup**: Remove temporary split article files

## Output Structure

```
construction/
├── input/              # Temporary split articles
│   ├── articles.txt    # All split articles (one per line)
│   └── mapping.json    # Mapping from original to split indices
└── output/
    └── {dataset_name}/
        ├── extract_triples.txt  # Extracted triples
        ├── refined_triples.txt  # Refined triples
        └── triples.txt          # Final merged triples
```

## Dataset Names

- `webnlg20`
- `carb-expert`
- `kelm_sub`
- `genwiki-hard`
- `scierc`
- `mine` (requires articles directory path as 3rd argument or `MINE_ARTICLES_PATH` environment variable)
- `all` (processes all datasets except mine)

## Requirements

- Python 3.x
- Required Python packages: nltk (for `split.py`), openai
- OpenAI API key (for GPT models) or vLLM server (for non-GPT models)

## Notes

- Articles longer than 2048 characters are automatically split at sentence boundaries
- Split articles are merged back after processing to maintain original article count
- All paths are passed as arguments - no hardcoded paths in Python files
- Refinement step defaults to local refiner; if `REFINER_MODEL` starts with `gpt`, it will use OpenAI

## Common Modules

The codebase keeps the pipeline modules minimal:

- `extractor.py`: prompt + extraction + postprocessing
- `verifier.py`: prompt building + refinement/extraction helpers
