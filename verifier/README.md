# Verifier Pipeline

This pipeline processes knowledge graph triple extraction and refinement from articles.

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

- **Non-GPT models**: Uses `extract.py` for extraction (e.g., qwen, mistral)
- **GPT models**: Uses `extract_gpt.py` for extraction (e.g., gpt, gpt-5.1)
- **Refinement**: Always uses `run.py` with qwen model (regardless of extraction model)

## Pipeline Steps

1. **Split Articles**: If articles exceed 2048 characters, split them at sentence boundaries
2. **Extract Triples**: Extract knowledge graph triples from articles using specified model
3. **Refine Triples**: Refine extracted triples using qwen model (always)
4. **Merge Triples**: Merge split triples back to original article count
5. **Cleanup**: Remove temporary split article files

## Output Structure

```
verifier/
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
- `CaRB` (maps to `CaRB-Expert`)
- `KELM-sub` (maps to `kelm_sub`)
- `GenWiki` (maps to `GenWiki-Hard`)
- `SCIERC`
- `mine` (requires articles directory path as 3rd argument or `MINE_ARTICLES_PATH` environment variable)
- `all` (processes all datasets except mine)

## Requirements

- Python 3.x
- Required Python packages: nltk, tiktoken (for split.py)
- OpenAI API key (for GPT models) or vLLM server (for other models)

## Notes

- Articles longer than 2048 characters are automatically split at sentence boundaries
- Split articles are merged back after processing to maintain original article count
- All paths are passed as arguments - no hardcoded paths in Python files
- Refinement step always uses qwen model regardless of extraction model

## Common Modules

The codebase uses shared modules to reduce duplication:

- `extract_base.py`: Common prompts and postprocessing functions for extraction
- `refiner_base.py`: Common refinement methods and prompt building
