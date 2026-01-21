# Canonicalization Pipeline

This pipeline performs KG canonicalization from row-wise triples.

## Input / Output

Input:
canonicalization/input/{dataset}/triples.txt

Each line must be:
[["h","r","t"], ["h2","r2","t2"], ...]

Output:
canonicalization/output/{dataset}/triples.txt

Intermediate:
canonicalization/work/{dataset}/*

## Usage

bash run.sh <gpt|qwen|mistral> <dataset> [api_base]

### GPT

export OPENAI_API_KEY=sk-xxxx
bash run.sh gpt CaRB

### Qwen (vLLM)

bash run.sh qwen CaRB http://localhost:8000/v1

### Mistral (vLLM)

bash run.sh mistral CaRB http://localhost:8001/v1

## Models

- GPT-5.1
- Qwen2.5-7B-Instruct
- Mistral-7B-Instruct-v0.3

## Steps

1. Case-insensitive triple deduplication
2. HEAD/TAIL focus conversion
3. Entity typing (few-shot + adaptive batching)
4. Merge HEAD/TAIL predictions
5. SBERT embedding with cache
6. Entity/Relation canonicalization (BM25 + cosine + LLM, elimination-style)
7. Strip embeddings
8. Restore row-wise triples

## Prompt Files

canonicalization/prompt/entity_types_v1.json  
canonicalization/prompt/fewshot_entity_typing_v1.jsonl

## Requirements

pip install openai sentence-transformers rank_bm25 scikit-learn
