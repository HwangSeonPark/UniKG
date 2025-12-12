# main.py
import argparse
import asyncio
import os
from pathlib import Path
from extract_entities import extract_entities
from extract_relations import extract_relations


# main.py
MODELS = {
    "mistral": {"name": "mistralai/Mistral-7B-Instruct-v0.3", "port": 8100},
    "qwen": {"name": "Qwen/Qwen2.5-7B-Instruct", "port": 8101},
    "llama-8b": {"name": "meta-llama/Llama-3.1-8B-Instruct", "port": 8102},
    "gpt-5": {"name": "gpt-4o", "port": None, "provider": "openai"},
    "gpt-5-mini": {"name": "gpt-4o-mini", "port": None, "provider": "openai"},
}


DATASETS = {
    "GenWiki": "/home/hyyang/my_workspace/KGC/datasets/construction/GenWiki-Hard/articles.txt",
    "SCIERC": "/home/hyyang/my_workspace/KGC/datasets/construction/SCIERC/articles.txt",
    "KELM-sub": "/home/hyyang/my_workspace/KGC/datasets/construction/kelm_sub/articles.txt",
    "CaRB": "/home/hyyang/my_workspace/KGC/datasets/construction/CaRB-Expert/articles.txt"
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()) + ["all"])
    parser.add_argument(
        "--dataset", 
        required=True, 
        nargs='+',  # 여러 개의 값을 받을 수 있도록 수정
        choices=list(DATASETS.keys()) + ["all"]
    )
    parser.add_argument("--max_concurrent", type=int, default=8)
    return parser.parse_args()


async def process(model, dset, args):
    with open(DATASETS[dset]) as f:
        texts = [l.strip() for l in f if l.strip()]
    
    out = Path("extract_LLM") / dset / model
    out.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{model} on {dset}: {len(texts)} texts")
    
    # OpenAI 모델인 경우 API 키 확인
    model_config = MODELS[model]
    if model_config.get("provider") == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it before running OpenAI models."
            )
        model_config["api_key"] = api_key
    
    ents = await extract_entities(texts, model_config, args.max_concurrent, out)
    await extract_relations(texts, ents, model_config, args.max_concurrent, out)
    
    print(f"Done: {model} on {dset}")


async def main():
    args = parse_args()
    models = list(MODELS.keys()) if args.model == "all" else [args.model]
    
    # dataset이 리스트로 들어오므로 처리 수정
    if "all" in args.dataset:
        dsets = list(DATASETS.keys())
    else:
        dsets = args.dataset
    
    for model in models:
        for dset in dsets:
            await process(model, dset, args)


if __name__ == "__main__":
    asyncio.run(main())
