#!/usr/bin/env python3
import argparse
import os
import json
import sys
from pathlib import Path

# 로컬 src 경로를 파이썬 모듈 검색 경로에 추가
root = Path(__file__).resolve().parent
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from kg_gen import KGGen


def read_lines(p):
    with open(p, "r", encoding="utf-8") as f:
        return [l.strip() for l in f.readlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="입력 텍스트 라인 파일 경로")
    ap.add_argument("--outdir", required=True, help="출력 폴더")
    ap.add_argument("--start", type=int, default=0, help="시작 라인 인덱스(0-based)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    # HF + transformers 로컬 모델 이름 (기본: mistralai/Mixtral-8x7B-Instruct-v0.1)
    hf_model = os.getenv("HF_MODEL_NAME", "mistralai/Mixtral-8x7B-Instruct-v0.1")

    # HF 로컬 경로로 초기화: api_key/api_base 없이 모델 이름만 전달
    kg = KGGen(
        model=hf_model,
        max_tokens=512,
        temperature=0.0,
        api_key=None,
        api_base=None,
    )

    lines = read_lines(args.input)
    start = max(0, args.start)
    triples_path = os.path.join(args.outdir, "triples.txt")
    graphs_dir = os.path.join(args.outdir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    with open(triples_path, "w", encoding="utf-8") as wf:
        total = len(lines)
        for i, text in enumerate(lines[start:], start=start):
            print(f"[info] Processing line {i+1}/{total}")
            graph = kg.generate(
                input_data=text,
            )
            triples = list({tuple(r) for r in graph.relations})
            wf.write(json.dumps(triples, ensure_ascii=False) + "\n")
            with open(os.path.join(graphs_dir, f"graph_{i:06d}.json"), "w", encoding="utf-8") as jf:
                json.dump(
                    {
                        "entities": list(graph.entities),
                        "relations": [list(t) for t in triples],
                        "edges": list(graph.edges),
                    },
                    jf,
                    ensure_ascii=False,
                    indent=2,
                )


if __name__ == "__main__":
    main()
