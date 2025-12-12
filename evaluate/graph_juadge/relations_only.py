# main_relations_only.py
# 이미 추출된 엔티티로부터 관계만 추출하는 스크립트
import argparse
import asyncio
import os
from pathlib import Path
from extract_relations import extract_relations


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
    parser = argparse.ArgumentParser(description="엔티티 파일로부터 관계만 추출")
    parser.add_argument("--entities_file", required=True, help="추출된 엔티티 파일 경로")
    parser.add_argument("--text_file", required=True, help="원본 텍스트 파일 경로")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--max_concurrent", type=int, default=8, help="동시 처리 개수")
    parser.add_argument("--start_idx", type=int, default=0, help="시작 인덱스 (0부터)")
    parser.add_argument("--end_idx", type=int, default=None, help="종료 인덱스 (None이면 끝까지)")
    return parser.parse_args()


async def main():
    args = parse_args()
    
    # 엔티티 파일 읽기
    ent_path = Path(args.entities_file)
    txt_path = Path(args.text_file)
    
    if not ent_path.exists():
        raise FileNotFoundError(f"엔티티 파일을 찾을 수 없음: {ent_path}")
    if not txt_path.exists():
        raise FileNotFoundError(f"텍스트 파일을 찾을 수 없음: {txt_path}")
    
    # 파일 읽기
    with open(ent_path, "r", encoding="utf-8") as f:
        ents = [line.strip() for line in f if line.strip()]
    
    with open(txt_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    
    # 인덱스 범위 설정
    start = args.start_idx
    end = args.end_idx if args.end_idx is not None else len(texts)
    
    if len(texts) != len(ents):
        print(f"경고: 텍스트({len(texts)})와 엔티티({len(ents)}) 개수 불일치")
        end = min(end, len(texts), len(ents))
    
    # 범위 선택
    sel_txt = texts[start:end]
    sel_ent = ents[start:end]
    
    print(f"\n=== 관계 추출 시작 ===")
    print(f"모델: {args.model}")
    print(f"범위: {start} ~ {end-1} (총 {len(sel_txt)}개)")
    print(f"엔티티 파일: {ent_path}")
    print(f"텍스트 파일: {txt_path}")
    
    # 모델 설정
    mcfg = MODELS[args.model]
    
    # OpenAI 모델인 경우 API 키 확인
    if mcfg.get("provider") == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않음")
        mcfg["api_key"] = key
    
    # 출력 디렉토리 (엔티티 파일과 같은 위치)
    out = ent_path.parent
    
    # 관계 추출 실행
    await extract_relations(sel_txt, sel_ent, mcfg, args.max_concurrent, out)
    
    print(f"\n=== 완료 ===")
    print(f"트리플 저장: {out / 'triples.txt'}")


if __name__ == "__main__":
    asyncio.run(main())

