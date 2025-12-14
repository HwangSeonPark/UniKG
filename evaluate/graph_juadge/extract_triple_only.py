# extract_relations.py
import asyncio
import re
import json
from pathlib import Path
from tqdm.asyncio import tqdm
from api_client import call_api
import argparse
import os


RELATION_PROMPT = """Goal:
Transform the text into a semantic graph (a list of triples) with the given text and entities. Extract subject-predicate-object triples. A predicate (1-3 words) defines the relationship between the subject and object. Subject and object are entities.

Note:
1. Generate triples as many as possible.
2. Each item in the list must be a triple with strictly three items.
3. Output ONLY the list format, no explanation, no JSON markdown.

Examples:
Example#1:
Text: "Shotgate Thickets is a nature reserve in the United Kingdom operated by the Essex Wildlife Trust."
Entity List: ["Shotgate Thickets", "Nature reserve", "United Kingdom", "Essex Wildlife Trust"]
Output: [["Shotgate Thickets", "instance of", "Nature reserve"], ["Shotgate Thickets", "country", "United Kingdom"], ["Shotgate Thickets", "operator", "Essex Wildlife Trust"]]

Example#2:
Text: "The Eiffel Tower, located in Paris, France, is a famous landmark and a popular tourist attraction. It was designed by the engineer Gustave Eiffel and completed in 1889."
Entity List: ["Eiffel Tower", "Paris", "France", "landmark", "Gustave Eiffel", "1889"]
Output: [["Eiffel Tower", "located in", "Paris"], ["Eiffel Tower", "located in", "France"], ["Eiffel Tower", "instance of", "landmark"], ["Eiffel Tower", "attraction type", "tourist attraction"], ["Eiffel Tower", "designed by", "Gustave Eiffel"], ["Eiffel Tower", "completion year", "1889"]]

Now, your turn:
Text: {text}
Entity List: {ents}
Output:"""


def parse(raw):
    """
    다양한 모델 출력을 표준 형식으로 파싱
    목표 형식: [['subj', 'pred', 'obj'], ...]
    """
    raw = raw.strip()
    
    # JSON 마크다운 블록 제거
    if "```json" in raw:
        start = raw.find("```json") + 7
        end = raw.rfind("```")
        if end > start:
            raw = raw[start:end].strip()
    elif "```" in raw:
        start = raw.find("```") + 3
        end = raw.rfind("```")
        if end > start:
            raw = raw[start:end].strip()
    
    # 설명 텍스트 제거
    lines = raw.split('\n')
    clean = []
    for line in lines:
        l = line.strip()
        # 설명 문장 스킵
        if any(l.startswith(x) for x in ["Based on", "Here is", "Note that", "I will", "The semantic", "Semantic Graph:", "Output:"]):
            continue
        if l and not l.startswith('#'):
            clean.append(l)
    raw = ' '.join(clean)
    
    # 모든 트리플 추출 시도
    trips = []
    
    # 방법 1: 중첩 리스트 형태 [["a","b","c"], ["d","e","f"]]
    match = re.search(r'\[\s*\[.*\]\s*\]', raw, re.DOTALL)
    if match:
        try:
            txt = match.group(0).replace("'", '"')
            data = json.loads(txt)
            for item in data:
                if isinstance(item, list) and len(item) == 3:
                    trips.append(item)
        except:
            pass
    
    # 방법 2: 연속된 개별 트리플 ["a","b","c"] ["d","e","f"]
    if not trips:
        pat = r'\[\s*"[^"]+"\s*,\s*"[^"]+"\s*,\s*"[^"]+"\s*\]'
        mats = re.findall(pat, raw)
        for m in mats:
            try:
                txt = m.replace("'", '"')
                item = json.loads(txt)
                if isinstance(item, list) and len(item) == 3:
                    trips.append(item)
            except:
                pass
    
    # 방법 3: 작은따옴표 버전 ['a','b','c'] ['d','e','f']
    if not trips:
        pat = r"\[\s*'[^']+'\s*,\s*'[^']+'\s*,\s*'[^']+'\s*\]"
        mats = re.findall(pat, raw)
        for m in mats:
            try:
                txt = m.replace("'", '"')
                item = json.loads(txt)
                if isinstance(item, list) and len(item) == 3:
                    trips.append(item)
            except:
                pass
    
    # 결과 반환
    if trips:
        return str(trips).replace('"', "'")
    else:
        return "[]"


async def extract_relations(texts, ents, model_info, max_c, out_dir):
    sema = asyncio.Semaphore(max_c)
    out_f = out_dir / "triples.txt"
    
    # 순서 보장을 위한 결과 저장 리스트
    ress = [None] * len(texts)
    
    async def _extract(idx, text, ent):
        async with sema:
            prompt = RELATION_PROMPT.format(text=text, ents=ent)
            res = await call_api(prompt, model_info)
            # 파싱 처리
            clean = parse(res)
            ress[idx] = clean
            return clean
    
    out_f.write_text("")
    
    tasks = [_extract(i, t, e) for i, (t, e) in enumerate(zip(texts, ents))]
    await tqdm.gather(*tasks, desc="Extracting relations")
    
    # 순서대로 파일에 기록
    with open(out_f, "w", encoding="utf-8") as f:
        for r in ress:
            if r is not None:
                f.write(f"{r}\n")


# 직접 실행할 수 있도록 추가
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities_file", required=True, help="Path to entities.txt file")
    parser.add_argument("--text_file", required=True, help="Path to articles.txt file")
    parser.add_argument("--model", required=True, choices=["gpt-5", "gpt-5-mini", "mistral", "qwen", "llama-8b"])
    parser.add_argument("--max_concurrent", type=int, default=8)
    args = parser.parse_args()
    
    # 모델 설정
    MODELS = {
        "mistral": {"name": "mistralai/Mistral-7B-Instruct-v0.3", "port": 8100},
        "qwen": {"name": "Qwen/Qwen2.5-7B-Instruct", "port": 8101},
        "llama-8b": {"name": "meta-llama/Llama-3.1-8B-Instruct", "port": 8102},
        "gpt-5": {"name": "gpt-4o", "port": None, "provider": "openai"},
        "gpt-5-mini": {"name": "gpt-4o-mini", "port": None, "provider": "openai"},
    }
    
    model_info = MODELS[args.model]
    
    # OpenAI API 키 확인
    if model_info.get("provider") == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        model_info["api_key"] = api_key
    
    # 파일 읽기
    entities_file = Path(args.entities_file)
    text_file = Path(args.text_file)
    
    with open(entities_file, "r") as f:
        ents = [line.strip() for line in f if line.strip()]
    
    with open(text_file, "r") as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(texts)} texts and {len(ents)} entity lists")
    
    if len(texts) != len(ents):
        raise ValueError(f"Mismatch: {len(texts)} texts but {len(ents)} entity lists")
    
    # 출력 디렉토리는 entities_file과 같은 디렉토리에 저장
    out_dir = entities_file.parent
    
    # 실행
    asyncio.run(extract_relations(texts, ents, model_info, args.max_concurrent, out_dir))
    print(f"Done! Triples saved to {out_dir / 'triples.txt'}")


