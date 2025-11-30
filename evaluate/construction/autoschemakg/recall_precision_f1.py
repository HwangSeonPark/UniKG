import os
import json
import ast  
from openai import OpenAI

# ========= 0. OpenRouter 클라이언트 설정 =========
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),  # 환경변수에 키 세팅 필요
)

# ========= 1. 평가용 프롬프트 템플릿 =========
EVAL_PROMPT_TEMPLATE = """
You are a precise information extraction evaluator.

You are given:
1. ORIGINAL TEXT
2. GOLD TRIPLES: the list of true triples.
3. PREDICTED TRIPLES: the list of triples produced by a system.

Your task:
- For EACH predicted triple, decide whether it is CORRECT or INCORRECT.
- A predicted triple is CORRECT if:
  (a) It is semantically equivalent to one of the GOLD TRIPLES
      (even if wording differs), AND
  (b) It is supported by the ORIGINAL TEXT.
- Otherwise, it is INCORRECT.
- When matching, focus on meaning, not exact string equality
  (e.g., "location" vs "locatedIn" can be treated as the same
   if they clearly express the same fact in the text).

Important constraints:
- Use ONLY the ORIGINAL TEXT as factual source.
- GOLD TRIPLES tell you which facts are intended as ground truth,
  but you must still check if a predicted triple is supported by the text.
- Do NOT add new facts that are not in the GOLD TRIPLES.

Output format (JSON only):
Return a JSON object with EXACTLY these keys:
- "correct_predicted_indices": list of 0-based indices of PREDICTED_TRIPLES that are correct.
- "incorrect_predicted_indices": list of 0-based indices of PREDICTED_TRIPLES that are incorrect.
- "matched_gold_indices": list of 0-based indices of GOLD_TRIPLES that are covered by at least one correct predicted triple.

Use 0-based indexing for both lists.
Do NOT include any extra commentary, only valid JSON.

ORIGINAL TEXT:
----------------
{original_text}
----------------

GOLD TRIPLES (JSON, flattened list):
----------------
{gold_triples_json}
----------------

PREDICTED TRIPLES (JSON, flattened list):
----------------
{pred_triples_json}
----------------

Now return ONLY the JSON object.
"""


# ========= 2. DeepSeek 호출 헬퍼 =========
def ask_deepseek(prompt: str) -> str:
    completion = client.chat.completions.create(
#         model="deepseek/deepseek-chat-v3.1:free",
#         model="deepseek/deepseek-chat-v3-0324:free",    
#         model="deepseek/deepseek-r1-0528-qwen3-8b:free",
#         model="deepseek/deepseek-r1-0528:free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return completion.choices[0].message.content


# ========= 3. triples 구조 평탄화(flatten) =========
def flatten_triple_groups(triples):
    """
    triples 예시:
      [
        ['Trane', 'location', 'Swords,_Dublin'],
        ['Ciudad_Ayala', 'populationMetro', '1777539'],
        ...
      ]
    또는
      [
        [
          ['Trane','location','Swords,_Dublin'],
          ['Trane','country','Ireland']
        ],
        ...
      ]
    """
    flat = []
    for item in triples:
        if isinstance(item, list) and item and isinstance(item[0], list):
            # [ [triple, triple, ...], ... ] 구조
            for t in item:
                flat.append(t)
        else:
            # 이미 ['s','p','o'] 형태라고 가정
            flat.append(item)
    return flat


# ========= 4. txt 폴더에서 파일 읽기 =========

# 텍스트는 "각 줄 = 하나의 ORIGINAL TEXT"
def read_text_file(path: str):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 빈 줄 스킵
            lines.append(line)
    return lines


# 트리플 파일: 각 줄에 1개 또는 여러 개 트리플
def read_triple_file(path: str):
    """
    각 줄의 예시 형식:

    - 트리플 1개만 있는 경우:
        ['Ciudad_Ayala', 'governmentType', 'Council-manager_government']

    - 여러 개 트리플이 있는 경우 (리스트로 묶어서):
        [['Ciudad_Ayala', 'populationMetro', '1777539'],
         ['Ciudad_Ayala', 'leaderTitle', 'City Manager']]

    반환 형식:
        [
          [ ['s','p','o'] ],                        # 1개짜리인 줄
          [ ['s1','p1','o1'], ['s2','p2','o2'] ],   # 여러 개짜리인 줄
          ...
        ]
    """
    per_line_triples = []

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            # 끝에 콤마가 붙어있는 경우 방지 (예: "...],")
            if line.endswith(","):
                line = line[:-1].rstrip()

            obj = ast.literal_eval(line)

            # obj가 트리플 하나인지, 트리플 리스트인지 판별
            # - ['s','p','o']   -> obj[0]는 str, list 아님 → 1개짜리로 보고 [obj]로 감싸기
            # - [['s','p','o']] -> obj[0]는 list            → 여러 개(or 최소 1개)짜리 그대로 사용
            if isinstance(obj, list) and obj and isinstance(obj[0], list):
                # 이미 트리플 리스트 형식
                per_line_triples.append(obj)
            else:
                # 트리플 1개만 있는 경우로 보고, 한 줄을 [obj]로 통일
                per_line_triples.append([obj])

    return per_line_triples


# ========= 5. 평가 함수 (TP, FP, FN, Recall) =========
def evaluate_with_deepseek(text: str, gold_triples, pred_triples):
    # 1) flat list로 변환
    gold_flat = flatten_triple_groups(gold_triples)
    pred_flat = flatten_triple_groups(pred_triples)

    gold_json = json.dumps(gold_flat, ensure_ascii=False, indent=2)
    pred_json = json.dumps(pred_flat, ensure_ascii=False, indent=2)

    # 2) 프롬프트 생성
    prompt = EVAL_PROMPT_TEMPLATE.format(
        original_text=text,
        gold_triples_json=gold_json,
        pred_triples_json=pred_json,
    )

    # 3) DeepSeek judge 호출
    raw = ask_deepseek(prompt).strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        print("WARN: Failed to parse judge JSON. Raw output:")
        print(raw)
        # 실패 시 일단 전부 incorrect로 처리
        result = {
            "correct_predicted_indices": [],
            "incorrect_predicted_indices": list(range(len(pred_flat))),
            "matched_gold_indices": [],
        }

    correct_idx = result.get("correct_predicted_indices", [])
    incorrect_idx = result.get("incorrect_predicted_indices", [])
    matched_gold_idx = result.get("matched_gold_indices", [])

    # 4) TP, FP, FN 계산
    TP = len(correct_idx)
    FP = len(incorrect_idx)
    FN = max(len(gold_flat) - len(set(matched_gold_idx)), 0)

    denom = TP + FN
    recall = TP / denom if denom > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "num_gold_triples": len(gold_flat),
        "num_pred_triples": len(pred_flat),
        "judge_raw": raw,  # 디버깅용
    }


# ========= 6. 실행 예시 =========
if __name__ == "__main__":
    # txt 폴더 안의 파일 경로 (원하는 대로 수정 가능)
    TEXT_PATH = "/home/hspark/hspark/unikg/datasets/example/construction/articles.txt"
    GOLD_PATH = "/home/hspark/hspark/unikg/datasets/example/construction/triples_gold.txt"
    PRED_PATH = "/home/hspark/hspark/unikg/datasets/example/construction/triples.txt"

    # 각 줄 단위로 읽기
    text_lines = read_text_file(TEXT_PATH)        # list[str]
    gold_by_line = read_triple_file(GOLD_PATH)    # list[list[triple]]
    pred_by_line = read_triple_file(PRED_PATH)    # list[list[triple]]

    n = min(len(text_lines), len(gold_by_line), len(pred_by_line))
    if not (len(text_lines) == len(gold_by_line) == len(pred_by_line)):
        print(
            f"WARN: line count mismatch. text={len(text_lines)}, "
            f"gold={len(gold_by_line)}, pred={len(pred_by_line)}. "
            f"Using first {n} lines only."
        )

    total_TP = total_FP = total_FN = 0
    total_gold = total_pred = 0

    for i in range(n):
        t = text_lines[i]
        gold_group = gold_by_line[i]   # 이 줄의 gold 트리플들
        pred_group = pred_by_line[i]   # 이 줄의 pred 트리플들

        metrics = evaluate_with_deepseek(t, gold_group, pred_group)

        total_TP += metrics["TP"]
        total_FP += metrics["FP"]
        total_FN += metrics["FN"]
        total_gold += metrics["num_gold_triples"]
        total_pred += metrics["num_pred_triples"]

    denom_recall = total_TP + total_FN
    denom_prec = total_TP + total_FP

    macro_recall = total_TP / denom_recall if denom_recall > 0 else 0.0
    macro_precision = total_TP / denom_prec if denom_prec > 0 else 0.0
    macro_f1 = (
        2 * macro_precision * macro_recall / (macro_precision + macro_recall)
        if (macro_precision + macro_recall) > 0
        else 0.0
    )

    print("=== Evaluation Result (line-wise aggregated) ===")
    print(f"TP: {total_TP}")
    print(f"FP: {total_FP}")
    print(f"FN: {total_FN}")
    print(f"Precision: {macro_precision}")
    print(f"Recall: {macro_recall}")
    print(f"F1: {macro_f1}")
    print(f"num_gold_triples: {total_gold}")
    print(f"num_pred_triples: {total_pred}")
