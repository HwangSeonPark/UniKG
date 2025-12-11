import os
import json
from pathlib import Path
from typing import List, Dict
from collections import defaultdict, Counter
import random

from tqdm import tqdm
from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer

# ===== 경로 설정 =====
BASE_DIR = Path("/home/hspark/hspark/unikg/datasets/schemagen/v2")

TRAIN_FILE = BASE_DIR / "final_train.jsonl"
VALID_FILE = BASE_DIR / "final_valid.jsonl"
TEST_FILE  = BASE_DIR / "final_test.jsonl"

PRED_SAVE_DIR = BASE_DIR / "llm_pred"
PRED_SAVE_DIR.mkdir(exist_ok=True)


# ===== JSONL 로드 =====
def load_jsonl(path: Path) -> List[Dict]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    print(f"[LOAD] {path.name}: {len(data)} samples")
    return data


# ===== 라벨 추출 =====
def extract_unique_labels(train, valid, test):
    labels = []
    for dataset in (train, valid, test):
        for obj in dataset:
            lbl = obj.get("labels", [])
            if lbl:
                labels.append(lbl[0])
    return sorted(set(labels))


# ===== 글로벌 few-shot 예시 구성 =====
def build_global_examples(
    train_data: List[Dict],
    top_k_head: int = 10,        # 빈도 상위 라벨 개수
    total_examples: int = 30,    # 최종 예시 총 개수 (예: 10 + 10)
    seed: int = 42,
) -> List[Dict]:
    """
    1) 라벨 빈도 기준 상위 top_k_head 개 라벨에서 각 1개씩 예시 선택
    2) 나머지 라벨들에서 랜덤하게 라벨 몇 개를 뽑아 각 1개씩 예시 선택
       → 최종 예시 개수가 total_examples에 가깝게 되도록 구성
    """
    rng = random.Random(seed)

    # 라벨별로 샘플 모으기
    label_to_items: Dict[str, List[Dict]] = defaultdict(list)
    for obj in train_data:
        labels = obj.get("labels", [])
        if not labels:
            continue
        lab = labels[0]
        label_to_items[lab].append(obj)

    # 라벨 빈도 계산 후 내림차순 정렬
    label_counts = {lab: len(items) for lab, items in label_to_items.items()}
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)

    if not sorted_labels:
        print("[WARN] No labels found in train data. Global examples will be empty.")
        return []

    # 1) 상위 라벨들
    head_labels = [lab for lab, _ in sorted_labels[:top_k_head]]

    # 2) 나머지 tail 라벨들
    tail_labels = [lab for lab, _ in sorted_labels[top_k_head:]]

    examples: List[Dict] = []

    # 상위 라벨들에서 각 1개씩 예시 선택
    for lab in head_labels:
        items = label_to_items.get(lab, [])
        if not items:
            continue
        ex = rng.choice(items)
        examples.append(ex)

    # 남은 예시 수 계산
    remaining = max(0, total_examples - len(examples))

    # tail 라벨에서 라벨 몇 개를 랜덤 샘플링하여 각 1개씩 예시 선택
    if remaining > 0 and tail_labels:
        k = min(remaining, len(tail_labels))
        sampled_tail_labels = rng.sample(tail_labels, k=k)
        for lab in sampled_tail_labels:
            items = label_to_items.get(lab, [])
            if not items:
                continue
            ex = rng.choice(items)
            examples.append(ex)

    print("\n=== Global Few-shot 예시 통계 ===")
    print(f"총 라벨 수: {len(label_to_items)}")
    print(f"상위 {top_k_head}개 라벨에서 뽑은 예시 수: {min(len(examples), top_k_head)}")
    print(f"전체 글로벌 예시 수: {len(examples)} (요청: {total_examples})")

    # 디버깅용: 어떤 라벨들이 예시에 포함됐는지
    ex_label_counts = Counter(ex["labels"][0] for ex in examples if ex.get("labels"))
    print("예시에 포함된 라벨 분포:")
    for lab, cnt in ex_label_counts.most_common():
        print(f"  {lab}: {cnt}")

    return examples


# ===== 프롬프트 템플릿 (글로벌 few-shot 지원) =====
def build_prompt(
    text: str,
    label_candidates: List[str],
    global_examples: List[Dict],
) -> str:
    """
    text: 지금 분류할 인스턴스의 전체 text
    label_candidates: 가능한 라벨 목록 (전체 400개 등)
    global_examples: train에서 뽑은 전역 few-shot 예시 리스트
    """
    labels_str = ", ".join(f'"{lab}"' for lab in label_candidates)

    # --- Few-shot 예시 블록 구성 (global_examples 사용) ---
    example_lines = []
    for ex in global_examples:
        ex_text = ex["text"]
        ex_labels = ex.get("labels", [])
        if not ex_labels:
            continue
        ex_label = ex_labels[0]

        block = (
            "Example:\n"
            "Input text:\n"
            + ex_text
            + "\nOutput:\n"
            + '{"label": "' + ex_label + '"}\n'
        )
        example_lines.append(block)

    examples_block = "\n\n".join(example_lines)

    # --- 전체 프롬프트 ---
    if examples_block:
        few_shot_section = (
            "Below are labeled examples. Follow the same format.\n\n"
            + examples_block
            + "\n\nNow classify the following instance.\n"
        )
    else:
        few_shot_section = "Now classify the following instance.\n"

    prompt = f"""
You are an expert classifier for a knowledge-graph triple tagging task.

Each input contains:
- [HEAD] ... [/HEAD] : the head entity text
- [REL] ... [/REL]   : the relation text
- [TAIL] ... [/TAIL] : the tail entity text
- [FOCUS] ... [/FOCUS] : which component to classify (HEAD or TAIL)

Your task:
1. Read and interpret the semantic meaning of HEAD, REL, and TAIL.
2. Based on the FOCUS tag:
   - FOCUS = "HEAD": classify the type of the HEAD entity.
   - FOCUS = "TAIL": classify the type of the TAIL entity.

Always use the meaning of the entity text and relation context when selecting a type.

Classification rules:
────────────────────────────────────────
• Determine whether the relation (REL) corresponds to a datatype property or an object property by observing the examples and text semantics.

Datatype property cases:
- TAIL is a literal value (integer, double, boolean,  string, date, gYear)
- HEAD must be typed using the object-entity label set (non-literal).
- TAIL must be classified into exactly one of:
  - integer  (integer, nonNegativeInteger, positiveInteger)
  - double   (double, float)
  - boolean  (boolean)
  - string   (textual literals, names, labels, values without explicit meaning)
  - date     (format YYYY-MM-DD)
  - gYear    (format YYYY)

Object property cases:
- Both HEAD and TAIL must be classified using the object-entity label set only.
- Literal-type labels must NOT be used.

Allowed labels:
{labels_str}

Few-shot Examples:
{few_shot_section}

Input:
{text}

Return exactly ONE JSON object:
{{
  "label": "<one of the allowed labels>"
}}

Rules:
- Output only one JSON object.
- No extra text.
- No explanation.
- No multiple labels.
- No list of labels.
""".strip()

    return prompt


# ===== Llama 3.1 Instruct용 system 프롬프트 / chat 템플릿 =====
SYSTEM_PROMPT = (
    "You are an expert classifier for a knowledge-graph triple tagging task. "
    "Follow the user's instructions carefully and always answer with a JSON object."
)


def apply_llama_chat_template(tokenizer, user_prompt: str) -> str:
    """
    Llama 3.1 Instruct가 기대하는 chat 포맷으로 감싸기.
    메시지 구조: system + user, 그리고 assistant 응답을 생성하게 함.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # assistant 역할 시작 지점까지 포함
    )
    return chat_prompt


# ===== vLLM 추론 =====
def run_vllm_inference(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    save_name: str = "llama_3_1_8b_instruct_eval",
    max_new_tokens: int = 128,
    batch_size: int = 16,
    top_k_head: int = 10,         # 상위 라벨 개수
    total_few_shot_examples: int = 20,  # 글로벌 예시 총 개수
):
    """
    vLLM으로 테스트 전체를 추론하고 결과를 저장.

    - 라벨 빈도 기준 상위 top_k_head 개 라벨에서 각 1개씩 예시 추출
    - 나머지 라벨들 중에서 추가로 예시를 뽑아,
      최종 글로벌 few_shot 예시 개수를 total_few_shot_examples 근처로 맞춤
    - meta-llama/Llama-3.1-8B-Instruct 기준 chat 템플릿 적용
    """
    print(f"[INFO] Model = {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device = {device}")

    # 🔹 Llama 3.1 Instruct 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # vLLM 엔진 초기화
    llm = LLM(
        model=model_name,
        dtype="float16",
        gpu_memory_utilization=0.4,
        max_model_len=8192,
        max_num_seqs=batch_size,
        enforce_eager=True,
    )

    # 생성 파라미터 (temperature=0, greedy)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
    )

    # 데이터 로드
    train = load_jsonl(TRAIN_FILE)
    valid = load_jsonl(VALID_FILE)
    test  = load_jsonl(TEST_FILE)

    unique_labels = extract_unique_labels(train, valid, test)
    print("\n=== 라벨 정보 ===")
    print("총 라벨 개수:", len(unique_labels))
    print("라벨 목록 예시 (앞 20개):", unique_labels[:20])

    # train 에서 글로벌 few-shot 예시 구성
    global_examples = build_global_examples(
        train_data=train,
        top_k_head=top_k_head,
        total_examples=total_few_shot_examples,
        seed=42,
    )

    # --- 프롬프트 리스트 구성 (테스트 전체) ---
    prompts = []
    gold_labels = []
    texts = []

    for idx, obj in enumerate(test):
        text = obj["text"]
        gold = obj["labels"][0]

        texts.append(text)
        gold_labels.append(gold)

        # 기존 프롬프트 생성
        user_prompt = build_prompt(text, unique_labels, global_examples)
        # Llama 3.1 chat 템플릿 적용
        final_prompt = apply_llama_chat_template(tokenizer, user_prompt)

        # 디버깅: 첫 번째 프롬프트 한번 찍어보기
        if idx == 0:
            print("\n===== SAMPLE FINAL PROMPT =====")
            print(final_prompt)
            print("================================\n")

        prompts.append(final_prompt)

    print(f"\n=== vLLM 배치 추론 시작 (테스트 {len(prompts)}개) ===")

    pred_labels = []

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)

        for j, out in enumerate(outputs):
            generated = out.outputs[0].text
            gen_stripped = generated.strip()

            # 디버깅: 맨 처음 몇 개만 raw output 출력
            if i == 0 and j < 3:
                print("=== RAW OUTPUT ===")
                print(repr(generated))
                print("==================")

            # JSON만 추려서 label 파싱
            try:
                s = gen_stripped.index("{")
                e = gen_stripped.rindex("}") + 1
                obj = json.loads(gen_stripped[s:e])
                label = (obj.get("label") or "").strip()
                if not label:
                    label = "<EMPTY_LABEL>"
            except Exception as e:
                print("[PARSE ERROR]", e)
                print("RAW STRIPPED:", repr(gen_stripped))
                label = "<PARSE_ERROR>"

            pred_labels.append(label)

    # --- 정확도 및 저장 ---
    assert len(pred_labels) == len(gold_labels)

    total = len(gold_labels)
    correct = 0
    pred_results = []

    for text, gold, pred in zip(texts, gold_labels, pred_labels):
        is_correct = int(pred == gold)
        correct += is_correct
        pred_results.append({
            "text": text,
            "gold": gold,
            "pred": pred,
            "correct": is_correct
        })

    acc = correct / total if total > 0 else 0.0

    print(f"\n=== 평가 결과 ===")
    print(f"총 샘플 수: {total}")
    print(f"정답 수: {correct}")
    print(f"정확도 (accuracy): {acc:.4f}")

    # 저장
    save_path = PRED_SAVE_DIR / f"{save_name}.jsonl"
    with save_path.open("w", encoding="utf-8") as f:
        for item in pred_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[SAVE] Predictions saved to: {save_path}")


if __name__ == "__main__":
    # 기본 실행 예시
    run_vllm_inference(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        save_name="llama_vllm_fewshot_v3_30.jsonl",
        top_k_head=10,           
        total_few_shot_examples=30  
    )

