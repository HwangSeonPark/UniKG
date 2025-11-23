# COMBO 데이터셋

A Complete Open KG Canonicalization Benchmark

## 개요

COMBO는 지식그래프 정규화(Knowledge Graph Canonicalization)를 위한 종합 벤치마크 데이터셋입니다. 같은 원본 데이터를 세 가지 다른 방식으로 정규화하여, 다양한 정규화 task를 평가할 수 있습니다.

**출처**: COMBO - A Complete Open KG Canonicalization Benchmark
**처리 날짜**: 2025-11-14
**총 데이터**: valid 3,465개 / test 14,023개

---

## 세 가지 정규화 Task

### 1. NPC-E (Noun Phrase Canonicalization - Entity Level)

**목적**: 같은 개체를 지칭하는 다양한 명사구 표현을 표준 엔티티 ID로 정규화

**정규화 방식**: 엔티티 이름 → 엔티티 ID

**예시**:
```
원본 문장:
"Jesse Carmichael contributed to the keyboard instrument"

Test 트리플:
[jesse carmichael, contributed to, keyboard instrument]

Golden 트리플:
[Q459375, instrument, Q52954]
  ↑ Wikidata ID     ↑ 표준 관계     ↑ Wikidata ID
```

**데이터 크기**:
- Valid: 2,768개
- Test: 11,176개
- 총: 13,944개

**적용 조건**: r.id가 존재하는 경우 (관계가 이미 정규화된 경우)

---

### 2. RPC (Relation Phrase Canonicalization)

**목적**: 같은 의미의 다양한 관계 표현을 표준 관계명으로 정규화

**정규화 방식**: 관계 표현 → 표준 관계명 (엔티티는 이름 유지)

**예시**:
```
원본 문장:
"Steven Doane is a professor of cello"

Test 트리플:
[steven doane, professor of, cello]

Golden 트리플:
[steven doane, instrument, cello]
  ↑ 이름 유지       ↑ 표준 관계     ↑ 이름 유지
```

**관계 표현 정규화 예시**:
- "professor of" → "instrument"
- "performed on" → "instrument"
- "plays" → "instrument"
- "was born in" → "birth place"

**데이터 크기**:
- Valid: 697개
- Test: 2,847개
- 총: 3,544개

**적용 조건**: r.id가 None인 경우 (관계가 정규화되지 않은 경우)

---

### 3. NPC-O (Noun Phrase Canonicalization - Ontology Level)

**목적**: 같은 타입(클래스)에 속하는 엔티티들을 타입 ID로 정규화

**정규화 방식**: 엔티티 이름 → 엔티티 타입 (Wikidata instance)

**예시**:
```
원본 문장:
"Jesse Carmichael contributed to the keyboard instrument"

Test 트리플:
[jesse carmichael, contributed to, keyboard instrument]

Golden 트리플:
[Q5, instrument, Q1254773]
 ↑ Human    ↑ 표준 관계    ↑ Musical Instrument
```

**타입 정규화 예시**:
- "Toronto" = "New York" → Q515 (City)
- "Jesse Carmichael" = "Steven Doane" → Q5 (Human)
- "cello" = "drums" → Q34379 (Musical Instrument)

**데이터 크기**:
- Valid: 3,465개 (전체 데이터)
- Test: 14,023개 (전체 데이터)
- 총: 17,488개

**적용 조건**: 모든 데이터 (전체 데이터셋을 타입으로 정규화)

---

## 데이터 구조

### Test 데이터 (JSONL 형식)

**파일 위치**: `test/COMBO/{TASK}@{SPLIT}.jsonl`

**형식**:
```json
{
  "original": "원본 문장 전체",
  "tris": [
    ["head_entity", "relation_phrase", "tail_entity"]
  ]
}
```

**예시**:
```json
{
  "original": "cozy powell , roger taylor and rowan atkinson performed on drum kit .",
  "tris": [
    ["cozy powell", "performed on", "drum kit"]
  ]
}
```

**특징**:
- 원본 문장에서 추출된 비정형 트리플
- 같은 Test 데이터가 세 가지 task에 모두 사용됨
- Task별로 정규화 목표만 다름

---

### Golden 데이터 (TXT 형식)

**파일 위치**: `golden/COMBO/{TASK}@{SPLIT}.txt`

**형식**: `[head, relation, tail]` (한 줄에 하나씩)

#### NPC-E Golden
```
[Q459375, instrument, Q52954]
[Q2234394, instrument, Q46185]
[Q493730, instrument, Q128309]
```
- Head: Wikidata 엔티티 ID
- Relation: 표준 관계명
- Tail: Wikidata 엔티티 ID

#### RPC Golden
```
[steven doane, instrument, cello]
[gerhardt thomas fuchs, instrument, drums]
[chris vrenna, instrument, drum kit]
```
- Head: 엔티티 이름 (원본 유지)
- Relation: 표준 관계명 (정규화됨)
- Tail: 엔티티 이름 (원본 유지)

#### NPC-O Golden
```
[Q5, instrument, Q8371]
[Q5, instrument, Q1254773]
[Q5, instrument, Q46185]
```
- Head: Wikidata 타입 ID (instance)
- Relation: 표준 관계명
- Tail: Wikidata 타입 ID (instance)

---

## 파일 구조

```
datasets/
├── golden/COMBO/
│   ├── NPC-E@valid.txt    (2,768줄)
│   ├── NPC-E@test.txt     (11,176줄)
│   ├── RPC@valid.txt      (697줄)
│   ├── RPC@test.txt       (2,847줄)
│   ├── NPC-O@valid.txt    (3,465줄)
│   └── NPC-O@test.txt     (14,023줄)
│
└── test/COMBO/
    ├── NPC-E@valid.jsonl  (2,768줄)
    ├── NPC-E@test.jsonl   (11,176줄)
    ├── RPC@valid.jsonl    (697줄)
    ├── RPC@test.jsonl     (2,847줄)
    ├── NPC-O@valid.jsonl  (3,465줄)
    └── NPC-O@test.jsonl   (14,023줄)
```

---

## 통계 정보

### 전체 통계

| Task | Valid | Test | 총합 |
|------|-------|------|------|
| **NPC-E** | 2,768 | 11,176 | 13,944 |
| **RPC** | 697 | 2,847 | 3,544 |
| **NPC-O** | 3,465 | 14,023 | 17,488 |

### Task별 특징

#### NPC-E
- **대상**: 엔티티가 이미 정규화 가능한 데이터
- **조건**: r.id != None
- **비율**: 전체의 79.9% (valid 기준)

#### RPC
- **대상**: 관계가 정규화되지 않은 데이터
- **조건**: r.id == None
- **비율**: 전체의 20.1% (valid 기준)
- **고유 표준 관계**: 79개

#### NPC-O
- **대상**: 모든 데이터
- **조건**: 전체 데이터셋
- **비율**: 100%
- **고유 타입**: 2,946개

---

## 사용 예시

### Python으로 데이터 읽기

```python
import json

# Test 데이터 읽기
def load_test_data(task, split):
    path = f"test/COMBO/{task}@{split}.jsonl"
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Golden 데이터 읽기
def load_golden_data(task, split):
    path = f"golden/COMBO/{task}@{split}.txt"
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(line.strip())
    return data

# 사용 예시
test_data = load_test_data('NPC-E', 'valid')
golden_data = load_golden_data('NPC-E', 'valid')

print(f"Test: {test_data[0]}")
print(f"Golden: {golden_data[0]}")
```

### 출력 예시
```
Test: {'original': 'the drums were provided by...', 'tris': [['jesse carmichael', 'contributed to', 'keyboard instrument']]}
Golden: [Q459375, instrument, Q52954]
```

---

## 데이터 처리 코드

데이터 생성 코드: `Data_pre/combo.py`

```python
from combo import Combo

# 데이터 처리
combo = Combo(
    root="/path/to/COMBO",
    out="/path/to/datasets"
)
combo.run()
combo.check()
```

---

## 핵심 개념 정리

### 1. 같은 원본, 다른 정규화

**중요**: 세 가지 task는 **같은 Test 데이터**를 사용하지만, **Golden 데이터가 다릅니다**.

```
원본 문장: "Cozy Powell performed on drum kit"
Test 트리플: [cozy powell, performed on, drum kit]

→ NPC-E Golden: [Q14341, instrument, Q128309]
→ RPC Golden:   [cozy powell, instrument, drum kit]
→ NPC-O Golden: [Q5, instrument, Q128309]
```

### 2. 정규화 단계

```
비정형 트리플
    ↓
┌───┴────────────────────────────┐
│                                │
↓ NPC-E                   ↓ RPC │
엔티티 ID로 정규화        관계 정규화 │
                                │
                        ↓ NPC-O │
                      타입으로 정규화 │
                                │
└────────────────────────────────┘
```

### 3. 적용 시나리오

- **NPC-E**: 엔티티 링킹, 개체명 정규화
- **RPC**: 관계 추출, 의미 정규화
- **NPC-O**: 온톨로지 매핑, 타입 분류

---

## 주의사항

1. **Test 데이터 중복**
   - NPC-E, RPC, NPC-O의 Test 데이터는 내용상 중복될 수 있음
   - NPC-E와 NPC-O는 모두 전체 데이터를 포함하지만 정규화 방식이 다름

2. **데이터 개수 관계**
   - NPC-E + RPC ≠ NPC-O
   - NPC-O는 전체 데이터셋
   - NPC-E는 r.id가 있는 경우만
   - RPC는 r.id가 None인 경우만

3. **Instance 처리**
   - instance 필드는 리스트이지만, Golden에는 첫 번째 값만 사용
   - 예: `['Q5', 'Q215627']` → `Q5`

---

