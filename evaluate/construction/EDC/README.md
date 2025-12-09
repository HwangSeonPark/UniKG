# EDC 메트릭 모듈

입력 포맷(라인별 파이썬 리스트 문자열)과 XML 변환 → nervaluate → Exact Triple 매크로 계산 순서를 원본과 동일하게 유지한다.

## 지원 지표

|그룹|세부 항목|
|---|---|
|nervaluate (SUB/PRED/OBJ)|Partial / Strict / Exact / Entity Type Precision·Recall·F1|
|Exact Triple Macro|Pred/Gold 세트를 전체 라벨 공간으로 이진화한 매크로 Precision·Recall·F1|

- nervaluate 실행 중 생성되는 XML은 `result_xmls/{dataset}` 폴더에 저장된다.

---

## 빈 문자열 처리

### 문제점

원래 코드에서는 빈 문자열(`''`)이 하나의 클래스로 취급되어, 여러 빈 문자열이 서로 매칭되어 점수가 부풀려지는 문제가 있었는다.

**예시:**
- 정답: `[['apple', 'founded by', 'steve jobs']]`
- 예측: `[['apple', 'founded by', '']]`

원래 코드에서는 빈 문자열이 무시되어 Precision이 1.0000(만점)으로 나왔는다.

### 해결 방법

각 빈 문자열을 고유한 오답 토큰으로 치환하여, 각각이 독립적인 오답으로 처리되도록 수정했는다.

### 코드 구현

**`webnlg/common.py`의 `getRefs` 및 `getCands` 함수:**

```python
# 빈 문자열을 고유 오답 토큰으로 치환
eidx = 0  # 빈 문자열을 고유 토큰으로 치환하기 위한 인덱스
for entry in allreftriples:
    newtriples = []
    for triple in entry:
        # ... 정규화 과정 ...
        adjusttriple = newtriple.split(" | ")
        
        # 빈 문자열 오답 처리: 각 요소가 빈칸이면 고유 토큰으로 치환
        for i in range(len(adjusttriple)):
            if not adjusttriple[i].strip():
                adjusttriple[i] = f'<emp_{eidx}>'
                eidx += 1
        
        newtriple = " | ".join(adjusttriple)
        newtriples.append(newtriple)
```

**처리 예시:**

```
입력:  ['apple', 'founded by', '']
       ['google', 'ceo', '']
       ['microsoft', 'founder', '']

처리 후:
       ['apple', 'founded by', '<emp_0>']
       ['google', 'ceo', '<emp_1>']
       ['microsoft', 'founder', '<emp_2>']
```

각 빈 문자열이 `<emp_0>`, `<emp_1>`, `<emp_2>`로 치환되어 서로 다른 클래스로 취급되므로, 정답과 절대 매칭되지 않다.

\

1. **고유 토큰은 정답과 절대 매칭되지 않음**
   ```
   정답: ['apple', 'founded by', 'steve jobs']
   예측: ['apple', 'founded by', '<emp_0>']
   
   평가 시:
   - 'apple' == 'apple' → ✓
   - 'founded by' == 'founded by' → ✓
   - 'steve jobs' == '<emp_0>' → ✗ (절대 같을 수 없음!)
   ```
   `<emp_0>`는 특수 토큰이므로 정답의 실제 값(`'steve jobs'`)과 절대 같을 수 없다. 따라서 빈 문자열이 있는 요소는 항상 오답으로 처리됩니다.

2. **각 고유 토큰은 서로 다른 클래스**
   ```
   예측1: ['apple', 'founded by', '<emp_0>']
   예측2: ['google', 'ceo', '<emp_1>']
   예측3: ['microsoft', 'founder', '<emp_2>']
   ```
   `<emp_0>`, `<emp_1>`, `<emp_2>`는 서로 다른 클래스이므로:
   - 정답과 매칭되지 않음 (오답 처리)
   - 빈 문자열끼리도 서로 매칭되지 않음 (점수 부풀림 방지)

3. **실제 평가 과정**
   ```
   입력: ['apple', 'founded by', '']
   
   처리:
   1. triple 분리: ['apple', 'founded by', '']
   2. 빈 문자열 발견 → '<emp_0>'로 치환
   3. 최종: ['apple', 'founded by', '<emp_0>']
   
   평가:
   - 'apple' == 'apple' → ✓
   - 'founded by' == 'founded by' → ✓
   - 'steve jobs' == '<emp_0>' → ✗
   
   점수: 2/3 = 0.6667
   ```

**결론:**
- 빈 문자열(`''`)을 고유 토큰(`<emp_0>`, `<emp_1>`, ...)으로 치환
- 고유 토큰은 정답과 절대 매칭되지 않음 → **오답 처리**
- 각 고유 토큰은 서로 다른 클래스 → 빈 문자열끼리 매칭 안 됨
- 결과: 빈 문자열이 올바르게 오답으로 처리되어 점수가 부풀려지지 않음

---

## 평가 방식별 점수 계산 방법

### 1. Exact Triple

**방식:** Position-based evaluation (위치별 정확도 계산)

**계산 방법:**
- 각 triple의 Subject, Predicate, Object를 위치별로 정확히 비교
- 일치한 요소 수 / 전체 요소 수

**코드:**

```python
def calculateExactTripleScore(reflist, candlist):
    total = 0
    match = 0
    
    # 각 entry 처리
    for ref_entry, cand_entry in zip(reflist, candlist):
        # 각 entry의 triple 처리
        for ref_trip_str, cand_trip_str in zip(ref_entry, cand_entry):
            # triple 문자열을 요소로 분리
            ref_parts = [p.strip() for p in ref_trip_str.split(" | ")]
            cand_parts = [p.strip() for p in cand_trip_str.split(" | ")]
            
            # 요소별 비교
            for ref_elem, cand_elem in zip(ref_parts, cand_parts):
                total += 1
                if ref_elem == cand_elem:
                    match += 1
    
    # Precision = Recall = 일치한 요소 수 / 전체 요소 수
    prec = match / total
    rec = match / total
    f1 = 2 * prec * rec / (prec + rec)
    
    return prec, rec, f1
```

**특징:**
- 가장 단순하고 직관적인 평가 방식
- 각 요소가 정확히 일치해야만 점수 획득
- 부분 일치를 인정하지 않음

---

### 2. Entity Type

**방식:** NER 기반 entity type 평가

**Entity Type이란?**
- Entity Type은 **triple의 구조적 위치**를 나타내는 태그입니다
- 실제 NER의 entity type이 아니라, triple의 위치에 따라 자동으로 할당됩니다:
  - 첫 번째 요소 (Subject) → `SUB`
  - 두 번째 요소 (Predicate) → `PRED`
  - 세 번째 요소 (Object) → `OBJ`

**Entity Type 추출 방법:**
```python
# 코드에서 triple의 인덱스에 따라 자동 할당
for idx, attrib in enumerate(indextriple):
    if idx == 0:
        getrefdict(..., "SUB", "SUB", ...)  # Subject
    elif idx == 1:
        getrefdict(..., "PRED", "PRED", ...)  # Predicate
    else:
        getrefdict(..., "OBJ", "OBJ", ...)  # Object
```

**계산 방법:**
- `nervaluate` 라이브러리의 `ent_type` 메트릭 사용
- Entity의 타입(SUB, PRED, OBJ)만 일치하면 점수 획득
- 실제 값은 다르더라도 타입이 맞으면 인정

**예시:**
- 정답: `['apple', 'founded by', 'steve jobs']`
  - Entity Type: SUB='apple', PRED='founded by', OBJ='steve jobs'
- 예측: `['apple company', 'founded by', 'steve']`
  - Entity Type: SUB='apple company', PRED='founded by', OBJ='steve'
- 결과: 
  - SUB 위치에 값이 있음 → 타입 일치 ✓
  - PRED 위치에 값이 있음 → 타입 일치 ✓
  - OBJ 위치에 값이 있음 → 타입 일치 ✓
  - **Entity Type F1 = 1.0000** (모든 타입이 일치)

**특징:**
- 가장 관대한 평가 방식
- 위치(태그)만 맞으면 점수 획득
- 실제 값이 다르더라도 타입이 맞으면 인정

---

### 3. Exact

**방식:** NER 기반 exact match 평가

**계산 방법:**
- `nervaluate` 라이브러리의 `exact` 메트릭 사용
- 각 요소(SUB, PRED, OBJ)가 정확히 일치해야 점수 획득
- 단어 단위로 정확히 매칭되어야 함

**예시:**
- 정답: `['apple', 'founded by', 'steve jobs']`
- 예측: `['apple company', 'founded by', 'steve']`
- 결과: 정확히 일치하는 요소가 적어 낮은 점수

**특징:**
- 단어 단위 정확한 일치를 요구
- 부분 일치를 인정하지 않음

---

### 4. Partial

**방식:** NER 기반 partial match 평가

**계산 방법:**
- `nervaluate` 라이브러리의 `partial` 메트릭 사용
- **겹치는 토큰(단어)의 비율**을 계산
- 공통 단어가 있으면 그 비율만큼 부분 점수 획득

**부분 일치 기준:**
- 각 요소(SUB, PRED, OBJ)를 단어 단위로 분리하여 비교
- 공통 단어가 있으면 겹치는 단어 수 / 전체 단어 수로 점수 계산
- 예: `'apple'` vs `'apple company'` → 공통 단어 `'apple'` 1개 / 전체 2개 = 0.5점

**상세 예시:**

| 정답 | 예측 | 공통 단어 | Partial 점수 | 설명 |
|------|------|----------|--------------|------|
| `'apple'` | `'apple company'` | `'apple'` | 0.5 | 1개 공통 / 2개 전체 |
| `'steve jobs'` | `'steve'` | `'steve'` | 0.5 | 1개 공통 / 2개 전체 |
| `'apple'` | `'apple'` | `'apple'` | 1.0 | 완전 일치 |
| `'apple'` | `'google'` | 없음 | 0.0 | 공통 단어 없음 |

**전체 예시:**
- 정답: `['apple', 'founded by', 'steve jobs']`
- 예측: `['apple company', 'founded by', 'steve']`
- 계산:
  - SUB: `'apple'` vs `'apple company'` → 공통 `'apple'` → 0.5점
  - PRED: `'founded by'` vs `'founded by'` → 완전 일치 → 1.0점
  - OBJ: `'steve jobs'` vs `'steve'` → 공통 `'steve'` → 0.5점
- 결과: 평균 0.6667점

**특징:**
- 부분 일치를 인정하는 관대한 평가
- 공통 단어가 있으면 그 비율만큼 점수 획득
- Exact보다 항상 같거나 높은 점수 (부분 일치 인정)

---

### 5. Strict

**방식:** NER 기반 strict match 평가

**계산 방법:**
- `nervaluate` 라이브러리의 `strict` 메트릭 사용
- 모든 요소가 정확히 일치하고, 위치도 정확해야 점수 획득
- 가장 엄격한 평가 기준

**예시:**
- 정답: `['apple', 'founded by', 'steve jobs']`
- 예측: `['apple company', 'founded by', 'steve']`
- 결과: 정확히 일치하지 않아 낮은 점수

**특징:**
- 가장 엄격한 평가 방식
- 완벽한 일치만 인정

---

## 평가 방식 비교

| 평가 방식 | 엄격도 | 부분 일치 | 특징 |
|----------|--------|----------|------|
| **Exact Triple** | 중간 | ❌ | 위치별 정확한 일치 |
| **Entity Type** | 낮음 | ✅ | Entity type만 맞으면 인정 |
| **Exact** | 높음 | ❌ | 단어 단위 정확한 일치 |
| **Partial** | 낮음 | ✅ | 부분 문자열 일치 인정 |
| **Strict** | 매우 높음 | ❌ | 완벽한 일치만 인정 |

---

## 예시 데이터 및 결과

### 예시 1: 완벽히 일치

**입력:**
```
정답: [['apple', 'founded by', 'steve jobs']]
예측: [['apple', 'founded by', 'steve jobs']]
```

**결과:**
| 평가 방식 | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| Exact Triple | 1.0000 | 1.0000 | 1.0000 |
| Entity Type | 1.0000 | 1.0000 | 1.0000 |
| Exact | 1.0000 | 1.0000 | 1.0000 |
| Partial | 1.0000 | 1.0000 | 1.0000 |
| Strict | 1.0000 | 1.0000 | 1.0000 |

---

### 예시 2: Object만 빈 문자열

**입력:**
```
정답: [['apple', 'founded by', 'steve jobs']]
예측: [['apple', 'founded by', '']]
```

**처리 과정:**
```
예측 처리: ['apple', 'founded by', ''] 
        → ['apple', 'founded by', '<emp_0>']
```

**결과:**
| 평가 방식 | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| Exact Triple | 0.6667 | 0.6667 | 0.6667 |
| Entity Type | 0.6667 | 0.6667 | 0.6667 |
| Exact | 0.6667 | 0.6667 | 0.6667 |
| Partial | 0.6667 | 0.6667 | 0.6667 |
| Strict | 0.6667 | 0.6667 | 0.6667 |

**설명:** 
- Subject와 Predicate는 완벽히 일치하고, Object만 빈 문자열(오답)
- **모든 평가 방식이 0.6667로 같은 이유:**
  - **Exact Triple**: 위치별로 비교 → SUB ✓, PRED ✓, OBJ ✗ → 2/3 = 0.6667
  - **Entity Type**: Entity type 비교 → SUB ✓, PRED ✓, OBJ ✗ → 2/3 = 0.6667
  - **Exact**: 정확한 일치 비교 → SUB ✓, PRED ✓, OBJ ✗ → 2/3 = 0.6667
  - **Partial**: 부분 일치 비교 → SUB ✓, PRED ✓, OBJ ✗ → 2/3 = 0.6667
  - **Strict**: 엄격한 일치 비교 → SUB ✓, PRED ✓, OBJ ✗ → 2/3 = 0.6667
  
  **결론:** 모든 평가 방식이 "2개는 완벽히 맞고 1개는 완전히 틀림"이라는 동일한 상황을 평가하므로 점수가 같는다.

---

### 예시 3: Object만 다른 값

**입력:**
```
정답: [['apple', 'founded by', 'steve jobs']]
예측: [['apple', 'founded by', 'bill gates']]
```

**결과:**
| 평가 방식 | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| Exact Triple | 0.6667 | 0.6667 | 0.6667 |
| Entity Type | 0.6667 | 0.6667 | 0.6667 |
| Exact | 0.6667 | 0.6667 | 0.6667 |
| Partial | 0.6667 | 0.6667 | 0.6667 |
| Strict | 0.6667 | 0.6667 | 0.6667 |

**설명:**
- Subject와 Predicate는 완벽히 일치하고, Object만 다른 값(오답)
- **모든 평가 방식이 0.6667로 같은 이유:** 예시 2와 동일하게 "2개는 완벽히 맞고 1개는 완전히 틀림" 상황
- 빈 문자열(`''`)과 다른 값(`'bill gates'`) 모두 동일하게 오답으로 처리됨
- 이는 빈 문자열이 올바르게 오답으로 처리되고 있음을 의미합니다

---

### 예시 4: 부분 일치

**입력:**
```
정답: [['apple', 'founded by', 'steve jobs']]
예측: [['apple company', 'founded by', 'steve']]
```

**결과:**
| 평가 방식 | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| Exact Triple | 0.3333 | 0.3333 | 0.3333 |
| Entity Type | 1.0000 | 1.0000 | 1.0000 |
| Exact | 0.3333 | 0.3333 | 0.3333 |
| Partial | 0.6667 | 0.6667 | 0.6667 |
| Strict | 0.3333 | 0.3333 | 0.3333 |

**설명:**
- **Exact Triple**: 위치별 정확한 일치만 비교 → SUB ✗, PRED ✓, OBJ ✗ → 1/3 = 0.3333
- **Entity Type**: Entity type만 비교 (값은 무시) → SUB ✓, PRED ✓, OBJ ✓ → 1.0000
- **Exact**: 단어 단위 정확한 일치 비교 → SUB ✗, PRED ✓, OBJ ✗ → 0.3333
- **Partial**: 부분 문자열 일치 인정 → SUB 부분일치, PRED ✓, OBJ 부분일치 → 0.6667
- **Strict**: 완벽한 일치만 인정 → SUB ✗, PRED ✓, OBJ ✗ → 0.3333

**차이가 나는 이유:**
- 예시 2, 3과 달리 **부분 일치**가 발생하는 경우입니다
- 'apple' vs 'apple company': 부분 일치
- 'steve jobs' vs 'steve': 부분 일치
- **Entity Type**은 값이 다르더라도 타입만 맞으면 인정하므로 1.0000
- **Partial**은 부분 일치를 인정하므로 0.6667
- 나머지는 정확한 일치만 인정하므로 0.3333

---

### 예시 5: 여러 트리플 혼합

**입력:**
```
정답: 
  [['apple', 'founded by', 'steve jobs']]
  [['google', 'ceo', 'sundar pichai']]
  [['microsoft', 'founder', 'bill gates']]

예측:
  [['apple', 'founded by', 'steve jobs']]  # 완벽히 일치
  [['google', 'ceo', '']]                   # Object 빈칸
  [['microsoft', 'created by', 'bill gates']]  # Predicate 다름
```

**처리 과정:**
```
예측 처리:
  ['google', 'ceo', ''] → ['google', 'ceo', '<emp_0>']
```

**결과:**
| 평가 방식 | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| Exact Triple | 0.7778 | 0.7778 | 0.7778 |
| Entity Type | 0.7778 | 0.7778 | 0.7778 |
| Exact | 0.7778 | 0.7778 | 0.7778 |
| Partial | 0.7778 | 0.7778 | 0.7778 |
| Strict | 0.7778 | 0.7778 | 0.7778 |

**설명:** 
- 9개 요소 중 7개 일치 (apple 3개 + google 2개 + microsoft 2개)
- 7/9 = 0.7778점

---

### 예시 6: 여러 빈 문자열

**입력:**
```
정답:
  [['apple', 'founded by', 'steve jobs']]
  [['google', 'ceo', 'sundar pichai']]
  [['microsoft', 'founder', 'bill gates']]

예측:
  [['apple', 'founded by', '']]
  [['google', 'ceo', '']]
  [['microsoft', 'founder', '']]
```

**처리 과정:**
```
예측 처리:
  ['apple', 'founded by', ''] → ['apple', 'founded by', '<emp_0>']
  ['google', 'ceo', ''] → ['google', 'ceo', '<emp_1>']
  ['microsoft', 'founder', ''] → ['microsoft', 'founder', '<emp_2>']
```

**결과:**
| 평가 방식 | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| Exact Triple | 0.6667 | 0.6667 | 0.6667 |
| Entity Type | 0.6667 | 0.6667 | 0.6667 |
| Exact | 0.6667 | 0.6667 | 0.6667 |
| Partial | 0.6667 | 0.6667 | 0.6667 |
| Strict | 0.6667 | 0.6667 | 0.6667 |

**설명:**
- 9개 요소 중 6개 일치 (각 triple의 Subject와 Predicate)
- 6/9 = 0.6667점
- 각 빈 문자열이 독립적인 오답으로 처리되어 서로 매칭되지 않음

---

## 평가 방식별 점수 차이 요약

### 점수가 같은 경우

**예시 2, 3, 6과 같이 모든 평가 방식이 같은 점수를 내는 경우:**

- **상황:** 일부 요소는 완벽히 일치하고, 일부 요소는 완전히 틀림
- **예시:** `['apple', 'founded by', 'steve jobs']` vs `['apple', 'founded by', '']`
  - SUB: 완벽히 일치 ✓
  - PRED: 완벽히 일치 ✓
  - OBJ: 완전히 틀림 ✗
  
- **결과:** 모든 평가 방식이 2/3 = 0.6667점
  - **Exact Triple**: 위치별 비교 → 2개 맞음
  - **Entity Type**: 타입 비교 → 2개 맞음
  - **Exact**: 정확한 일치 비교 → 2개 맞음
  - **Partial**: 부분 일치 비교 → 2개 맞음 (완벽히 일치도 부분 일치로 인정)
  - **Strict**: 엄격한 일치 비교 → 2개 맞음

**이유:** 모든 평가 방식이 "완벽히 맞음"과 "완전히 틀림"을 구분할 수 있으므로, 동일한 상황에서는 같은 점수가 나옵니다.

### 점수가 다른 경우

**예시 4와 같이 평가 방식별로 다른 점수가 나오는 경우:**

- **상황:** 부분 일치가 발생하는 경우
- **예시:** `['apple', 'founded by', 'steve jobs']` vs `['apple company', 'founded by', 'steve']`
  - SUB: 부분 일치 ('apple' vs 'apple company')
  - PRED: 완벽히 일치 ✓
  - OBJ: 부분 일치 ('steve jobs' vs 'steve')
  
- **결과:** 평가 방식별로 다른 점수
  - **Exact Triple**: 0.3333 - 정확한 일치만 인정 (PRED만 맞음)
  - **Entity Type**: 1.0000 - 타입만 맞으면 인정 (모든 타입 일치)
  - **Exact**: 0.3333 - 정확한 일치만 인정 (PRED만 맞음)
  - **Partial**: 0.6667 - 부분 일치 인정 (SUB, PRED, OBJ 모두 부분/완전 일치)
  - **Strict**: 0.3333 - 완벽한 일치만 인정 (PRED만 맞음)

**이유:** 각 평가 방식이 "부분 일치"를 다르게 처리하기 때문입니다.

---

## 주요 개선사항

1. **빈 문자열 오답 처리**
   - 각 빈 문자열을 고유 토큰(`<emp_0>`, `<emp_1>`, ...)으로 치환
   - 여러 빈칸끼리 서로 매칭되어 점수가 부풀려지는 문제 해결

2. **정확한 점수 계산**
   - Position-based evaluation으로 각 요소를 정확히 비교
   - 3개 중 2개 맞히면 정확히 2/3 = 0.6667점

3. **다양한 평가 방식 지원**
   - 각 평가 방식이 서로 다른 기준으로 점수 계산
   - 부분 일치, 엄격한 평가 등 다양한 시나리오 지원
