# Tree-KG 메트릭 모듈

Tree-KG의 ER/PC/F1/MEC/RS 평가 메트릭 구현

## 지원 지표

|지표|설명|범위|
|---|---|---|
|Entity Recall (ER)|GT 엔티티가 예측 그래프에서 재현된 비율|[0, 1]|
|Precision (PC)|예측 엔티티가 GT에서 정확한 비율|[0, 1]|
|F1 Score (F1)|ER과 PC의 조화 평균|[0, 1]|
|Mapping-based Edge Connectivity (MEC)|GT 엣지가 예측 그래프에서 연결된 비율|[0, 1]|
|Relation Strength (RS)|예측 관계의 강도 (LLM 평가)|[0, 10]|

## 코드 구조

- `common.py`: 공통 함수 (임베딩, 매핑, BFS)
- `er.py`: Entity Recall 계산
- `pc.py`: Precision 계산
- `f1.py`: F1 Score 계산
- `mec.py`: Mapping-based Edge Connectivity 계산
- `rs.py`: Relation Strength 계산
- `eval_all.py`: 통합 평가 (권장)

## 사용법

### 1. 통합 평가 (권장)

```python
from evaluate.construction.Tree-KG.eval_all import eval

gt = [["entity1", "relation", "entity2"], ...]
pd = [["entity_a", "relation", "entity_b"], ...]

# 자동으로 임베딩 최적화
results = eval(gt, pd, key=api_key)
print(results)
# {'er': 0.85, 'pc': 0.90, 'f1': 0.87, 'mec': 0.80, 'rs': 7.5}
```

### 2. 개별 메트릭 평가

```python
from evaluate.construction.Tree-KG.er import er
from evaluate.construction.Tree-KG.pc import pc
from evaluate.construction.Tree-KG.mec import mec

er_score = er(gt, pd, key=api_key)
pc_score = pc(gt, pd, key=api_key)
mec_score = mec(gt, pd, key=api_key)
```

### 3. 배치 평가 (효율적)

```python
from evaluate.construction.common.preload import prep_batch
from evaluate.construction.Tree-KG.eval_all import eval

gt_list = [gt1, gt2, gt3, ...]
pd_list = [pd1, pd2, pd3, ...]

# 모든 엔티티를 한번에 임베딩 (효율성 극대화)
prep_batch(gt_list, pd_list)

# 각 샘플 평가 (캐시 사용)
for gt, pd in zip(gt_list, pd_list):
    result = eval(gt, pd, prep=False)
```

## 임베딩 시스템

통합 임베딩 시스템을 사용하여 중복 계산 방지:
- **글로벌 캐시**: GraphJudge와 Tree-KG가 캐시 공유
- **배치 처리**: 여러 엔티티를 한번에 임베딩
- **자동 GPU 사용**: CUDA 사용 가능하면 자동 활성화

GraphJudge도 bert-base-uncased를 사용하므로, 
**두 메트릭을 함께 평가할 때 임베딩을 한번만 수행**합니다.

자세한 내용: `../common/EMBEDDING_USAGE.md` 참고

## API 키 설정

Gemini API 키 (LLM 매핑용):

```bash
export GEMINI_API_KEY="your-api-key"
```

또는 `.env` 파일:

```
GEMINI_API_KEY=your-api-key
```

> RS는 API 키가 없으면 계산 불가 (None 반환)

## 논문 메트릭과의 일치

논문에 명시된 메트릭 정의를 정확히 구현:

1. **ER**: 엔티티 임베딩 → Top-5 검색 → LLM 매핑
2. **PC**: 동일한 프로세스 (역방향)
3. **F1**: 2 × ER×PC / (ER+PC)
4. **MEC**: 매핑된 엔티티 쌍의 경로 연결 확인
5. **RS**: LLM 기반 관계 강도 평가 (0-10)
