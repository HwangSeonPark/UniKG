# SAC-KG 메트릭 모듈

SAC-KG는 원문 텍스트와 예측 트리플을 함께 사용해 LLM으로 정밀도를 검증하는 평가 절차다.  
코드는 `baseline/construction`의 SAC-KG 평가 스크립트를 기반으로 하며, LLM 호출 인터페이스(_llm) 역시 동일하게 사용한다.

## 지원 지표
|지표|설명|
|---|---|
|Precision|텍스트에서 LLM이 "정확"으로 판단한 트리플 비율|
|Number of Recalls|텍스트당 평균으로 검증된(정확 판정된) 트리플 수|

## 실행 방법
```
python evaluate/construction/SAC-KG/main.py \
  --dataset /home/hyyang/my_workspace/KGC/evaluate/construction/references \
  --api-key $OPENROUTER_API_KEY
```

- `pred.txt`와 `sen.txt`의 라인 수가 동일해야 하며, 각 라인은 파이썬 리스트 문자열 형식이다.
- `--text` 옵션으로 텍스트 파일을 직접 지정할 수 있다.
- OpenRouter API 키가 없으면 실행이 중단된다.

## 코드 구조
- `metrics/llm_precision.py` : 텍스트 기반 LLM 질의, 판단 로깅, 지표 계산
- `runner.py` : 입력 검증과 스코어 집계
- `main.py` : CLI 파서 및 결과 출력

원본 논문의 메시지 템플릿/판단 기준을 그대로 유지했으며, 테스트용 더미 데이터나 축약 로직은 포함하지 않는다.


