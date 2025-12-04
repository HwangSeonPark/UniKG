# Tree-KG 메트릭 모듈

Tree-KG의 ER/PC/MEC/RS 흐름의 평가 스크립트다.  

## 지원 지표
|지표|설명|
|---|---|
|Entity Recall (ER)|GT 엔티티를 예측 그래프 Top-5 후보와 LLM 매핑 후 재현된 비율|
|Precision (PC)|예측 엔티티 기준으로 GT 후보를 찾은 뒤 LLM 매핑 합격률|
|Mapping-based Edge Connectivity (MEC)|ER/PC 매핑 테이블을 이용해 GT 엣지가 예측 그래프에서 경로로 연결되는 비율|
|Relation Strength (RS)| 예측 트리플을 LLM에게 평가시켜 0~10 점수 평균을 산출|

> RS는 OpenRouter API 키가 없으면 계산할 수 없다. 키가 없을 때는 `None`으로 표기된다.



## 코드 구조
- `metrics/*.py` : ER/PC/MEC/RS 원천 로직
- `runner.py` : 메트릭 실행 오케스트레이션
- `main.py` : CLI 파서 및 결과 요약
