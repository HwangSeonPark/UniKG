# EDC 메트릭 모듈


입력 포맷(라인별 파이썬 리스트 문자열)과 XML 변환 → nervaluate → Exact Triple 매크로 계산 순서를 원본과 동일하게 유지한다.

## 지원 지표
|그룹|세부 항목|
|---|---|
|nervaluate (SUB/PRED/OBJ)|Partial / Strict / Exact Precision·Recall·F1|
|Exact Triple Macro|Pred/Gold 세트를 전체 라벨 공간으로 이진화한 매크로 Precision·Recall·F1|


- nervaluate 실행 중 생성되는 XML은 `result_xmls/{dataset}` 폴더에 저장된다.
