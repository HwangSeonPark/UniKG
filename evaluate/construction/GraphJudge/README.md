# GraphJudge 메트릭 모듈

GraphJudge는 그래프를 엣지 시퀀스로 변환한 뒤 헝가리안 매칭으로 품질을 측정하는 논문이다. 
본 모듈은  G-BERTScore/G-BLEU/G-ROUGE를 계산한다.

## 지원 지표
|지표|설명|
|---|---|
|G-BERTScore (Precision/Recall/F1)|트리플을 문장으로 간주해 BERTScore F1을 비용 행렬로 구성 후 매칭|
|G-BLEU (Precision/Recall/F1)|BLEU 점수 행렬을 이용한 그래프 매칭|
|G-ROUGE (Precision/Recall/F1)|ROUGE-2 Precision 기반 비용 행렬을 이용한 매칭|


- 입력은 라인별 트리플 리스트 문자열이어야 한다.
- GPU 사용이 어려울 경우 `bert-score`가 CPU로 자동 전환된다.

## 코드 구조
- `/g_*.py` : 각 지표의 원본 구현




