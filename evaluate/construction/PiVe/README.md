# PiVe 메트릭 모듈

PiVe는 Triple Match·Graph Match·G-BERTScore·Graph Edit Distance를 통해 구조적 정합성을 평가하는 프레임워크다.  


## 지원 지표
|지표|설명|
|---|---|
|Triple Match (T-Precision/Recall/F1)|트리플 문자열 집합을 이진화한 micro P/R/F1|
|Graph Match (G-Precision/Recall/F1)|그래프 동형성(isomorphism) 판정 정확도. 원 코드가 accuracy만 제공하므로 세 값이 동일하다.|
|G-BERTScore (G-BS)|GraphJudge의 G-BERTScore 모듈을 그대로 사용|
|Graph Edit Distance (GED)|정규화된 그래프 편집 거리 평균|




## 코드 구조
- `triple_f1.py`, `graph_f1.py`, `ged.py`



