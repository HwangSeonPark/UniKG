

MINE은 이제 Hugging Face에서 평가 데이터를 자동으로 불러오므로, 로컬 파일을 관리하지 않고도 평가를 더 쉽게 실행할 수 있습니다.

## MINE 평가 지표

### MINE-1: 지식 유지력

MINE-1은 지식 그래프 추출기가 하위 작업에 의존하지 않고 기사에서 포착할 수 있는 정보의 비율을 근사합니다. 하위 작업은 성능 향상이 지식 그래프 추출기 자체에서 비롯된 것인지, 아니면 추출 과정의 측면에서 비롯된 것인지 모호하게 만들 수 있습니다.

**데이터셋 특성:**
- 기사 100개, 각 기사에 포함된 것으로 알려진 15개의 사실
- 기사 평균 길이: 592단어 (표준편차 85단어, 범위: 440-976단어)
- 다양한 주제: 예술, 문화 및 사회 (24개 기사), 과학 (27개 기사), 기술 (19개 기사), 심리학/인간 경험 (18개 기사), 역사 및 문명 (17개 기사)

**평가 과정:**
1. 평가 대상 지식 그래프 추출기를 사용하여 각 기사에서 지식 그래프 추출
2. 15개 사실 각각에 대해 해당 지식 그래프에서 복원 가능 여부 검증
3. **의미적 질의 과정:**
   - SentenceTransformers의 all-MiniLM-L6-v2 모델을 사용하여 15개 사실과 모든 지식 그래프 노드 임베딩 생성
   - 각 사실에 대해 KG 내에서 의미적으로 가장 유사한 상위 k개 노드 검색
   - 상위 k개 노드로부터 두 관계 이내의 모든 노드를 포함하도록 결과 확장(다중 홉 추론 가능)
   - 이 노드들로 유도된 부분 그래프를 LLM에 전달
4. LLM은 이진 점수를 출력: 검색된 노드와 관계만으로 사실이 추론될 경우 1, 그렇지 않을 경우 0
5. 최종 MINE-1 점수는 100개 기사 전체에서 15개 사실에 대해 1로 평가된 비율의 평균값

**검증:** 무작위로 선정된 60개 사실-KG 쌍에 대한 수동 평가와 LLM 판단 간의 일치율은 90.2%, 상관관계는 0.80을 달성했습니다.

### MINE-2: KG 지원 RAG

MINE-2는 위키피디아 1,995개 문서 기반 20,400개 질문으로 구성된 WikiQA 데이터셋을 활용해 KG 지원 RAG 시스템을 평가합니다.

**평가 과정:**
1. WikiQA에서 참조된 모든 문서의 정보를 통합한 단일 KG 구축
2. 데이터셋의 각 질문에 대해:
   - **검색:** all-MiniLM-L6-v2 모델을 사용하여 질문과 모든 KG 트리플을 임베딩
   - 질문과 각 트리플 간의 코사인 유사도 계산
   - 각 트리플에 대한 BM25 관련성 점수 계산
   - BM25 점수와 코사인 유사도 점수를 동일 가중치로 결합하여 최종 유사도 점수 산출
   - 합산 점수가 가장 높은 상위 10개 트리플 선택
   - 상위 10개 트리플 내 노드로부터 2단계 이내에 위치한 추가 트리플 10개를 연결하여 확장 (총 20개 트리플)
3. 검색된 20개 트리플, 관련 텍스트 조각, 원본 질문을 LLM에 제공
4. LLM이 이러한 입력값을 기반으로 답변 생성
5. LLM-as-a-Judge를 사용하여 LM 응답을 평가하여 질문에 대한 정답이 포함되었는지 판단

## 빠른 시작

### 1. MINE-1 평가 실행

1. OpenAI API 키를 환경 변수로 설정:
   - **Windows PowerShell:** `$env:OPENAI_API_KEY="your_actual_key_here"`
   - **Linux/Mac:** `export OPENAI_API_KEY="your_actual_key_here"`

2. MINE-1 평가 스크립트 실행:
   ```bash
   python _1_evaluation.py --model openai/gpt-5-nano --evaluation-model local
   ```

3. 결과는 `results/{모델 구성명}/`에 저장됩니다:
   - `results_{i}.json` - 각 에세이에 대한 평가 결과
   - `kg_{i}.json` - 각 에세이에 대해 생성된 지식 그래프

### 2. MINE-2 평가 실행 (KG 지원 RAG)

MINE-2는 WikiQA 데이터셋의 **질문과 답변**을 사용하여 KG 지원 RAG 시스템을 평가합니다.

1. WikiQA 데이터셋 다운로드:
   ```bash
   cd ../wikiqa
   python _1_download_articles.py
   ```

2. WikiQA 문서들로부터 KG 생성:
   ```bash
   python _2_generate_kgs.py --split-name test
   ```

3. KG 지원 RAG 평가 실행:
   - `experiments/wikiqa/utils/cluster_and_deduplication.py`의 `KGAssistedRAG` 클래스를 사용하여 질문에 대한 답변을 생성하고 평가합니다
   - 각 질문에 대해 KG에서 관련 트리플을 검색하고, LLM이 답변을 생성한 후, LLM-as-a-Judge로 정답 포함 여부를 평가합니다

### 3. 결과 비교

종합 비교 차트 및 통계 생성:
```bash
python _2_compare_results.py
```

다음이 생성됩니다:
- `results/results.png` - 종합 비교 플롯
- `results/summary.txt` - 상세 통계 및 순위
- `results/comparisons/` - 쌍별 비교 플롯

### 4. 대화형 시각화 대시보드

Streamlit 대시보드를 실행하여 결과를 대화식으로 탐색하세요:
```bash
streamlit run _3_visualize.py
```

대시보드 제공 기능:
- 📄 **에세이 브라우저** - 에세이 주제 및 내용 확인
- 🔍 **쿼리 분석** - 각 쿼리에 대한 검색된 컨텍스트 및 평가 결과 확인

## 데이터 로딩

### MINE-1 데이터

평가 스크립트는 자동으로:
- ✅ **Hugging Face에서 평가 데이터 다운로드** ([kg-gen-MINE-evaluation-dataset](https://huggingface.co/datasets/josancamon/kg-gen-MINE-evaluation-dataset))
  - 각 에세이에 대한 15개의 사실(facts)이 `generated_queries` 필드에 포함되어 있습니다
  - MINE-1은 질문-답변 형식이 아니라 사실 복구(fact recovery) 평가이므로 질문 필드는 없습니다
- ✅ **Hugging Face 사용 불가 시 로컬 파일로 대체**
- ✅ 데이터 소스에 대한 **명확한 상태 메시지 표시**

**원본 에세이:** [kg-gen-evaluation-essays](https://huggingface.co/datasets/kyssen/kg-gen-evaluation-essays)에서 이용 가능 - 지식 그래프 생성에 사용하세요.

### MINE-2 데이터

MINE-2는 WikiQA 데이터셋을 사용하며, **질문(question)과 답변(answer)이 포함되어 있습니다**. 
- WikiQA 데이터셋: 1,995개 Wikipedia 문서 기반 20,400개 질문-답변 쌍
- 각 질문에 대해 KG에서 관련 트리플을 검색하고, LLM이 답변을 생성한 후, LLM-as-a-Judge로 정답 포함 여부를 평가합니다
- WikiQA 데이터셋은 `experiments/wikiqa/_1_download_articles.py`를 통해 다운로드할 수 있습니다

## 로컬 개발

로컬 파일을 사용하거나 Hugging Face를 이용할 수 없는 경우:
- MINE-1: 평가 사실(facts)이 포함된 [`answers.json`](answers.json) 파일이 존재하는지 확인하세요
- MINE-2: WikiQA 데이터셋이 필요합니다
- 스크립트가 자동으로 로컬 파일을 감지하여 대체 파일로 사용합니다