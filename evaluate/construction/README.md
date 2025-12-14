# Construction Metric Suite

선행연구의 평가 코드를 정리한 공간입니다.
## 현재 제가 절대 경로로 박아놔서... 경로는 여러분의 절대경로로 하시면 됩니다 

## 폴더 구조

| 경로 | 설명 |
|------|------|
| `common/` | 엔티티 매핑, 그래프 입출력, 공통 헬퍼 |
| `Tree-KG/`, `EDC/`, `SAC-KG/`, `GraphJudge/`, `PiVe/` | 모델별 메트릭과 README, runner, CLI |
| `utils/` | 데이터셋 경로/동적 모듈 로딩 헬퍼 |
| `references/` | 데이터셋 샘플 |

---

##Baseline 모델 평가

### 1. 예측 트리플 파일 구조

Baseline 모델(EDC, GraphJudge 등)의 예측 트리플 파일은 다음과 같은 구조로 준비되어야 합니다.

```
EXTRACT_DIR/
├── GenWiki/
│   └── triples.txt
├── CaRB/
│   └── triples.txt
├── KELM-sub/
│   └── triples.txt
├── SCIERC/
│   └── triples.txt
└── webnlg20/
    └── triples.txt
```

**예시: EDC 모델**
```
/home/hyyang/my_workspace/KGC/evaluate/EDC/
├── GenWiki/
│   └── triples.txt
├── CaRB/
│   └── triples.txt
└── ...
```

### 2. `baseline.sh` 설정

`baseline.sh` 파일을 열어 다음 변수들을 설정하세요:

```bash
# 로그 디렉터리 경로
# 예: /home/hyyang/my_workspace/KGC/evaluate/construction/logs
LOG_DIR=""

# 예측 트리플 파일이 있는 모델의 폴더 경로
# 예: /home/hyyang/my_workspace/KGC/evaluate/EDC
EXTRACT_DIR=""

# 정답 데이터셋 디렉터리 경로
# 예: /home/hyyang/my_workspace/KGC/datasets/construction
GOLDEN_DIR=""

# 평가할 메트릭 목록 (콤마로 구분) 우리는 이것만 쓸거니까 아래 그대로 유지 하면 됩니다. 
# 사용 가능: edc, graphjudge, tree, sac
MODELS="edc,graphjudge"
```

### 3. 데이터셋 매핑

스크립트 내부의 데이터셋 매핑은 다음과 같이 설정되어 있습니다:

| 예측 파일 디렉터리명 | 정답 파일 디렉터리명 |
|---------------------|-------------------|
| `GenWiki` | `GenWiki-Hard` |
| `CaRB` | `CaRB-Expert` |
| `KELM-sub` | `kelm_sub` |
| `SCIERC` | `SCIERC` |
| `webnlg20` | `webnlg20` |

필요에 따라 `DS_MAP` 배열을 수정할 수 있지만 여러분은 안해도 됩니다 

### 4. 실행 방법

```bash
cd /home/hyyang/my_workspace/KGC/evaluate/construction
./baseline.sh
```

스크립트는 각 데이터셋에 대해 자동으로 평가를 실행하고, 결과를 로그 파일에 저장합니다. 


### 아래 내용은 안 읽으셔도 됩니다 감사합니다 

------------------------------

## 🔧 공통 실행기 (직접 실행)

개별 파일에 대해 평가를 실행하려면:

```bash
python3 -m evaluate.construction.main \
  --pred /path/to/pred.txt \
  --gold /path/to/gold.txt \
  --models edc,graphjudge \
  --log /path/to/log_file
```

### 인자 설명

| 인자 | 필수 | 설명 |
|------|------|------|
| `--pred` | ✅ | 예측 트리플 파일 경로 |
| `--gold` | ✅ | 정답 트리플 파일 경로 (SAC-KG 제외) |
| `--text` | ❌ | 원문 텍스트 파일 경로 (SAC-KG 등에서만 필요) |
| `--models` | ❌ | 평가할 메트릭 (콤마 구분, 기본값: `all`) |
| `--log` | ❌ | 로그 파일 경로 (미지정 시 자동 생성) |
| `--api-key` | ❌ | LLM 메트릭용 Gemini API 키 (환경변수 `GEMINI_API_KEY` 사용 가능) |

### 사용 가능한 메트릭

- `edc`: EDC (Partial, Strict, Exact, Exact Triple)
- `graphjudge`: GraphJudge (G-BLEU, G-ROUGE, G-BERTScore)
- `tree`: Tree-KG (Entity Recall, Precision, F1, MEC, RS)
- `sac`: SAC-KG (LLM Precision, 텍스트 파일 필요)
- `all`: 모든 메트릭 실행

### 예시: Dataset 디렉터리 지정

```bash
python3 -m evaluate.construction.main \
  --dataset /home/hyyang/my_workspace/KGC/evaluate/construction/references \
  --pred pred1.txt \
  --gold g_article.txt \
  --models edc,graphjudge
```

`--dataset`을 지정하면 상대 경로로 파일명만 지정할 수 있습니다.

---

## 📋 데이터셋 요구사항

### 파일 형식

- `pred.txt`, `gold.txt`, `article.txt` (선택) 세 파일 모두 **라인별 파이썬 리스트 문자열**이어야 합니다.
- 각 라인은 하나의 샘플을 나타냅니다.

### 중요 사항

- ⚠️ **라인 수 일치**: `pred.txt`와 `gold.txt`의 라인 수가 서로 다르면 평가에서 오류가 발생할 수 있습니다.
- ⚠️ **전처리**: 각 메트릭의 README에서 요구하는 추가 전처리(예: 텍스트-트리플 라인 수 일치)를 반드시 충족해야 합니다.

### 예시 파일 형식

```
["entity1 | relation1 | entity2"]
["entity3 | relation2 | entity4"]
["entity5 | relation3 | entity6"]
```

---

## 📝 참고사항

- 모든 평가는 프로젝트 루트 디렉터리(`/home/hyyang/my_workspace/KGC`)에서 실행되어야 합니다.
- 로그 파일은 지정한 `LOG_DIR`에 저장됩니다.
- CSV 결과 파일은 `evaluate/construction/result.csv`에 자동으로 저장됩니다.
