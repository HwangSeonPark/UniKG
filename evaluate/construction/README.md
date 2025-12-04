# Construction Metric Suite

이 디렉터리는 Tree-KG / EDC / SAC-KG / GraphJudge / PiVe 선행연구의 평가 코드를 정리한 공간이다.


## 폴더 구조
|경로|설명|
|---|---|
|`common/`|엔티티 매핑, 그래프 입출력, 공통 헬퍼|
|`Tree-KG/`, `EDC/`, `SAC-KG/`, `GraphJudge/`, `PiVe/`|모델별 메트릭과 README, runner, CLI|
|`utils/`|데이터셋 경로/동적 모듈 로딩 헬퍼|
|`references/`|데이터 데이터셋 샘플 |

## 공통 실행기
```
python -m evaluate.construction.main \
  --pred /path/to/pred1.txt \
  --gold /path/to/g_article.txt \
  --text /path/to/article.txt \
  --models tree,edc,graphjudge,pive \
  --api-key $OPENROUTER_API_KEY
```

또는 dataset 디렉터리를 지정하고 파일명만 입력:
```
python -m evaluate.construction.main \
  --dataset /home/hyyang/my_workspace/KGC/evaluate/construction/references \
  --pred pred1.txt \
  --gold g_article.txt \
  --text article.txt \
  --models tree,edc,graphjudge,pive \
  --api-key $OPENROUTER_API_KEY
```

- `--pred`, `--gold`는 필수 인자입니다.
- `--text`는 SAC-KG 등 텍스트 기반 메트릭에서 필요합니다.
- `--models`에 `all`을 주면 다섯 모델을 직렬로 실행합니다.
- `--dataset`을 지정하면 상대 경로로 파일명만 지정할 수 있습니다.
- SAC-KG, Tree-KG RS 에선 OpenRouter API 키가 필요합니다.

## 데이터셋 요구사항
- `pred.txt`, `golden.txt`, `article.txt` (선택) 세 파일 모두 **라인별 파이썬 리스트 문자열**이어야 한다.
- 라인 수가 서로 다르면 nervaluate/LLM 평가에서 오류가 발생한다.
- README에서 요구하는 추가 전처리(예: 텍스트-트리플 라인 수 일치)를 반드시 충족해야 한다.




