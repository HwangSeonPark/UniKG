#!/bin/bash

# 작업 디렉터리 설정
cd /home/hyyang/my_workspace/KGC || exit 1

# 로그 디렉터리 생성
# 예: /home/hyyang/my_workspace/KGC/evaluate/construction/logs
LOG_DIR=""
mkdir -p "$LOG_DIR"

# 기본 경로 설정

#예측 트리플 파일이 있는 모델의 폴더(txt말고 폴더명) 경로 예:../KGC/evaluate/EDC

#예: /home/hyyang/my_workspace/KGC/evaluate/EDC 
EXTRACT_DIR=""

#예: /home/hyyang/my_workspace/KGC/datasets/construction
GOLDEN_DIR=""
# 평가할 매트릭스 목록 
MODELS="edc,graphjudge"

# 데이터셋 매핑
declare -A DS_MAP=(
    ["GenWiki"]="GenWiki-Hard"
    ["CaRB"]="CaRB-Expert"
    ["KELM-sub"]="kelm_sub"
    ["SCIERC"]="SCIERC"
    ["webnlg20"]="webnlg20"
    )

# 데이터셋 목록
DATASETS=("GenWiki" "CaRB" "KELM-sub" "SCIERC" "webnlg20")

# 각 데이터셋에 대해 평가 실행
for DS in "${DATASETS[@]}"; do
    # 데이터셋 디렉터리명 매핑
    GOLD_DS="${DS_MAP[$DS]}"
    

    # GOLD: datasets/construction 디렉터리는 매핑된 데이터셋명 사용
    PRED="${EXTRACT_DIR}/${DS}/triples.txt"
    GOLD="${GOLDEN_DIR}/${GOLD_DS}/triples.txt"

    
    # 로그 파일 경로 생성 (데이터셋_result 형식)
    LOG_FILE="${LOG_DIR}_${DS}_result"
    
    # 파일 존재 여부 확인
    if [ ! -f "$PRED" ]; then
        echo "[경고] ${PRED} 파일이 없습니다. 건너뜁니다."
        continue
    fi
    
    if [ ! -f "$GOLD" ]; then
        echo "[경고] ${GOLD} 파일이 없습니다. 건너뜁니다."
        continue
    fi
    
    # 평가 실행
    echo "=========================================="
    echo "데이터셋: ${DS} (${BASELINE})"
    echo "=========================================="
    echo "Pred: ${PRED}"
    echo "Gold: ${GOLD}"
    echo "Log: ${LOG_FILE}"
    echo ""
    
    python3 -m evaluate.construction.main \
        --pred "$PRED" \
        --gold "$GOLD" \
        --models "$MODELS" \
        --log "$LOG_FILE"
    
    echo ""
    echo ""
done

echo "=========================================="
echo "${BASELINE} 모델 평가 완료"
echo "=========================================="

