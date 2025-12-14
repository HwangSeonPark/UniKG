#!/bin/bash

# 작업 디렉터리 설정
cd /home/hyyang/my_workspace/KGC || exit 1

# 로그 디렉터리 생성
LOG_DIR="/home/hyyang/my_workspace/KGC/evaluate/construction/logs"
mkdir -p "$LOG_DIR"

# 기본 경로 설정
EXTRACT_DIR="/home/hyyang/my_workspace/KGC/evaluate/graph_juadge/extract_LLM"
DATASET_DIR="/home/hyyang/my_workspace/KGC/datasets/construction"
LLM_NAME="gpt-5-mini"

# 평가할 모델 목록 (Tree-KG, SAC-KG, PiVe 제외)
MODELS="edc,graphjudge"

# 데이터셋 매핑: extract_LLM 디렉터리명 -> datasets/construction 디렉터리명
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
    
    # 파일 경로 설정
    # PRED: extract_LLM 디렉터리는 원래 데이터셋명 사용
    # GOLD: datasets/construction 디렉터리는 매핑된 데이터셋명 사용
    PRED="${EXTRACT_DIR}/${DS}/${LLM_NAME}/triples.txt"
    GOLD="${DATASET_DIR}/${GOLD_DS}/triples.txt"
    
    # 로그 파일 경로 생성 (llm이름_데이터셋_result 형식)
    LOG_FILE="${LOG_DIR}/${LLM_NAME}_${DS}_result"
    
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
    echo "데이터셋: ${DS} (${LLM_NAME})"
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
echo "${LLM_NAME} 모델 평가 완료"
echo "=========================================="

