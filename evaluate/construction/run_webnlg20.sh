#!/bin/bash

# webnlg20 데이터셋만 평가하는 스크립트
# 모든 LLM 모델에 대해 webnlg20 평가를 수행합니다

# 작업 디렉터리 설정
cd /home/hyyang/my_workspace/KGC || exit 1

# 로그 파일 설정
LOG_FILE="webnlg20.log"

# 기본 경로 설정
EXTRACT_DIR="/home/hyyang/my_workspace/KGC/evaluate/graph_juadge/extract_LLM"
DATASET_DIR="/home/hyyang/my_workspace/KGC/datasets/construction"

# 평가할 모델 목록
MODELS="edc,graphjudge"

# 데이터셋 설정
DS="webnlg20"
GOLD_DS="webnlg20"

# LLM 모델 목록
LLM_MODELS=("gpt-5-mini" "gpt-5.1" "llama-8b" "mistral" "qwen")

# 전체 평가 시작 시간 기록
TOTAL_START=$(date +%s)

echo "========================================" | tee -a "$LOG_FILE"
echo "webnlg20 데이터셋 평가 시작" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "LLM 모델: ${LLM_MODELS[@]}" | tee -a "$LOG_FILE"
echo "데이터셋: ${DS}" | tee -a "$LOG_FILE"
echo "평가 메트릭: $MODELS" | tee -a "$LOG_FILE"
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 각 LLM 모델에 대해
for LLM_NAME in "${LLM_MODELS[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "LLM 모델: ${LLM_NAME}" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    
    # 파일 경로 설정
    PRED="${EXTRACT_DIR}/${GOLD_DS}/${LLM_NAME}/triples.txt"
    GOLD="${DATASET_DIR}/${GOLD_DS}/triples.txt"
    TEXT="${DATASET_DIR}/${GOLD_DS}/articles.txt"
    
    # 파일 존재 여부 확인
    if [ ! -f "$PRED" ]; then
        echo "[경고] ${PRED} 파일이 없습니다. 건너뜁니다." | tee -a "$LOG_FILE"
        continue
    fi
    
    if [ ! -f "$GOLD" ]; then
        echo "[경고] ${GOLD} 파일이 없습니다. 건너뜁니다." | tee -a "$LOG_FILE"
        continue
    fi
    
    # 평가 실행
    echo "" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "데이터셋: ${DS} | 모델: ${LLM_NAME}" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "Pred: ${PRED}" | tee -a "$LOG_FILE"
    echo "Gold: ${GOLD}" | tee -a "$LOG_FILE"
    echo "Text: ${TEXT}" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    python3 -m evaluate.construction.main \
        --pred "$PRED" \
        --gold "$GOLD" \
        --text "$TEXT" \
        --models "$MODELS" 2>&1 | tee -a "$LOG_FILE"
    
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "${LLM_NAME} 모델 평가 완료" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

# 전체 평가 종료 시간 계산
TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "webnlg20 평가 완료" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "종료 시간: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "소요 시간: ${HOURS}시간 ${MINUTES}분 ${SECONDS}초" | tee -a "$LOG_FILE"
echo "로그 파일: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "결과 파일: evaluate/construction/result.csv" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
