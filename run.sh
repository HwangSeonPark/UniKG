#!/bin/bash

# KG Construction Evaluation Runner
# 사용법: ./run.sh <config_number>

# 데이터셋 경로 설정
PRED="/home/hyyang/my_workspace/KGC/evaluate/graph_juadge/extract_LLM/GenWiki/mistral/triples.txt"
GOLD="/home/hyyang/my_workspace/KGC/datasets/construction/GenWiki-Hard/triples.txt"
TEXT="/home/hyyang/my_workspace/KGC/datasets/construction/GenWiki-Hard/articles.txt"

# 작업 디렉터리 설정
cd /home/hyyang/my_workspace/KGC

# 로그 디렉터리 생성
mkdir -p /home/hyyang/my_workspace/KGC/evaluate/construction/logs

# 설정 번호 (인자로 받음, 기본값 0)
CFG=${1:-0}

case $CFG in
  1)
    # 1. Tree-KG만 실행 (엔티티 임베딩)
    echo "==> Tree-KG 평가 시작"
    python3 -m evaluate.construction.main \
      --pred "$PRED" \
      --gold "$GOLD" \
      --models tree
    ;;
    
  2)
    # 2. GraphJudge만 실행 (문장 임베딩)
    echo "==> GraphJudge 평가 시작"
    python3 -m evaluate.construction.main \
      --pred "$PRED" \
      --gold "$GOLD" \
      --models graphjudge
    ;;
    
  3)
    # 3. PiVe만 실행 (문장 임베딩)
    echo "==> PiVe 평가 시작"
    python3 -m evaluate.construction.main \
      --pred "$PRED" \
      --gold "$GOLD" \
      --models pive
    ;;
    
  4)
    # 4. EDC만 실행 (임베딩 없음)
    echo "==> EDC 평가 시작"
    python3 -m evaluate.construction.main \
      --pred "$PRED" \
      --gold "$GOLD" \
      --models edc
    ;;
    
  5)
    # 5. SAC-KG만 실행 (LLM 기반)
    echo "==> SAC-KG 평가 시작"
    python3 -m evaluate.construction.main \
      --pred "$PRED" \
      --text "$TEXT" \
      --models sac
    ;;
    
  6)
    # 6. GraphJudge + PiVe (문장 임베딩 공유)
    echo "==> GraphJudge + PiVe 평가 시작"
    python3 -m evaluate.construction.main \
      --pred "$PRED" \
      --gold "$GOLD" \
      --models graphjudge,pive
    ;;
    
  7)
    # 7. GraphJudge + EDC (문장 임베딩)
    echo "==> GraphJudge + EDC 평가 시작"
    python3 -m evaluate.construction.main \
      --pred "$PRED" \
      --gold "$GOLD" \
      --models graphjudge,edc
    ;;
    
  8)
    # 8. Tree-KG + EDC (엔티티 임베딩)
    echo "==> Tree-KG + EDC 평가 시작"
    python3 -m evaluate.construction.main \
      --pred "$PRED" \
      --gold "$GOLD" \
      --models tree,edc
    ;;
    
  9)
    # 9. Tree-KG + GraphJudge (모델 공유)
    echo "==> Tree-KG + GraphJudge 평가 시작"
    python3 -m evaluate.construction.main \
      --pred "$PRED" \
      --gold "$GOLD" \
      --models tree,graphjudge
    ;;
    
  10)
    # 10. llama-8b 전체 데이터셋 평가 (EDC 포함)
    echo "==> llama-8b 전체 데이터셋 평가 시작"
    echo ""
    
    # CaRB 데이터셋
    echo "=========================================="
    echo "Dataset: CaRB (llama-8b)"
    echo "=========================================="
    python3 -m evaluate.construction.main \
      --pred /home/hyyang/my_workspace/KGC/evaluate/graph_juadge/extract_LLM/CaRB/llama-8b/triples.txt \
      --gold /home/hyyang/my_workspace/KGC/datasets/construction/CaRB-Expert/triples.txt \
      --models edc,graphjudge,pive
    echo ""
    
    # GenWiki 데이터셋  
    echo "=========================================="
    echo "Dataset: GenWiki (llama-8b)"
    echo "=========================================="
    python3 -m evaluate.construction.main \
      --pred /home/hyyang/my_workspace/KGC/evaluate/graph_juadge/extract_LLM/GenWiki/llama-8b/triples.txt \
      --gold /home/hyyang/my_workspace/KGC/datasets/construction/GenWiki-Hard/triples.txt \
      --models edc,graphjudge,pive
    echo ""
    
    # KELM-sub 데이터셋
    echo "=========================================="
    echo "Dataset: KELM-sub (llama-8b)"
    echo "=========================================="
    python3 -m evaluate.construction.main \
      --pred /home/hyyang/my_workspace/KGC/evaluate/graph_juadge/extract_LLM/KELM-sub/llama-8b/triples.txt \
      --gold /home/hyyang/my_workspace/KGC/datasets/construction/kelm_sub/triples.txt \
      --models edc,graphjudge,pive
    echo ""
    
    # SCIERC 데이터셋
    echo "=========================================="
    echo "Dataset: SCIERC (llama-8b)"
    echo "=========================================="
    python3 -m evaluate.construction.main \
      --pred /home/hyyang/my_workspace/KGC/evaluate/graph_juadge/extract_LLM/SCIERC/llama-8b/triples.txt \
      --gold /home/hyyang/my_workspace/KGC/datasets/construction/SCIERC/triples.txt \
      --models edc,graphjudge,pive
    echo ""
    
    echo "==> llama-8b 전체 데이터셋 평가 완료"
    ;;
    
  *)
    # 사용법 출력
    echo "KG Construction Evaluation Runner"
    echo ""
    echo "사용법: ./run.sh <config_number>"
    echo ""
    echo "설정 번호:"
    echo "  1  - Tree-KG만 (엔티티 임베딩)"
    echo "  2  - GraphJudge만 (문장 임베딩)"
    echo "  3  - PiVe만 (문장 임베딩)"
    echo "  4  - EDC만 (임베딩 없음)"
    echo "  5  - SAC-KG만 (LLM 기반)"
    echo "  6  - GraphJudge + PiVe (문장 임베딩 공유)"
    echo "  7  - GraphJudge + EDC"
    echo "  8  - Tree-KG + EDC"
    echo "  9  - Tree-KG + GraphJudge (BERT 모델 공유)"
    echo "  10 - 전체 메트릭 실행"
    echo ""
    echo "예제:"
    echo "  ./run.sh 1    # Tree-KG만 실행"
    echo "  ./run.sh 9    # Tree-KG + GraphJudge 실행"
    echo "  ./run.sh 10   # 모든 메트릭 실행"
    echo ""
    echo "경로 설정 (스크립트 내부 수정 필요):"
    echo "  PRED: $PRED"
    echo "  GOLD: $GOLD"
    echo "  TEXT: $TEXT"
    ;;
esac

