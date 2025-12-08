from typing import Tuple
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score
from .webnlg.common import convert_to_xml, getRefs, getCands


def calculateExactTripleScore(reflist, candlist) -> Tuple[float, float, float]:
    """
    Exact triple 기반 P/R/F1 
    reflist, candlist: entry별 triple 문자열 리스트 (예: [['a | b | c', 'd | e | f'], ...])
    """
    # 위치별 정확도 계산 (position-based evaluation)
    # 빈 문자열은 common.py에서 이미 고유 토큰으로 치환됨
    total = 0
    match = 0
    
    # 각 entry 처리
    for ref_entry, cand_entry in zip(reflist, candlist):
        # 각 entry의 triple 처리
        for ref_trip_str, cand_trip_str in zip(ref_entry, cand_entry):
            # triple 문자열을 요소로 분리
            ref_parts = [p.strip() for p in ref_trip_str.split(" | ")]
            cand_parts = [p.strip() for p in cand_trip_str.split(" | ")]
            
            # 요소별 비교
            for ref_elem, cand_elem in zip(ref_parts, cand_parts):
                total += 1
                if ref_elem == cand_elem:
                    match += 1
    
    # Precision = Recall = 일치한 요소 수 / 전체 요소 수
    if total > 0:
        prec = match / total
        rec = match / total
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    else:
        prec = rec = f1 = 0.0
    
    return float(prec), float(rec), float(f1)


def f1(pred_path: str, gold_path: str) -> Tuple[float, float, float]:
    """
    XML 변환 → 정규화 → Exact Triple 매크로 P/R/F1 계산
    """
    pred_xml_path, gold_xml_path = convert_to_xml(pred_path, gold_path, max_length_diff=None)
    _, reflist = getRefs(gold_xml_path)  # 정규화된 리스트 사용
    _, candlist = getCands(pred_xml_path)  # 정규화된 리스트 사용
    return calculateExactTripleScore(reflist, candlist)


