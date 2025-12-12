from typing import List, Dict, Optional
from evaluate.construction.common.preload import prep_tree
from .er import er
from .pc import pc
from .f1 import f1
from .mec import mec
from .rs import rs

try:
    from evaluate.construction.common import logger as log
except Exception:
    log = None


def eval(gt: List[List[str]], 
         pd: List[List[str]], 
         key: Optional[str] = None,
         prep: bool = True) -> Dict[str, float]:
    """
    Tree-KG 모든 메트릭을 한번에 평가
    - 임베딩을 미리 수행하여 효율성 향상
    
    Args:
        gt: GT 트리플 [[h,r,t], ...]
        pd: PD 트리플 [[h,r,t], ...]
        key: Gemini API 키 (선택)
        prep: 미리 임베딩 여부 (기본: True)
    
    Returns:
        {'er': float, 'pc': float, 'f1': float, 'mec': float, 'rs': float}
    """
    # 미리 임베딩 
    if prep:
        if log:
            log.step("엔티티 임베딩", f"GT {len(gt)}개, PD {len(pd)}개 트리플")
        prep_tree(gt, pd)
        if log:
            log.done("엔티티 임베딩")
    
    # 각 메트릭 계산
    res = {}
    
    if log:
        log.step("Entity Recall (ER)", "계산 중")
    res['er'] = er(gt, pd, key=key)
    if log:
        log.done("Entity Recall (ER)", f"값: {res['er']:.4f}")
    
    if log:
        log.step("Precision (PC)", "계산 중")
    res['pc'] = pc(gt, pd, key=key)
    if log:
        log.done("Precision (PC)", f"값: {res['pc']:.4f}")
    
    if log:
        log.step("F1 Score", "계산 중")
    res['f1'] = f1(gt, pd, key=key)
    if log:
        log.done("F1 Score", f"값: {res['f1']:.4f}")
    
    if log:
        log.step("MEC", "계산 중")
    res['mec'] = mec(gt, pd, key=key)
    if log:
        log.done("MEC", f"값: {res['mec']:.4f}")
    
    # RS는 키가 있을 때만
    if log:
        log.step("Relation Strength (RS)", "계산 중")
    rsval = rs(pd, key=key)
    if rsval is not None:
        res['rs'] = rsval
        if log:
            log.done("Relation Strength (RS)", f"값: {rsval:.4f}")
    elif log:
        log.done("Relation Strength (RS)", "API 키 필요")
    
    return res

