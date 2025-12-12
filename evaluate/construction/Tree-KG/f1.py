from typing import List
from .er import er
from .pc import pc


def f1(gt: List[List[str]], pd: List[List[str]], key: str = None) -> float:
    """
    F1 점수 계산 (ER과 PC의 조화 평균)
    
    Args:
        gt: GT 트리플 [[h,r,t], ...]
        pd: PD 트리플 [[h,r,t], ...]
        key: Gemini API 키 (없으면 fallback)
    
    Returns:
        F1 값 [0,1]
    """
    # ER (Entity Recall) 계산
    ervl = er(gt, pd, key=key)
    
    # PC (Precision) 계산
    pcvl = pc(gt, pd, key=key)
    
    # 조화 평균 계산
    if ervl + pcvl == 0.0:
        return 0.0
    
    return 2.0 * ervl * pcvl / (ervl + pcvl)


