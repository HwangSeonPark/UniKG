from typing import List
from .common import _ent, _top5, _map


def er(gt: List[List[str]], pd: List[List[str]], key: str = None) -> float:
    """
    ER 계산

    Args:
        gt: GT 트리플 [[h,r,t], ...]
        pd: PD 트리플 [[h,r,t], ...]
        key: OpenRouter API 키(없으면 fallback)

    Returns:
        ER 값 [0,1]
    """
    gtes = list(_ent(gt))
    pdes = list(_ent(pd))
    if len(gtes) == 0:
        return 0.0

    mapd = 0
    for ge in gtes:
        top5 = _top5(ge, pdes)
        mr = _map(ge, top5, key=key)
        if mr:
            mapd += 1
    return mapd / len(gtes)


