from typing import List, Dict
from collections import defaultdict
from .common import _ent, _top5, _map, _bfs


def mec(gt: List[List[str]], pd: List[List[str]], key: str = None) -> float:
    """
    MEC 계산

    Args:
        gt: GT 트리플 [[h,r,t], ...]
        pd: PD 트리플 [[h,r,t], ...]
        key: OpenRouter API 키(없으면 fallback)

    Returns:
        MEC 값 [0,1]
    """
    if len(gt) == 0:
        return 0.0

    gtes = list(_ent(gt))
    pdes = list(_ent(pd))

    # 매핑 테이블
    mptb: Dict[str, str] = {}
    for ge in gtes:
        top5 = _top5(ge, pdes)
        mr = _map(ge, top5, key=key)
        if mr:
            mptb[ge] = mr

    # 무방향 그래프 구성 (관계 무시)
    grph: Dict[str, List[str]] = defaultdict(list)
    for h, r, t in pd:
        grph[h].append(t)
        grph[t].append(h)

    # 연결 여부 (관계 무시, 엔티티 연결만 확인)
    conn = 0
    for gh, gr, gtt in gt:
        if gh in mptb and gtt in mptb:
            mph = mptb[gh]
            mpt = mptb[gtt]
            if _bfs(grph, mph, mpt):
                conn += 1

    return conn / len(gt)


