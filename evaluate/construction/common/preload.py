from typing import List, Set
from evaluate.construction.common.embedding import preload_texts


def _exte(trips: List[List[str]]) -> Set[str]:
    """
    트리플에서 모든 엔티티 추출
    
    Args:
        trips: 트리플 리스트 [[h,r,t], ...]
    
    Returns:
        엔티티 집합
    """
    ents = set()
    for trip in trips:
        if len(trip) >= 3:
            h, r, t = trip[0], trip[1], trip[2]
            # 문자열 타입 확인 및 빈 문자열 처리
            if isinstance(h, str):
                hnrm = h.strip() if h and h.strip() else '<EMPTY>'
            else:
                hnrm = str(h)
            if isinstance(t, str):
                tnrm = t.strip() if t and t.strip() else '<EMPTY>'
            else:
                tnrm = str(t)
            ents.add(hnrm)
            ents.add(tnrm)
    return ents


def prep_tree(gt: List[List[str]], pd: List[List[str]], 
              mname: str = 'bert-base-uncased'):
    """
    Tree-KG 평가 전에 모든 엔티티를 미리 임베딩
    - ER, PC, F1, MEC 평가 시 효율성 향상
    - bert-base-uncased 사용 (GraphJudge와 캐시 공유)
    
    Args:
        gt: GT 트리플 리스트
        pd: PD 트리플 리스트
        mname: 모델명 (기본: bert-base-uncased)
    """
    # GT와 PD의 모든 엔티티 추출
    ents = _exte(gt) | _exte(pd)
    
    # 배치로 미리 임베딩
    if ents:
        preload_texts(list(ents), mname=mname)


def prep_batch(gt_list: List[List[List[str]]], 
               pd_list: List[List[List[str]]],
               mname: str = 'bert-base-uncased'):
    """
    여러 샘플의 Tree-KG 평가 전에 모든 엔티티를 미리 임베딩
    - bert-base-uncased 사용 (GraphJudge와 캐시 공유)
    
    Args:
        gt_list: GT 트리플 리스트의 리스트
        pd_list: PD 트리플 리스트의 리스트
        mname: 모델명 (기본: bert-base-uncased)
    """
    # 모든 샘플의 엔티티 수집
    allent = set()
    for gt in gt_list:
        allent |= _exte(gt)
    for pd in pd_list:
        allent |= _exte(pd)
    
    # 배치로 미리 임베딩
    if allent:
        preload_texts(list(allent), mname=mname)

