import os
from typing import List, Dict, Set, Optional
from collections import deque
import numpy as np
import requests

# 통합 임베딩 시스템 사용 (모든 메트릭이 캐시 공유)
# GraphJudge와 동일한 bert-base-uncased 사용으로 캐시 공유
from evaluate.construction.common.embedding import emb as _emb_raw, cos as _cos

# 로거 임포트
try:
    from evaluate.construction.common import logger as log
except Exception:
    log = None

# Tree-KG는 bert-base-uncased 사용 (GraphJudge와 캐시 공유)
def _emb(txt, batch=False):
    """bert-base-uncased로 임베딩 (GraphJudge와 동일 모델)"""
    return _emb_raw(txt, mname='bert-base-uncased', batch=batch)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# API 키 및 설정
_KEY = os.getenv('GEMINI_API_KEY')
_FB = not _KEY  # 키 없으면 fallback 모드
_GURL = "https://generativelanguage.googleapis.com/v1beta/models"


def _ent(trips: List[List[str]]) -> Set[str]:
    """
    트리플 리스트에서 엔티티 추출
    - 빈 문자열은 '<EMPTY>'로 정규화
    - head, tail 엔티티만 추출 (relation 제외)
    """
    ents = set()
    for h, r, t in trips:
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


def _top5(que: str, cand: List[str]) -> List[str]:
    """
    쿼리와 가장 유사한 상위 5개 후보 반환
    - 임베딩 기반 코사인 유사도로 정렬
    - 배치 임베딩으로 효율성 향상
    """
    # 배치로 한번에 임베딩 (효율적)
    qv = _emb(que)
    cvs = _emb(cand, batch=True) if isinstance(cand, list) else [_emb(c) for c in cand]
    
    # 유사도 계산
    sims = [(c, _cos(qv, cv)) for c, cv in zip(cand, cvs)]
    sims.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in sims[:5]]


def _map(gte: str, top5: List[str], key: Optional[str] = None) -> Optional[str]:
    """
    GT 엔티티를 후보 중에서 매핑
    - 정확히 일치하면 즉시 반환
    - LLM으로 최종 판단 (키 있으면)
    - fallback: 임베딩 유사도 0.7 이상이면 매핑
    
    Args:
        gte: Ground Truth 엔티티
        top5: 후보 엔티티 리스트 (최대 5개)
        key: Gemini API 키 (선택)
    
    Returns:
        매핑된 엔티티 또는 None
    """
    if not top5:
        return None
    
    # 정확히 일치하는 경우
    if gte in top5:
        return gte
    
    # 키 없으면 임베딩 기반 매핑
    if _FB and not key:
        best = top5[0]
        bsim = _cos(_emb(gte), _emb(best))
        return best if bsim > 0.7 else None
    
    # LLM 프롬프트 구성
    pmt = f"""Role:
"You are an entity mapping expert specializing in knowledge graph evaluation."

Task/Constraints:
"Given a ground truth entity and candidate entities, determine which candidate represents the same entity as the ground truth. If no candidate matches, respond with 'none'."

Ground Truth: "{gte}"

Candidates:
{chr(10).join(f'{i+1}. {e}' for i, e in enumerate(top5))}

Output Template:
Respond with ONLY the entity name (or "none").
"""
    
    try:
        # LLM 호출하여 매핑 결정
        if log:
            log.info(f"  [매핑] GT: {gte} -> 후보: {top5}")
        resp = _llm(pmt, temp=0.1, key=key)
        resp = resp.strip().strip('"').strip("'")
        
        # 응답에서 후보 매칭
        for cand in top5:
            if cand.lower() in resp.lower() or resp.lower() in cand.lower():
                return cand
        
        # "none" 응답이면 매핑 실패
        if "none" in resp.lower():
            return None
        
        # 응답 파싱 실패 시 임베딩 기반 fallback
        best = top5[0]
        bsim = _cos(_emb(gte), _emb(best))
        return best if bsim > 0.7 else None
        
    except Exception:
        # 예외 발생 시 임베딩 기반 fallback
        best = top5[0]
        bsim = _cos(_emb(gte), _emb(best))
        return best if bsim > 0.7 else None


def _llm(msg: str, temp: float = 0.1, key: Optional[str] = None) -> str:
    """
    Gemini API 호출
    - 키 없으면 ValueError 발생
    
    Args:
        msg: 프롬프트 메시지
        temp: 온도 파라미터 (기본 0.1)
        key: API 키 (선택, 없으면 환경변수 사용)
    
    Returns:
        LLM 응답 텍스트
    """
    k = key or _KEY
    if not k:
        raise ValueError("NO_KEY")
    
    mname = "gemini-2.5-flash"
    url = f"{_GURL}/{mname}:generateContent?key={k}"
    
    hdrs = {"Content-Type": "application/json"}
    body = {
        "contents": [{
            "parts": [{"text": msg}]
        }],
        "generationConfig": {"temperature": temp}
    }
    
    # API 요청
    if log:
        log.llm(msg[:200], "...")
    resp = requests.post(url, headers=hdrs, json=body, timeout=30)
    
    # 에러 처리
    if not resp.ok:
        try:
            edata = resp.json()
            if 'error' in edata:
                emsg = edata['error'].get('message', str(edata))
                raise ValueError(f"Gemini API 오류 ({resp.status_code}): {emsg}")
        except:
            pass
        resp.raise_for_status()
    
    # 응답 파싱
    rjson = resp.json()
    if 'candidates' not in rjson or len(rjson['candidates']) == 0:
        raise ValueError(f"Gemini API 응답 오류: {rjson}")
    
    rtxt = rjson['candidates'][0]['content']['parts'][0]['text']
    if log:
        log.llm("...", rtxt[:200])
    return rtxt


def _bfs(g: Dict[str, List[str]], s: str, t: str) -> bool:
    """
    BFS로 그래프에서 두 노드 간 경로 존재 확인
    
    Args:
        g: 그래프 (인접 리스트)
        s: 시작 노드
        t: 목표 노드
    
    Returns:
        경로 존재 여부
    """
    if s == t:
        return True
    
    visit = {s}
    queue = deque([s])
    
    while queue:
        curr = queue.popleft()
        if curr == t:
            return True
        
        if curr in g:
            for nbr in g[curr]:
                if nbr not in visit:
                    visit.add(nbr)
                    queue.append(nbr)
    
    return False

