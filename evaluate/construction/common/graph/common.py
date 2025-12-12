import os
import json
import re
import sys
from typing import List, Dict, Set, Optional
from collections import defaultdict, deque
import numpy as np
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


_KEY = os.getenv('GEMINI_API_KEY')
_FB = not _KEY  # 키가 없으면 fallback 모드
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

_ECCH: Dict[str, np.ndarray] = {}
_KEY_VALID = None  # 키 유효성 캐시: None(미검증), True(유효), False(무효)
_FALLBACK_MODE_NOTIFIED = False  # fallback 전환 알림 여부
_API_CHECK_DONE = False  # API 연결 테스트 완료 여부




def _ent(trips: List[List[str]]) -> Set[str]:
    """
    트리플에서 엔티티 추출
    빈 문자열은 '<EMPTY>'로 정규화하여 포함
    """
    ents = set()
    for h, r, t in trips:
        # 빈 문자열이나 공백은 '<EMPTY>'로 정규화
        h_norm = h.strip() if h and h.strip() else '<EMPTY>'
        t_norm = t.strip() if t and t.strip() else '<EMPTY>'
        ents.add(h_norm)
        ents.add(t_norm)
    return ents


def _llm(msg: str, temp: float = 0.1, key: Optional[str] = None) -> str:
    """
    LLM API 호출 (Gemini API 사용, 키 없으면 에러 발생)
    """
    k = key or _KEY
    if not k:
        raise ValueError("NO_KEY")
    
    model_name = "gemini-2.5-flash"
    url = f"{_GEMINI_BASE_URL}/{model_name}:generateContent?key={k}"
    
    hdrs = {
        "Content-Type": "application/json"
    }
    body = {
        "contents": [{
            "parts": [{
                "text": msg
            }]
        }],
        "generationConfig": {
            "temperature": temp
        }
    }
    resp = requests.post(url, headers=hdrs, json=body, timeout=30)
    if not resp.ok:
        try:
            error_data = resp.json()
            if 'error' in error_data:
                error_msg = error_data['error'].get('message', str(error_data))
                raise ValueError(f"Gemini API 오류 ({resp.status_code}): {error_msg}")
        except:
            pass
        resp.raise_for_status()
    result = resp.json()
    if 'candidates' not in result or len(result['candidates']) == 0:
        raise ValueError(f"Gemini API 응답 오류: {result}")
    return result['candidates'][0]['content']['parts'][0]['text']


def _fbem(txt: str) -> np.ndarray:
    vec = np.zeros(1536)
    for i, ch in enumerate(txt[:768]):
        vec[i % 1536] += ord(ch) / 1000.0
    for i in range(len(txt) - 1):
        bigr = txt[i:i+2]
        idx = hash(bigr) % 768 + 768
        vec[idx] += 1.0
    n = np.linalg.norm(vec)
    if n > 0:
        vec = vec / n
    return vec




def _cos(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    코사인 유사도 계산
    - 임베딩 원본이 정규화되어 있지 않은 경우도 안전하게 [-1,1] 범위로 산출
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def _top5(que: str, cand: List[str]) -> List[str]:
    qv = _emb(que)
    sims = []
    for c in cand:
        cv = _emb(c)
        sims.append((c, _cos(qv, cv)))
    sims.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in sims[:5]]


def _map(gte: str, top5: List[str], key: Optional[str] = None) -> Optional[str]:
    if not top5:
        return None
    if gte in top5:
        return gte
    if _FB and not key:
        best = top5[0]
        bsim = _cos(_emb(gte), _emb(best))
        return best if bsim > 0.7 else None

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
        resp = _llm(pmt, temp=0.1, key=key)
        resp = resp.strip().strip('"').strip("'")
        for cand in top5:
            if cand.lower() in resp.lower() or resp.lower() in cand.lower():
                return cand
        if "none" in resp.lower():
            return None
        best = top5[0]
        bsim = _cos(_emb(gte), _emb(best))
        return best if bsim > 0.7 else None
    except Exception:
        best = top5[0]
        bsim = _cos(_emb(gte), _emb(best))
        return best if bsim > 0.7 else None


def _bfs(g: Dict[str, List[str]], s: str, t: str) -> bool:
    if s == t:
        return True
    v = {s}
    q = deque([s])
    while q:
        c = q.popleft()
        if c == t:
            return True
        if c in g:
            for n in g[c]:
                if n not in v:
                    v.add(n)
                    q.append(n)
    return False


