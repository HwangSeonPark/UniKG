import os
import numpy as np
from typing import List, Dict, Optional, Union
from collections import OrderedDict

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# 로거 임포트
try:
    from evaluate.construction.common import logger as log
except Exception:
    log = None


# 글로벌 임베딩 캐시 (모든 메트릭이 공유)
_GCCH: Dict[str, Dict[str, np.ndarray]] = {}  # {모델명: {텍스트: 벡터}}
_MODS: Dict[str, object] = {}  # 로드된 모델 인스턴스
_DEVC = None  # 디바이스 설정


def _gdev() -> str:
    """
    사용 가능한 디바이스 반환
    - CUDA 사용 가능하면 cuda:1 (또는 cuda:0)
    - 아니면 cpu
    """
    global _DEVC
    if _DEVC is not None:
        return _DEVC
    
    try:
        import torch
        if torch.cuda.is_available():
            # cuda:1 우선, 없으면 cuda:0
            if torch.cuda.device_count() > 1:
                _DEVC = "cuda:1"
            else:
                _DEVC = "cuda:0"
        else:
            _DEVC = "cpu"
    except Exception:
        _DEVC = "cpu"
    
    return _DEVC


def _lmod(mname: str):
    """
    모델 로드 (캐시 사용)
    
    Args:
        mname: 모델명 ('bert-base-uncased', )
    
    Returns:
        모델 인스턴스 또는 None
    """
    global _MODS
    
    if mname in _MODS:
        return _MODS[mname]
    
    try:
        # sentence-transformers 모델
        if 'MiniLM' in mname or 'sentence' in mname.lower():
            from sentence_transformers import SentenceTransformer
            mod = SentenceTransformer(mname)
            dev = _gdev()
            if dev != "cpu":
                mod = mod.to(dev)
            _MODS[mname] = mod
            return mod
        
        # BERT 모델 (transformers 사용)
        elif 'bert' in mname.lower():
            from transformers import AutoTokenizer, AutoModel
            import torch
            tok = AutoTokenizer.from_pretrained(mname)
            mod = AutoModel.from_pretrained(mname)
            dev = _gdev()
            if dev != "cpu":
                mod = mod.to(dev)
            mod.eval()
            _MODS[mname] = (tok, mod)
            return (tok, mod)
        
    except Exception:
        pass
    
    _MODS[mname] = None
    return None


def _fbem(txt: str, dim: int = 768) -> np.ndarray:
    """
    fallback 임베딩 (해시 기반)
    - 모델 로드 실패 시 사용
    - bert-base-uncased와 동일한 768 차원
    
    Args:
        txt: 텍스트
        dim: 벡터 차원 (기본: 768)
    
    Returns:
        임베딩 벡터
    """
    vec = np.zeros(dim)
    half = dim // 2
    for i, ch in enumerate(txt[:half]):
        vec[i % dim] += ord(ch) / 1000.0
    for i in range(len(txt) - 1):
        bigr = txt[i:i+2]
        idx = hash(bigr) % half + half
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def emb(txt: Union[str, List[str]], 
        mname: str = 'bert-base-uncased',
        batch: bool = True) -> Union[np.ndarray, List[np.ndarray]]:
    """
    텍스트 임베딩 (글로벌 캐시 사용)
    
    Args:
        txt: 단일 텍스트 또는 텍스트 리스트
        mname: 모델명 (기본: bert-base-uncased, GraphJudge와 통일)
        batch: 배치 처리 여부 (리스트일 때만)
    
    Returns:
        임베딩 벡터 또는 벡터 리스트
    """
    global _GCCH
    
    # 캐시 초기화
    if mname not in _GCCH:
        _GCCH[mname] = {}
    
    # 단일 텍스트
    if isinstance(txt, str):
        if txt in _GCCH[mname]:
            if log:
                log.emb(txt, cached=True)
            return _GCCH[mname][txt]
        
        if log:
            log.emb(txt, cached=False)
        vec = _emb1(txt, mname)
        _GCCH[mname][txt] = vec
        return vec
    
    # 리스트 처리
    if batch:
        # 배치로 한번에 처리 (효율적)
        miss = [t for t in txt if t not in _GCCH[mname]]
        if miss:
            if log:
                log.info(f"  [임베딩 배치] {len(miss)}개 계산 중 (전체 {len(txt)}개)")
            vecs = _embm(miss, mname)
            for t, v in zip(miss, vecs):
                _GCCH[mname][t] = v
        elif log:
            log.info(f"  [임베딩 배치] {len(txt)}개 모두 캐시 사용")
        return [_GCCH[mname][t] for t in txt]
    else:
        # 하나씩 처리
        return [emb(t, mname, batch=False) for t in txt]


def _emb1(txt: str, mname: str) -> np.ndarray:
    """
    단일 텍스트 임베딩
    """
    mod = _lmod(mname)
    
    # 모델 로드 실패 시 fallback
    if mod is None:
        return _fbem(txt, 768)
    
    try:
        # sentence-transformers
        if isinstance(mod, object) and hasattr(mod, 'encode'):
            vec = mod.encode(txt, convert_to_numpy=True, show_progress_bar=False)
            return vec
        
        # BERT (transformers)
        elif isinstance(mod, tuple):
            tok, mdl = mod
            import torch
            with torch.no_grad():
                inps = tok(txt, return_tensors='pt', truncation=True, max_length=512)
                dev = _gdev()
                if dev != "cpu":
                    inps = {k: v.to(dev) for k, v in inps.items()}
                outs = mdl(**inps)
                # [CLS] 토큰 사용
                vec = outs.last_hidden_state[0, 0, :].cpu().numpy()
                return vec
    except Exception:
        pass
    
    # 예외 발생 시 fallback
    return _fbem(txt, 768)


def _embm(txts: List[str], mname: str) -> List[np.ndarray]:
    """
    배치 텍스트 임베딩 (효율적)
    """
    if not txts:
        return []
    
    mod = _lmod(mname)
    
    # 모델 로드 실패 시 fallback
    if mod is None:
        return [_fbem(t, 768) for t in txts]
    
    try:
        # sentence-transformers
        if isinstance(mod, object) and hasattr(mod, 'encode'):
            vecs = mod.encode(txts, convert_to_numpy=True, show_progress_bar=False, batch_size=32)
            return [vecs[i] for i in range(len(vecs))]
        
        # BERT (transformers)
        elif isinstance(mod, tuple):
            tok, mdl = mod
            import torch
            with torch.no_grad():
                inps = tok(txts, return_tensors='pt', truncation=True, max_length=512, padding=True)
                dev = _gdev()
                if dev != "cpu":
                    inps = {k: v.to(dev) for k, v in inps.items()}
                outs = mdl(**inps)
                # [CLS] 토큰 사용
                vecs = outs.last_hidden_state[:, 0, :].cpu().numpy()
                return [vecs[i] for i in range(len(vecs))]
    except Exception:
        pass
    
    # 예외 발생 시 fallback
    return [_fbem(t, 768) for t in txts]


def cos(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    코사인 유사도 계산
    
    Args:
        v1: 벡터1
        v2: 벡터2
    
    Returns:
        유사도 [-1, 1]
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def clear_cache(mname: Optional[str] = None):
    """
    캐시 초기화
    
    Args:
        mname: 특정 모델 캐시만 삭제 (None이면 전체)
    """
    global _GCCH
    if mname:
        if mname in _GCCH:
            _GCCH[mname].clear()
    else:
        _GCCH.clear()


def get_cache_size(mname: Optional[str] = None) -> int:
    """
    캐시 크기 반환
    
    Args:
        mname: 특정 모델 (None이면 전체)
    
    Returns:
        캐시된 임베딩 개수
    """
    global _GCCH
    if mname:
        return len(_GCCH.get(mname, {}))
    else:
        return sum(len(c) for c in _GCCH.values())


def preload_texts(txts: List[str], mname: str = 'bert-base-uncased'):
    """
    텍스트들을 미리 임베딩하여 캐시에 저장
    
    Args:
        txts: 텍스트 리스트
        mname: 모델명 (기본: bert-base-uncased)
    """
    uniq = list(set(txts))
    # 배치로 임베딩
    emb(uniq, mname=mname, batch=True)

