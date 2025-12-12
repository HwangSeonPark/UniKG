import os
import sys
import time
from datetime import datetime
from typing import Optional, Any
from pathlib import Path


# 글로벌 로거 인스턴스
_GLOG: Optional['Logger'] = None


class Logger:
    """
    평가 로그 관리 클래스
    - 콘솔 + 파일 동시 출력
    - 시간 추적
    - 상세한 진행상황 기록
    """
    
    def __init__(self, fpath: Optional[str] = None):
        """
        Args:
            fpath: 로그 파일 경로 (None이면 logs/ 디렉터리에 자동 생성)
        """
        if fpath is None:
        
            ldir = Path(__file__).parent.parent / "logs"
            ldir.mkdir(exist_ok=True)
        
            tstmp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fpath = str(ldir / f"eval_{tstmp}.log")
        
        self.fpath = fpath
        self.fhand = open(fpath, 'w', encoding='utf-8', buffering=1)
        self.strt = time.time()
        self.steps = {}  # 단계별 시작 시간
        
        
        self._write_header()
    
    def _write_header(self):
        """로그 헤더 작성"""
        hdr = f"""
시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
로그 파일: {self.fpath}
{'='*80}
"""
        self._write(hdr)
    
    def _write(self, msg: str):
        """콘솔 + 파일 동시 출력"""
        print(msg, end='')
        self.fhand.write(msg)
        self.fhand.flush()
    
    def info(self, msg: str):
        """일반 정보 로그"""
        tstmp = datetime.now().strftime('%H:%M:%S')
        elaps = time.time() - self.strt
        line = f"[{tstmp}] [{elaps:.2f}s] {msg}\n"
        self._write(line)
    
    def step(self, name: str, msg: str = ""):
        """단계 시작 로그"""
        self.steps[name] = time.time()
        tstmp = datetime.now().strftime('%H:%M:%S')
        elaps = time.time() - self.strt
        line = f"\n[{tstmp}] [{elaps:.2f}s] >> {name}"
        if msg:
            line += f": {msg}"
        line += "\n"
        self._write(line)
    
    def done(self, name: str, msg: str = ""):
        """단계 완료 로그"""
        if name in self.steps:
            dur = time.time() - self.steps[name]
            del self.steps[name]
        else:
            dur = 0.0
        
        tstmp = datetime.now().strftime('%H:%M:%S')
        elaps = time.time() - self.strt
        line = f"[{tstmp}] [{elaps:.2f}s] << {name} 완료 (소요: {dur:.2f}s)"
        if msg:
            line += f": {msg}"
        line += "\n"
        self._write(line)
    
    def emb(self, txt: str, cached: bool):
        """임베딩 로그"""
        stat = "캐시" if cached else "계산"
        self.info(f"  [임베딩] {stat}: {txt[:50]}...")
    
    def llm(self, pmt: str, resp: str):
        """LLM 호출 로그"""
        self.info(f"  [LLM 요청]")
        self.info(f"    프롬프트: {pmt[:100]}...")
        self.info(f"    응답: {resp[:200]}...")
    
    def res(self, name: str, val: Any):
        """결과 로그"""
        if isinstance(val, float):
            self.info(f"  {name}: {val:.4f}")
        elif isinstance(val, dict):
            self.info(f"  {name}:")
            for k, v in val.items():
                if isinstance(v, float):
                    self.info(f"    {k}: {v:.4f}")
                else:
                    self.info(f"    {k}: {v}")
        else:
            self.info(f"  {name}: {val}")
    
    def est(self, ntot: int, unit: str = "개"):
        """예상 소요 시간 계산"""
        # 단순 추정 (실제로는 샘플 기반으로 개선 가능)
        etime = ntot * 0.5  # 항목당 0.5초 추정
        self.info(f"예상 소요 시간: 약 {etime:.1f}초 (총 {ntot}{unit})")
    
    def close(self):
        """로그 종료"""
        total = time.time() - self.strt
        ftr = f"""
{'='*80}
평가 완료
총 소요 시간: {total:.2f}초 ({total/60:.2f}분)
종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""
        self._write(ftr)
        self.fhand.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


def init(fpath: Optional[str] = None) -> Logger:
    """
    글로벌 로거 초기화
    
    Args:
        fpath: 로그 파일 경로
    
    Returns:
        Logger 인스턴스
    """
    global _GLOG
    _GLOG = Logger(fpath)
    return _GLOG


def get() -> Optional[Logger]:
    """글로벌 로거 반환"""
    return _GLOG


def info(msg: str):
    """글로벌 로거로 정보 로그"""
    if _GLOG:
        _GLOG.info(msg)


def step(name: str, msg: str = ""):
    """글로벌 로거로 단계 시작"""
    if _GLOG:
        _GLOG.step(name, msg)


def done(name: str, msg: str = ""):
    """글로벌 로거로 단계 완료"""
    if _GLOG:
        _GLOG.done(name, msg)


def emb(txt: str, cached: bool):
    """글로벌 로거로 임베딩 로그"""
    if _GLOG:
        _GLOG.emb(txt, cached)


def llm(pmt: str, resp: str):
    """글로벌 로거로 LLM 로그"""
    if _GLOG:
        _GLOG.llm(pmt, resp)


def res(name: str, val: Any):
    """글로벌 로거로 결과 로그"""
    if _GLOG:
        _GLOG.res(name, val)


def est(ntot: int, unit: str = "개"):
    """글로벌 로거로 예상 시간"""
    if _GLOG:
        _GLOG.est(ntot, unit)


