import time
from datetime import datetime
from typing import Optional, Any
from pathlib import Path



_GLOG: Optional['Logger'] = None

class Logger:
    def __init__(self, fpath: Optional[str] = None):
        if fpath is None:
        
            ldir = Path(__file__).parent.parent / "logs"
            ldir.mkdir(exist_ok=True)
        
            tstmp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fpath = str(ldir / f"eval_{tstmp}.log")
        
        self.fpath = fpath
        self.fhand = open(fpath, 'w', encoding='utf-8', buffering=1)
        self.strt = time.time()
        self.steps = {}  
        self._write_header()
    
    def _write_header(self):
        hdr = f"""
Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Log file: {self.fpath}
{'='*80}
"""
        self._write(hdr)
    
    def _write(self, msg: str):
        print(msg, end='')
        self.fhand.write(msg)
        self.fhand.flush()
    
    def info(self, msg: str):
        tstmp = datetime.now().strftime('%H:%M:%S')
        elaps = time.time() - self.strt
        line = f"[{tstmp}] [{elaps:.2f}s] {msg}\n"
        self._write(line)
    
    def step(self, name: str, msg: str = ""):
        self.steps[name] = time.time()
        tstmp = datetime.now().strftime('%H:%M:%S')
        elaps = time.time() - self.strt
        line = f"\n[{tstmp}] [{elaps:.2f}s] >> {name}"
        if msg:
            line += f": {msg}"
        line += "\n"
        self._write(line)
    
    def done(self, name: str, msg: str = ""):
        if name in self.steps:
            dur = time.time() - self.steps[name]
            del self.steps[name]
        else:
            dur = 0.0
        
        tstmp = datetime.now().strftime('%H:%M:%S')
        elaps = time.time() - self.strt
        line = f"[{tstmp}] [{elaps:.2f}s] << {name} completed (duration: {dur:.2f}s)"
        if msg:
            line += f": {msg}"
        line += "\n"
        self._write(line)
    
    def res(self, name: str, val: Any):
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
    
    def close(self):
        total = time.time() - self.strt
        ftr = f"""
{'='*80}
Evaluation completed
Total duration: {total:.2f}s ({total/60:.2f}min)
End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""
        self._write(ftr)
        self.fhand.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


def init(fpath: Optional[str] = None) -> Logger:
    global _GLOG
    _GLOG = Logger(fpath)
    return _GLOG


def info(msg: str):

    if _GLOG:
        _GLOG.info(msg)


def step(name: str, msg: str = ""):

    if _GLOG:
        _GLOG.step(name, msg)


def done(name: str, msg: str = ""):
 
    if _GLOG:
        _GLOG.done(name, msg)


def res(name: str, val: Any):
    if _GLOG:
        _GLOG.res(name, val)


