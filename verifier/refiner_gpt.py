import json
import ast
import time
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from refiner_base import TpRefBase


class TpRef(TpRefBase):
    def __init__(self, api_key=None, max_workers=4):
        self.cli = OpenAI(api_key=api_key)
        self.mdl = "gpt-5.1"
        self.max_workers = max_workers
        self.stats = {'calls': 0, 'tokens': 0, 'time': 0.0, 'chars': 0}
        self.text_prefix = "Sentence"

    def _pars(self, res):
        """Parse JSON response"""
        try:
            if "```" in res:
                start = res.find("```json") + 7
                end = res.find("```", start)
                if end != -1:
                    res = res[start:end].strip()
            
            if '[' in res:
                start = res.find('[')
                res = res[start:]
            if ']' in res:
                end = res.rfind(']')
                res = res[:end+1]
            
            return json.loads(res)
        
        except Exception as e:
            print(f"  Parse error: {e}")
            return []

    def _call(self, msgs, temp, txt_len=0):
        """API call with statistics tracking"""
        try:
            st = time.time()
            resp = self.cli.chat.completions.create(
                model=self.mdl,
                messages=msgs,
                top_p=0.95
            )
            et = time.time()
            
            self.stats['calls'] = self.stats.get('calls', 0) + 1
            self.stats['time'] = self.stats.get('time', 0.0) + (et - st)
            if hasattr(resp, 'usage') and resp.usage:
                self.stats['tokens'] = self.stats.get('tokens', 0) + resp.usage.total_tokens
            if txt_len > 0:
                self.stats['chars'] = self.stats.get('chars', 0) + txt_len
            
            return resp.choices[0].message.content
        
        except Exception as e:
            print(f"  API error: {e}")
            return None

    def _extr(self, txt):
        """Extract empty triples"""
        print(f"  Extracting: {txt[:60]}...")
        msgs = self._mk_ex(txt, self.text_prefix)
        res = self._call(msgs, 0.3, len(txt))
        
        if res is None:
            return []
        
        tps = self._pars(res)
        tps = self._norm(tps)
        
        if not self._val(tps):
            print(f"  Extract failed")
            return []
        
        print(f"  Extracted {len(tps)} triples")
        return tps

    def proc_batch(self, txts, preds):
        """Batch processing with statistics"""
        res = [None] * len(txts)
        self.stats = {'calls': 0, 'tokens': 0, 'time': 0.0, 'chars': 0}
        total_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            futs = {exe.submit(self._proc_one, i, txt, pred): i
                    for i, (txt, pred) in enumerate(zip(txts, preds))}
            
            for fut in as_completed(futs):
                idx, tps, stat = fut.result()
                res[idx] = tps
                print(f"[{idx+1}/{len(txts)}] {stat}")
        
        total_time = time.time() - total_time
        self.stats['total_time'] = total_time
        return res

    def refn(self, txt, pred):
        """Refine triples"""
        msgs = self._mk_pr(txt, pred, self.text_prefix)
        res = self._call(msgs, 0.05, len(txt))
        
        if res is None:
            return pred
        
        tps = self._pars(res)
        tps = self._norm(tps)
        
        if not self._val(tps):
            return pred
        
        return tps


if __name__ == "__main__":
    print("refiner_gpt.py - TpRef class module")
    print("This file is imported by run_gpt.py to use as a library.")
    print("Actual execution is done using 'python verifier/run_gpt.py'.")
