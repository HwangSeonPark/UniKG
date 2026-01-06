import os
import json
import ast
import re
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from refiner_base import TpRefBase


class TpRef(TpRefBase):
    def __init__(self, host=None, port=None, max_workers=10, model=None, max_tokens=10000):
        # Get values from environment variables if arguments are not provided
        if host is None:
            host = os.getenv("REFINER_HOST", "localhost")
        if port is None:
            port = int(os.getenv("REFINER_PORT"))
        if model is None:
            model = os.getenv("REFINER_MODEL")
        self.cli = OpenAI(
            base_url=f"http://{host}:{port}/v1",
            api_key="none"
        )
        self.mdl = model
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        self.text_prefix = "articles"
    def _clean_response(self, res):
        """Remove code blocks and extract array from response"""
        if "```json" in res:
            start = res.find("```json")
            end = res.find("```", start + 7)
            if end != -1:
                res = res[start + 7:end].strip()
        elif "```" in res:
            start = res.find("```")
            end = res.find("```", start + 3)
            if end != -1:
                res = res[start + 3:end].strip()
        
        if '[' in res:
            start = res.find('[')
            res = res[start:]
            if ']' in res:
                end = res.rfind(']')
                res = res[:end+1]
        
        return res
    
    def _fix_json(self, res):
        """Fix common JSON errors"""
        fixed = res.strip()
        fixed = fixed.replace(',]', ']').replace(',}', '}')
        fixed = re.sub(r',\s*\]', ']', fixed)
        fixed = re.sub(r',\s*\}', '}', fixed)
        return fixed
    
    def _try_partial_parse(self, res, error_pos):
        """Try to parse partial response up to error position"""
        if error_pos <= 0:
            return None
        
        partial = res[:error_pos]
        last_complete = partial.rfind(']')
        if last_complete <= 0:
            return None
        
        bracket_count = 0
        for i in range(last_complete, -1, -1):
            if res[i] == ']':
                bracket_count += 1
            elif res[i] == '[':
                bracket_count -= 1
                if bracket_count == 0:
                    partial_res = res[:i+1] + ']'
                    try:
                        parsed = json.loads(partial_res)
                        if isinstance(parsed, list):
                            print(f"    Partial parse: {len(parsed)} triples (truncated at position {error_pos})", flush=True)
                            return parsed
                    except:
                        pass
        return None
    
    def _try_line_by_line_parse(self, res):
        """Try to parse response line by line"""
        lines = res.split('\n')
        all_triples = []
        for line in lines:
            line = line.strip()
            if not line or not (line.startswith('[') or line.startswith('(')):
                continue
            if line.endswith(','):
                line = line[:-1]
            try:
                item = ast.literal_eval(line)
                if isinstance(item, list) and len(item) == 3:
                    all_triples.append(item)
            except:
                continue
        if all_triples:
            print(f"    Line-by-line parse: {len(all_triples)} triples", flush=True)
            return all_triples
        return None
   
    def _pars(self, res):
        """Parse JSON response with robust error handling"""
        if not res or not isinstance(res, str):
            return []
        
        original_res = res
        
        try:
            res = self._clean_response(res)
            
            try:
                return json.loads(res)
            except json.JSONDecodeError as je:
                try:
                    parsed = ast.literal_eval(res)
                    if isinstance(parsed, list):
                        return parsed
                except:
                    pass
                
                fixed = self._fix_json(res)
                try:
                    return json.loads(fixed)
                except:
                    try:
                        parsed = ast.literal_eval(fixed)
                        if isinstance(parsed, list):
                            return parsed
                    except:
                        pass
                
                partial = self._try_partial_parse(res, je.pos)
                if partial:
                    return partial
                
                line_parse = self._try_line_by_line_parse(res)
                if line_parse:
                    return line_parse
                
                return []
           
        except Exception as e:
            pass
            
            try:
                parsed = ast.literal_eval(original_res)
                if isinstance(parsed, list):
                    return parsed
            except:
                pass
            return []
   
    def _call(self, msgs, temp):
        """API call"""
        try:
            resp = self.cli.chat.completions.create(
                model=self.mdl,
                messages=msgs,
                temperature=temp,
                max_tokens=self.max_tokens,
                top_p=0.95
            )
           
            return resp.choices[0].message.content
           
        except Exception as e:
            print(f"    API error: {e}", flush=True)
            return None
   
    def _extr(self, txt):
        """Extract empty triples"""
        msgs = self._mk_ex(txt, self.text_prefix)
        res = self._call(msgs, 0.3)
       
        if res is None:
            return []
       
        tps = self._pars(res)
        tps = self._norm(tps)
       
        if not self._val(tps):
            if isinstance(tps, list):
                valid_tps = []
                for tp in tps:
                    if isinstance(tp, list) and len(tp) == 3:
                        if all(isinstance(elem, str) and elem.strip() for elem in tp):
                            valid_tps.append(tp)
                if valid_tps:
                    return valid_tps
            return []
       
        return tps
    
    def refn(self, txt, pred):
        """Refine triples"""
        msgs = self._mk_pr(txt, pred, self.text_prefix)
        res = self._call(msgs, 0.05)
        
        if res is None:
            return pred
        
        tps = self._pars(res)
        tps = self._norm(tps)
        
        if not self._val(tps):
            return pred
        
        return tps
   
    def proc_batch(self, txts, preds, on_result=None):
        """Batch processing with optional callback"""
        res = [None] * len(txts)
       
        with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            futs = {exe.submit(self._proc_one, i, txt, pred): i
                    for i, (txt, pred) in enumerate(zip(txts, preds))}
           
            for fut in as_completed(futs):
                idx, tps, stat = fut.result()
                res[idx] = tps
                print(f"[{idx+1}/{len(txts)}] {stat}", flush=True)
                if on_result is not None:
                    on_result(idx, tps, stat, res)
       
        return res
