import os
import json
import ast
import re
import threading
from pathlib import Path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


def verifier_get_refine_examples():
    # get refine examples from few-shot prompt
    pth = Path(__file__).resolve().parent / "prompts" / "verifier_fewshot.jsonl"
    exs = []
    for ln in pth.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        exs.append(json.loads(ln))
    return exs


_REFINE_SYSTEM_PROMPT = (
    "You are an expert in refining knowledge graph triples. "
    "Keep entities concise, use standard abbreviations, and match exact text phrasing."
)
_EXTRACT_SYSTEM_PROMPT = "You are an expert in extracting knowledge graph triples from text."

_REFINE_USER_TEMPLATE = (
    "articles: {text}\n"
    "Original triples: {pred_str}\n\n"
    "For each triple, determine if it is correct by comparing with the articles.\n"
    "- If all triples are correct: keep them as-is (only apply format normalization like abbreviations, relation simplification).\n"
    "- If some triples are incorrect: fix only the incorrect ones based on the articles.\n\n"
    "Output ONLY JSON array:\n"
    '[["subject", "relation", "object"], ...]'
)
_EXTRACT_USER_TEMPLATE = (
    "articles: {text}\n\n"
    "Extract all subject-relation-object triples from the articles.\n\n"
    "Output ONLY JSON array:\n"
    '[["subject", "relation", "object"], ...]'
)


def verifier_build_refine_user_prompt(txt: str, pred) -> str:
    # build refine user prompt
    pred_str = json.dumps(pred, ensure_ascii=False)
    return _REFINE_USER_TEMPLATE.format(text=txt, pred_str=pred_str)


def verifier_build_extract_user_prompt(txt: str) -> str:
    return _EXTRACT_USER_TEMPLATE.format(text=txt)


class TpRefBase:
    def _norm(self, tps):
        if not isinstance(tps, list):
            return tps

        norm = []
        for tp in tps:
            if isinstance(tp, list) and len(tp) == 3:
                s = str(tp[0]).replace('_', ' ').strip() if tp[0] else ""
                r = str(tp[1]).replace('_', ' ').strip() if tp[1] else ""
                o = str(tp[2]).replace('_', ' ').strip() if tp[2] else ""
                norm.append([s, r, o])
            else:
                norm.append(tp)

        return norm

    def _val(self, tps):
        if not isinstance(tps, list):
            return False

        if not tps:
            return False

        for tp in tps:
            if not isinstance(tp, list) or len(tp) != 3:
                return False
            for elem in tp:
                if not isinstance(elem, str) or not elem.strip():
                    return False

        return True

    def _is_empty(self, pstr):
        if pstr is None:
            return True

        pstr = str(pstr).strip()

        if not pstr or pstr == "" or pstr == "[]" or pstr == "[[]]":
            return True

        try:
            pred = ast.literal_eval(pstr)
            if not pred or pred == [] or pred == [[]]:
                return True
        except:
            pass

        return False

    def _mk_pr(self, txt, pred):
        # build refine message
        examples = verifier_get_refine_examples()
        usr = verifier_build_refine_user_prompt(txt, pred)

        msgs = [{"role": "system", "content": _REFINE_SYSTEM_PROMPT}]

        for ex in examples:
            user_msg = f"articles: {ex['sent']}\nOriginal: {json.dumps(ex['orig'])}"
            msgs.append({"role": "user", "content": user_msg})
            msgs.append({"role": "assistant", "content": json.dumps(ex['refn'], ensure_ascii=False)})

        msgs.append({"role": "user", "content": usr})

        return msgs

    def _mk_ex(self, txt):
        usr = verifier_build_extract_user_prompt(txt)

        return [
            {"role": "system", "content": _EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": usr}
        ]

    def _proc_one(self, idx, txt, pstr):
        try:
            if self._is_empty(pstr):
                print(f"[{idx}] Empty - extracting", flush=True)
                tps = self._extr(txt)
                stat = "extracted" if tps else "fail"
                return idx, tps if tps else [], stat

            pred = ast.literal_eval(pstr)

            if not pred or pred == [] or pred == [[]]:
                print(f"[{idx}] Parsed empty - extracting", flush=True)
                tps = self._extr(txt)
                stat = "extracted" if tps else "fail"
                return idx, tps if tps else [], stat

            refn = self.refn(txt, pred)

            if not self._val(refn):
                norm = self._norm(pred)
                if self._val(norm):
                    return idx, norm, "norm"
                return idx, pred, "kept"

            return idx, refn, "refined"

        except Exception as e:
            print(f"[{idx}] Error: {e}", flush=True)

            tps = self._extr(txt)
            if tps:
                return idx, tps, "err_extr"

            try:
                pred = ast.literal_eval(pstr)
                norm = self._norm(pred)
                if self._val(norm):
                    return idx, norm, "err_norm"
                return idx, pred, "err_kept"
            except:
                return idx, [], "err"


class TpRef(TpRefBase):
    def __init__(self, host=None, port=None, max_workers=10, model=None, max_tokens=10000):
        if model is None:
            model = os.getenv("REFINER_MODEL")

        if not model:
            raise ValueError("REFINER_MODEL is not set")

        self.mdl = model
        mdl_l = str(model).lower()
        gpt = mdl_l.startswith("gpt")

        if gpt:
            key = os.getenv("OPENAI_API_KEY", "").strip()
            if not key:
                raise ValueError("OPENAI_API_KEY is not set")
            self.cli = OpenAI(api_key=key)
        else:
            if host is None:
                host = os.getenv("REFINER_HOST", "localhost")
            if port is None:
                port = int(os.getenv("REFINER_PORT"))
            self.cli = OpenAI(
                base_url=f"http://{host}:{port}/v1",
                api_key="none"
            )
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        self.gpt = gpt
        self.ncll = 0
        self.titk = 0
        self.totk = 0
        self._lk = threading.Lock()

    def _clean_response(self, res):
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
        fixed = res.strip()
        fixed = fixed.replace(',]', ']').replace(',}', '}')
        fixed = re.sub(r',\s*\]', ']', fixed)
        fixed = re.sub(r',\s*\}', '}', fixed)
        return fixed

    def _try_partial_parse(self, res, error_pos):
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

        except Exception:
            try:
                parsed = ast.literal_eval(original_res)
                if isinstance(parsed, list):
                    return parsed
            except:
                pass
            return []

    def _call(self, msgs, temp):
        tkw = {"max_completion_tokens": self.max_tokens} if self.gpt else {"max_tokens": self.max_tokens}
        try:
            resp = self.cli.chat.completions.create(
                model=self.mdl,
                messages=msgs,
                temperature=temp,
                top_p=0.95,
                **tkw
            )
            if resp.usage:
                with self._lk:
                    self.ncll += 1
                    self.titk += (resp.usage.prompt_tokens or 0)
                    self.totk += (resp.usage.completion_tokens or 0)
            return resp.choices[0].message.content

        except Exception as e:
            print(f"    API error: {e}", flush=True)
            return None

    def _extr(self, txt):
        # build extract message
        msgs = self._mk_ex(txt)
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
        # build refine message
        msgs = self._mk_pr(txt, pred)
        res = self._call(msgs, 0.05)

        if res is None:
            return pred

        tps = self._pars(res)
        tps = self._norm(tps)

        if not self._val(tps):
            return pred

        return tps

    def proc_batch(self, txts, preds, on_result=None):
        # process batch of text and predictions
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
