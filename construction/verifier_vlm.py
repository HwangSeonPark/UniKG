import json
import ast

from pathlib import Path


def read_txt(name: str) -> str:
    """Read a UTF-8 prompt file from construction/prompts."""
    pth = Path(__file__).resolve().parent / "prompts" / name
    return pth.read_text(encoding="utf-8")


VERIFIER_REFINE_SYSTEM_PROMPT = (
    "You are an expert in refining knowledge graph triples. "
    "Keep entities concise, use standard abbreviations, and match exact text phrasing."
)

VERIFIER_EXTRACT_SYSTEM_PROMPT = (
    "You are an expert in extracting knowledge graph triples from scientific text. "
    "Extract concise triples using abbreviations when defined."
)


def verifier_get_refine_examples():
    return json.loads(read_txt("verifier_fewshot.txt"))


def verifier_build_refine_user_prompt(txt: str, pred, text_prefix: str) -> str:
    pred_str = json.dumps(pred, ensure_ascii=False)
    if text_prefix == "articles":
        return f"""articles: {txt}
Original triples: {pred_str}

For each triple, determine if it is correct by comparing with the articles.
- If all triples are correct: keep them as-is (only apply format normalization like abbreviations, relation simplification).
- If some triples are incorrect: fix only the incorrect ones based on the articles.

Output ONLY JSON array:
[["subject", "relation", "object"], ...]"""

    return f"""Sentence: {txt}
Original triples: {pred_str}

After determining whether the triple is correct, refine the triples: use abbreviations when defined, simplify relations, fix incorrect entities, remove triples not in sentence.

Output ONLY JSON array:
[["subject", "relation", "object"], ...]"""


def verifier_build_extract_user_prompt(txt: str, text_prefix: str) -> str:
    if text_prefix == "articles":
        return f"""articles: {txt}

Extract all subject-relation-object triples from the articles. Use abbreviations when they are defined in parentheses.

Output ONLY JSON array:
[["subject", "relation", "object"], ...]"""

    return f"""Sentence: {txt}

Extract all subject-relation-object triples from the sentence. Use abbreviations when they are defined in parentheses.

Output ONLY JSON array:
[["subject", "relation", "object"], ...]"""


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
        """Validate triples format"""
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
        """Check if triple string is empty"""
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
    
    def _mk_pr_examples(self):
        """Get common examples for refinement prompt"""
        return verifier_get_refine_examples()
    
    def _mk_pr(self, txt, pred, text_prefix="articles"):
        """Build refinement prompt with configurable text prefix"""
        sys = VERIFIER_REFINE_SYSTEM_PROMPT
        examples = self._mk_pr_examples()
        usr = verifier_build_refine_user_prompt(txt, pred, text_prefix)
        
        msgs = [{"role": "system", "content": sys}]
        
        for ex in examples:
            if text_prefix == "articles":
                user_msg = f"articles: {ex['sent']}\nOriginal: {json.dumps(ex['orig'])}"
            else:
                user_msg = f"Sentence: {ex['sent']}\nOriginal: {json.dumps(ex['orig'])}"
            
            msgs.append({"role": "user", "content": user_msg})
            msgs.append({"role": "assistant", "content": json.dumps(ex['refn'], ensure_ascii=False)})
        
        msgs.append({"role": "user", "content": usr})
        
        return msgs
    
    def _mk_ex(self, txt, text_prefix="articles"):
        """Build extraction prompt with configurable text prefix"""
        sys = VERIFIER_EXTRACT_SYSTEM_PROMPT
        usr = verifier_build_extract_user_prompt(txt, text_prefix)
        
        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": usr}
        ]
    
    def _proc_one(self, idx, txt, pstr):
        """Process single sample - must be implemented by subclass"""
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

