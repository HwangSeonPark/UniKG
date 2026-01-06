import json
import ast


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
        ex1 = {
            "sent": "This paper presents a digital signal processor (DSP) implementation used for real-time statistical voice conversion (VC) systems.",
            "orig": [["digital signal processor -lrb- dsp -rrb-", "implementation used for", "real-time statistical voice conversion -lrb- vc -rrb-"]],
            "refn": [["dsp", "used for", "real-time vc"]]
        }
        
        ex2 = {
            "sent": "We propose a method for automatic estimation of word significance oriented for speech-based information retrieval.",
            "orig": [["method", "used for", "paraphrase"]],
            "refn": [["method", "used for", "automatic estimation of word significance"]]
        }
        
        ex3 = {
            "sent": "The statistical machine learning algorithm is used for discovering smallest meaning-bearing units of language called morphemes in tasks like speech and text understanding, machine translation, information retrieval, and statistical language modeling.",
            "orig": [["speech and text understanding", "hyponym of", "tasks"], ["machine translation", "hyponym of", "tasks"], ["information retrieval", "hyponym of", "tasks"]],
            "refn": [["speech and text understanding", "conjunction", "machine translation"], ["machine translation", "conjunction", "information retrieval"], ["information retrieval", "conjunction", "statistical language modeling"]]
        }
        
        ex4 = {
            "sent": "Models are used for summary generation and various tasks that are part of SUMMAC-1 evaluation.",
            "orig": [["automatic summarization", "conjunction", "information extraction"], ["models", "used for", "summary generation"], ["positive feature vectors", "used for", "categorization task"], ["categorization task", "evaluate for", "system"]],
            "refn": [["models", "used for", "summary generation"], ["categorization task", "evaluate for", "system"]]
        }
        
        ex5 = {
            "sent": "Natural language processing using structured inheritance network paradigm and operations are used for SI-nets conceptual system.",
            "orig": [["natural language", "used for", "structured inheritance network paradigm"], ["operations", "used for", "si-nets"], ["nl", "used for", "conceptual system"]],
            "refn": [["nl", "used for", "si-nets"], ["operations", "used for", "si-nets"]]
        }
        
        ex6 = {
            "sent": "After the fall of communism in 1989 , Aromanians , Romanians and Vlachs have started initiatives to organize themselves under one common association .",
            "orig": [['aromanians', 'initiative', 'organize themselves under one common association'], ['romanians', 'initiative', 'organize themselves under one common association'], ['vlachs', 'initiative', 'organize themselves under one common association']],
            "refn": [['Aromanians', 'related', 'Vlachs'], ['Romanians', 'related', 'Aromanians'], ['Aromanians', 'related', 'Romanians'], ['Aromanians', 'death Year', '1989'], ['Aromanians', 'ethnic Group', 'Communist Party']]
        }
        
        ex7 = {
            "sent": "Kinaloor is a village in Kozhikode District in the state of Kerala , India . Kinalur is situated near by Vattoli in Thamarassery - Balussery State Highway.",
            "orig": [['kinaloor', 'located in the administrative territorial entity', 'kozhikode district'], ['kinaloor', 'state', 'kerala'], ['kinaloor', 'country', 'india'], ['kinaloor', 'nearby place', 'vattoli'], ['vattoli', 'nearby place', 'thamarassery - balussery state highway']],
            "refn": [['Kinalur', 'isPartOf', 'Kerala'], ['Kinalur', 'settlementType', 'village'], ['Kinalur', 'country', 'India'], ['Kinalur', 'isPartOf', 'Kozhikode District'], ['Kinalur', 'nearest City', 'Vattoli'], ['Kinalur', 'is Part Of', 'Thamarassery - Balussery State Highway']]
        }
        
        return [ex1, ex2, ex3, ex4, ex5, ex6, ex7]
    
    def _mk_pr(self, txt, pred, text_prefix="articles"):
        """Build refinement prompt with configurable text prefix"""
        sys = "You are an expert in refining knowledge graph triples. Keep entities concise, use standard abbreviations, and match exact text phrasing."
        
        examples = self._mk_pr_examples()
        
        if text_prefix == "articles":
            usr = f"""articles: {txt}
Original triples: {json.dumps(pred, ensure_ascii=False)}

For each triple, determine if it is correct by comparing with the articles. 
- If all triples are correct: keep them as-is (only apply format normalization like abbreviations, relation simplification).
- If some triples are incorrect: **fix only the incorrect ones based on the articles**.

Output ONLY JSON array:
[["subject", "relation", "object"], ...]"""
        else:
            usr = f"""Sentence: {txt}
Original triples: {json.dumps(pred, ensure_ascii=False)}

After determining whether the triple is correct, Refine the triples: use abbreviations when defined, simplify relations, fix incorrect entities, remove triples not in sentence.
Output ONLY JSON array:
[["subject", "relation", "object"], ...]"""
        
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
        sys = "You are an expert in extracting knowledge graph triples from scientific text. Extract concise triples using abbreviations when defined."
        
        if text_prefix == "articles":
            usr = f"""articles: {txt}


Extract all subject-relation-object triples from the articles. Use abbreviations when they are defined in parentheses.


Output ONLY JSON array:
[["subject", "relation", "object"], ...]"""
        else:
            usr = f"""Sentence: {txt}

Extract all subject-relation-object triples from the sentence. Use abbreviations when they are defined in parentheses.
Output ONLY JSON array:
[["subject", "relation", "object"], ...]"""
        
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

