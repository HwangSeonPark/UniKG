from typing import Dict, Any
import statistics
from .common import convert_to_xml, getRefs, getCands, calculateAllScores, calculateSystemScore


def _agg(selected, selected_per, key: str) -> Dict[str, Any]:
    if not selected:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "per_tag": {}}
    p = statistics.mean([x[key]["precision"] for x in selected])
    r = statistics.mean([x[key]["recall"] for x in selected])
    f = statistics.mean([x[key]["f1"] for x in selected])
    tags = {}
    for tag in ["SUB", "PRED", "OBJ"]:
        try:
            tp = statistics.mean([x[tag][key]["precision"] for x in selected_per])
            tr = statistics.mean([x[tag][key]["recall"] for x in selected_per])
            tf = statistics.mean([x[tag][key]["f1"] for x in selected_per])
            tags[tag] = {"precision": tp, "recall": tr, "f1": tf}
        except Exception:
            tags[tag] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {"precision": p, "recall": r, "f1": f, "per_tag": tags}


def ent_type(pred_path: str, gold_path: str) -> Dict[str, Any]:
    pred_xml_path, gold_xml_path = convert_to_xml(pred_path, gold_path, max_length_diff=None)
    reflist, newreflist = getRefs(gold_xml_path)
    candlist, newcandlist = getCands(pred_xml_path)
    tot, tot_tag = calculateAllScores(newreflist, newcandlist)
    sel, sel_tag, _, _ = calculateSystemScore(tot, tot_tag, newreflist, newcandlist)
    return _agg(sel, sel_tag, "ent_type")


