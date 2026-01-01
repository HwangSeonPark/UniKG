from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path

from evaluate.construction.utils.dataset import resolve_dataset_paths
from evaluate.construction.utils.importing import load_module
from evaluate.construction.common import logger as log


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    aliases: List[str]
    require_text: bool
    handler: Callable[[str, str, Optional[str], Optional[str]], Dict[str, Any]]


def _run_edc(pred_path: str, gold_path: str, text_path: Optional[str], api_key: Optional[str]) -> Dict[str, Any]:
    partial_mod = load_module("EDC/webnlg", "partial")
    strict_mod = load_module("EDC/webnlg", "strict")
    exact_mod = load_module("EDC/webnlg", "exact")

    partial_scores = partial_mod.partial(pred_path, gold_path)
    strict_scores = strict_mod.strict(pred_path, gold_path)
    exact_scores = exact_mod.exact(pred_path, gold_path)

    return {
        "Partial": partial_scores,
        "Strict": strict_scores,
        "Exact": exact_scores,
    }


def _run_graphjudge(pred_path: str, gold_path: str, text_path: Optional[str], api_key: Optional[str]) -> Dict[str, Any]:
    g_bleu_mod = load_module("GraphJudge", "g_bleu")
    g_rouge_mod = load_module("GraphJudge", "g_rouge")
    g_bertscore_mod = load_module("GraphJudge", "g_bertscore")
    return {
        "G-BLEU": g_bleu_mod.g_bleu(pred_path, gold_path),
        "G-ROUGE": g_rouge_mod.g_rouge(pred_path, gold_path),
        "G-BERTScore": g_bertscore_mod.g_bertscore(pred_path, gold_path),
    }


MODEL_SPECS: List[ModelSpec] = [
    ModelSpec("edc", "EDC", ["edc"], False, _run_edc),
    ModelSpec("graphjudge", "GraphJudge", ["graphjudge", "graph-judge", "gj"], False, _run_graphjudge),
]

SPEC_BY_KEY = {spec.key: spec for spec in MODEL_SPECS}
ALIAS_TO_KEY = {alias: spec.key for spec in MODEL_SPECS for alias in spec.aliases}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="KG Evaluation")
    ap.add_argument("--dataset", default=None, help="Dataset directory")
    ap.add_argument("--pred", required=True, help="Predicted triples file")
    ap.add_argument("--gold", default=None, help="Gold triples file")
    ap.add_argument("--text", default=None, help="Text file")
    ap.add_argument("--models", default="all", help="Models to run (comma-separated, or 'all')")
    ap.add_argument("--api-key", default=os.getenv("GEMINI_API_KEY"), help="API key")
    ap.add_argument("--log", default=None, help="Log file path")
    return ap.parse_args()


def _select_models(raw: str) -> List[ModelSpec]:
    token = raw.strip().lower()
    if token in ("", "all"):
        return MODEL_SPECS

    selected: List[ModelSpec] = []
    for part in raw.split(","):
        name = part.strip().lower()
        if not name:
            continue
        key = ALIAS_TO_KEY.get(name, name)
        spec = SPEC_BY_KEY.get(key)
        if not spec:
            raise ValueError(f"알 수 없는 모델: {part}")
        if spec not in selected:
            selected.append(spec)
    return selected


def _print_value(name: str, value: Any, indent: int = 0) -> None:
    pad = " " * indent
    if isinstance(value, dict):
        print(f"{pad}{name}:")
        for sub_key, sub_val in value.items():
            _print_value(sub_key, sub_val, indent + 2)
    else:
        if isinstance(value, float):
            display = f"{value:.4f}"
        else:
            display = str(value)
        print(f"{pad}{name}: {display}")


def _prep_emb(models: List[ModelSpec], args: argparse.Namespace) -> None:
    pass


def _run_model(spec: ModelSpec, dataset_dir: Optional[str], args: argparse.Namespace) -> Dict[str, Any]:
    if not args.gold:
        raise ValueError(f"{spec.label} requires --gold path")
    
    pred_path, gold_path, text_path = resolve_dataset_paths(
        dataset_dir, args.pred, args.gold, args.text, require_text=spec.require_text, require_gold=True
    )
    return spec.handler(pred_path, gold_path or "", text_path, args.api_key)


def _extr(res: Dict[str, Any]) -> Dict[str, float]:
    """
    결과에서 최종 결과표용 값 추출
    
    Returns:
        지정 순서에 맞춘 결과 딕셔너리
    """
    out = {}
    
    if "G-BERTScore" in res:
        gbs = res["G-BERTScore"]
        out["G-BERTScore (Accuracy)"] = gbs.get("precision", 0.0)
        out["G-BERTScore (Recall)"] = gbs.get("recall", 0.0)
        out["G-BERTScore (F1)"] = gbs.get("f1", 0.0)
    
    if "G-BLEU" in res:
        gbl = res["G-BLEU"]
        out["G-BLEU (Accuracy)"] = gbl.get("precision", 0.0)
        out["G-BLEU (Recall)"] = gbl.get("recall", 0.0)
        out["G-BLEU (F1)"] = gbl.get("f1", 0.0)
    
    if "G-ROUGE" in res:
        grg = res["G-ROUGE"]
        out["G-ROUGE (Accuracy)"] = grg.get("precision", 0.0)
        out["G-ROUGE (Recall)"] = grg.get("recall", 0.0)
        out["G-ROUGE (F1)"] = grg.get("f1", 0.0)
    
    if "Partial" in res:
        prt = res["Partial"]
        out["Partial (Precision)"] = prt.get("precision", 0.0)
        out["Partial (Recall)"] = prt.get("recall", 0.0)
        out["Partial (F1)"] = prt.get("f1", 0.0)
    
    if "Strict" in res:
        srt = res["Strict"]
        out["Strict (Precision)"] = srt.get("precision", 0.0)
        out["Strict (Recall)"] = srt.get("recall", 0.0)
        out["Strict (F1)"] = srt.get("f1", 0.0)
    
    if "Exact" in res:
        ext = res["Exact"]
        out["Exact (Precision)"] = ext.get("precision", 0.0)
        out["Exact (Recall)"] = ext.get("recall", 0.0)
        out["Exact (F1)"] = ext.get("f1", 0.0)
    
    return out


def _tbl(data: Dict[str, float]):
    """
    최종 결과표 출력
    
    Args:
        data: 지표별 값
    """
    cols = [
        "G-BERTScore (Accuracy)",
        "G-BERTScore (Recall)",
        "G-BERTScore (F1)",
        "G-BLEU (Accuracy)",
        "G-BLEU (Recall)",
        "G-BLEU (F1)",
        "G-ROUGE (Accuracy)",
        "G-ROUGE (Recall)",
        "G-ROUGE (F1)",
        "Partial (Precision)",
        "Partial (Recall)",
        "Partial (F1)",
        "Strict (Precision)",
        "Strict (Recall)",
        "Strict (F1)",
        "Exact (Precision)",
        "Exact (Recall)",
        "Exact (F1)",
    ]
    
    log.info("\n" + "="*120)
    log.info("최종 결과표")
    log.info("="*120)
    
    hdr = "\t".join(cols)
    log.info(hdr)
    log.info("-"*120)
    
    vals = []
    for col in cols:
        if col in data:
            vals.append(f"{data[col]:.4f}")
        else:
            vals.append("N/A")
    
    row = "\t".join(vals)
    log.info(row)
    log.info("="*120)


def _extr_ds_model(pred_path: str, gold_path: Optional[str]) -> tuple[str, str]:
    """
    경로에서 데이터셋 이름과 모델 이름 추출
    
    Args:
        pred_path: 예측 파일 경로
        gold_path: 정답 파일 경로 (선택)
    
    Returns:
        (데이터셋명, 모델명) 튜플
    """
    npath = os.path.normpath(os.path.abspath(pred_path))
    parts = npath.split(os.sep)
    
    ds = "Unknown"
    mdl = "Unknown"
    
    # 패턴 1: extract_LLM/{dataset}/{model}/triples.txt
    for i, p in enumerate(parts):
        if p == "extract_LLM" and i + 1 < len(parts):
            ds = parts[i + 1]
            if i + 2 < len(parts):
                mdl = parts[i + 2]
            break
    
    # 패턴 2: evaluate/dataset/{model}/{dataset}/triples.txt (우선 확인)
    if ds == "Unknown":
        for i, p in enumerate(parts):
            if p == "evaluate" and i + 1 < len(parts) and parts[i + 1] == "dataset":
                if i + 2 < len(parts) and i + 3 < len(parts):
                    mdl = parts[i + 2]
                    ds = parts[i + 3]
                    break
    
    # 패턴 3: evaluate/{model}/{dataset}/triples.txt (baseline.sh 등)
    if ds == "Unknown":
        for i, p in enumerate(parts):
            if p == "evaluate" and i + 1 < len(parts) and i + 2 < len(parts):
                next_part = parts[i + 1]
                # "dataset"이 아닌 경우에만 모델명으로 인식
                if next_part != "dataset":
                    mdl = next_part
                    ds = parts[i + 2]
                    break
    
    dsmap = {
        "GenWiki": "GenWiki",
        "GenWiki-Hard": "GenWiki",
        "CaRB": "CaRB",
        "CaRB-Expert": "CaRB",
        "KELM-sub": "KELM-sub",
        "kelm_sub": "KELM-sub",
        "SCIERC": "SCIERC",
        "webnlg20": "webnlg20",
    }
    ds = dsmap.get(ds, ds)
    
    mmap = {
        "gpt-5-mini": "gpt-5-mini",
        "gpt-5": "gpt-5.1",
        "gpt-5.1": "gpt-5.1",
        "qwen": "qwen",
        "mistral": "mistral",
        "llama-8b": "llama-8b",
        "gj": "gj",
        "edc": "edc",
        "kggen": "KGGen",
        "kggen_c": "KGGEN_c",
        "rakg": "RAKG",
        "rakg_c": "RAKG_c",
    }
    mdl = mmap.get(mdl.lower(), mdl)
    
    return ds, mdl


def _save_csv(dataset: str, model: str, data: Dict[str, float], csv_path: Optional[str] = None) -> None:
    """
    결과를 CSV 파일에 저장
    
    Args:
        dataset: 데이터셋 이름
        model: 모델 이름
        data: 메트릭 값 딕셔너리
        csv_path: CSV 파일 경로 (None이면 모델명 기반으로 생성)
    """
    if csv_path is None:
        csv_path = f"evaluate/construction/{model}_result.csv"
    
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    
    cols = [
        "G-BERTScore (Accuracy)",
        "G-BERTScore (Recall)",
        "G-BERTScore (F1)",
        "G-BLEU (Accuracy)",
        "G-BLEU (Recall)",
        "G-BLEU (F1)",
        "G-ROUGE (Accuracy)",
        "G-ROUGE (Recall)",
        "G-ROUGE (F1)",
        "Partial (Precision)",
        "Partial (Recall)",
        "Partial (F1)",
        "Strict (Precision)",
        "Strict (Recall)",
        "Strict (F1)",
        "Exact (Precision)",
        "Exact (Recall)",
        "Exact (F1)"
    ]
    
    rows = []
    if csv_file.exists():
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
        except Exception:
            rows = []
    
    nrow = {"dataset": dataset, "model": model}
    for col in cols:
        if col in data:
            nrow[col] = f"{data[col]:.4f}"
        else:
            nrow[col] = "0.0000"
    
    upd = False
    for i, row in enumerate(rows):
        if row.get("dataset") == dataset and row.get("model") == model:
            rows[i] = nrow
            upd = True
            break
    
    if not upd:
        rows.append(nrow)
    
    flds = ["dataset", "model"] + cols
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        wrtr = csv.DictWriter(f, fieldnames=flds)
        wrtr.writeheader()
        wrtr.writerows(rows)
    
    log.info(f"CSV 저장 완료: {csv_file} ({dataset}, {model})")


def main() -> None:
    args = _parse_args()
    
    lgr = log.init(args.log)
    
    try:
        models = _select_models(args.models)
        
        log.info("="*80)
        log.info("Knowledge Graph Construction Evaluation Suite")
        log.info(f"Pred file: {args.pred}")
        if args.gold:
            log.info(f"Gold file: {args.gold}")
        if args.text:
            log.info(f"Text file: {args.text}")
        if args.dataset:
            log.info(f"Dataset dir: {args.dataset}")
        log.info(f"평가 메트릭: {', '.join([m.label for m in models])}")
        log.info("="*80)
        
        _prep_emb(models, args)
        
        all_r = {}
        
        for spec in models:
            log.step(f"{spec.label} 평가", "시작")
            results = _run_model(spec, args.dataset, args)
            
            for key, value in results.items():
                log.res(key, value)
            
            all_r.update(results)
            
            log.done(f"{spec.label} 평가", "완료")
        
        fdata = _extr(all_r)
        _tbl(fdata)
        
        ds, mdl = _extr_ds_model(args.pred, args.gold)
        _save_csv(ds, mdl, fdata)
        
    finally:
        lgr.close()


if __name__ == "__main__":
    main()