from __future__ import annotations

import argparse
import csv
import os
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


def _run_metrix(pred_path: str, gold_path: str, text_path: Optional[str], api_key: Optional[str]) -> Dict[str, Any]:
    g_bleu_mod = load_module("Metrix", "g_bleu")
    g_rouge_mod = load_module("Metrix", "g_rouge")
    g_bertscore_mod = load_module("Metrix", "g_bertscore")
    return {
        "G-BLEU": g_bleu_mod.g_bleu(pred_path, gold_path),
        "G-ROUGE": g_rouge_mod.g_rouge(pred_path, gold_path),
        "G-BERTScore": g_bertscore_mod.g_bertscore(pred_path, gold_path),
    }


MODEL_SPECS: List[ModelSpec] = [
    ModelSpec("metrix", "Metrix", ["metrix", "metric", "mj"], False, _run_metrix),
]

SPEC_BY_KEY = {spec.key: spec for spec in MODEL_SPECS}
ALIAS_TO_KEY = {alias: spec.key for spec in MODEL_SPECS for alias in spec.aliases}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Metrix Evaluation")
    ap.add_argument("--dataset", default=None, help="Dataset directory")
    ap.add_argument("--pred", required=True, help="Predicted triples file")
    ap.add_argument("--gold", required=True, help="Gold triples file")
    ap.add_argument("--models", default="metrix", help="Models to run (default: metrix)")
    ap.add_argument("--log", default=None, help="Log file path")
    ap.add_argument("--analyze-errors", action="store_true", help="Perform error analysis and save to CSV")
    ap.add_argument("--error-output-dir", default="evaluate/construction/error_analysis", help="Directory to save error analysis results")
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
            raise ValueError(f"Unknown model: {part}")
        if spec not in selected:
            selected.append(spec)
    return selected


def _prep_emb(models: List[ModelSpec], args: argparse.Namespace) -> None:
    pass


def _run_model(spec: ModelSpec, dataset_dir: Optional[str], args: argparse.Namespace) -> Dict[str, Any]:
    pred_path, gold_path, text_path = resolve_dataset_paths(
        dataset_dir, args.pred, args.gold, None, require_text=spec.require_text, require_gold=True
    )
    return spec.handler(pred_path, gold_path or "", text_path, None)


def _extr(res: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract values for final results table from results
    
    Returns:
        Result dictionary in specified order
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
    
    return out


def _tbl(data: Dict[str, float]):
    """
    Print final results table
    
    Args:
        data: Metric values
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
    ]
    
    log.info("\n" + "="*120)
    log.info("Final Results Table")
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
    Extract dataset name and model name from path
    
    Args:
        pred_path: Path to prediction file
        gold_path: Path to gold file (optional)
    
    Returns:
        Tuple of (dataset_name, model_name)
    """
    npath = os.path.normpath(os.path.abspath(pred_path))
    parts = npath.split(os.sep)
    
    ds = "Unknown"
    mdl = "Unknown"
    
    # Pattern 1: extract_LLM/{dataset}/{model}/triples.txt
    for i, p in enumerate(parts):
        if p == "extract_LLM" and i + 1 < len(parts):
            ds = parts[i + 1]
            if i + 2 < len(parts):
                mdl = parts[i + 2]
            break
    
    # Pattern 2: evaluate/dataset/{model}/{dataset}/triples.txt (check first)
    if ds == "Unknown":
        for i, p in enumerate(parts):
            if p == "evaluate" and i + 1 < len(parts) and parts[i + 1] == "dataset":
                if i + 2 < len(parts) and i + 3 < len(parts):
                    mdl = parts[i + 2]
                    ds = parts[i + 3]
                    break
    
    # Pattern 3: evaluate/{model}/{dataset}/triples.txt (for baseline.sh, etc.)
    if ds == "Unknown":
        for i, p in enumerate(parts):
            if p == "evaluate" and i + 1 < len(parts) and i + 2 < len(parts):
                next_part = parts[i + 1]
                # Only recognize as model name if not "dataset"
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
    
    return ds, mdl


def _run_error_analysis(models: List[ModelSpec], args: argparse.Namespace, dataset: str, model: str) -> None:
    """
    Perform error analysis
    
    Args:
        models: List of model specs to evaluate
        args: Command-line arguments
        dataset: Dataset name
        model: Model name
    """
    if not args.gold:
        return
    
    pred_path, gold_path, _ = resolve_dataset_paths(
        args.dataset, args.pred, args.gold, None, require_text=False, require_gold=True
    )
    
    output_dir = args.error_output_dir
    
    for spec in models:
        if spec.key == "metrix":
            try:
                gj_analyze_mod = load_module("Metrix", "analyze_errors")
                gj_analyze_mod.analyze_metrix_errors(pred_path, gold_path, output_dir, dataset, model)
                log.info(f"Metrix error analysis completed: {output_dir}/{model}_{dataset}_metrix_errors.csv")
            except Exception as e:
                log.info(f"Metrix error analysis failed: {e}")



def _save_csv(dataset: str, model: str, data: Dict[str, float], csv_path: Optional[str] = None) -> None:
    """
    Save results to CSV file
    
    Args:
        dataset: Dataset name
        model: Model name
        data: Dictionary of metric values
        csv_path: CSV file path (if None, generated based on model name)
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
    
    log.info(f"CSV saved: {csv_file} ({dataset}, {model})")


def main() -> None:
    args = _parse_args()
    
    lgr = log.init(args.log)
    
    try:
        models = _select_models(args.models)
        
        log.info("="*80)
        log.info("Metrix Evaluation Suite")
        log.info(f"Pred file: {args.pred}")
        log.info(f"Gold file: {args.gold}")
        if args.dataset:
            log.info(f"Dataset dir: {args.dataset}")
        log.info(f"Evaluation metrics: {', '.join([m.label for m in models])}")
        log.info("="*80)
        
        _prep_emb(models, args)
        
        all_r = {}
        
        for spec in models:
            log.step(f"{spec.label} evaluation", "start")
            results = _run_model(spec, args.dataset, args)
            
            for key, value in results.items():
                log.res(key, value)
            
            all_r.update(results)
            
            log.done(f"{spec.label} evaluation", "completed")
        
        fdata = _extr(all_r)
        _tbl(fdata)
        
        ds, mdl = _extr_ds_model(args.pred, args.gold)
        _save_csv(ds, mdl, fdata)
        
        if args.analyze_errors:
            _run_error_analysis(models, args, ds, mdl)
        
    finally:
        lgr.close()


if __name__ == "__main__":
    main()