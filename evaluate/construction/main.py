from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from evaluate.construction.common.graph.io import load_flat
from evaluate.construction.utils.dataset import resolve_dataset_paths
from evaluate.construction.utils.importing import load_module


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    aliases: List[str]
    require_text: bool
    handler: Callable[[str, str, Optional[str], Optional[str]], Dict[str, Any]]


def _run_tree_kg(pred_path: str, gold_path: str, text_path: Optional[str], api_key: Optional[str]) -> Dict[str, Any]:
    gt = load_flat(gold_path)
    pd = load_flat(pred_path)
    er_mod = load_module("Tree-KG", "er")
    pc_mod = load_module("Tree-KG", "pc")
    mec_mod = load_module("Tree-KG", "mec")
    rs_mod = load_module("Tree-KG", "rs")

    results: Dict[str, Any] = {
        "Entity Recall (ER)": er_mod.er(gt, pd, key=api_key),
        "Precision (PC)": pc_mod.pc(gt, pd, key=api_key),
        "Mapping-based Edge Connectivity (MEC)": mec_mod.mec(gt, pd, key=api_key),
    }

    rs_score = rs_mod.rs(pd, key=api_key)
    results["Relation Strength (RS)"] = rs_score
    return results


def _run_edc(pred_path: str, gold_path: str, text_path: Optional[str], api_key: Optional[str]) -> Dict[str, Any]:
    partial_mod = load_module("EDC/webnlg", "partial")
    strict_mod = load_module("EDC/webnlg", "strict")
    exact_mod = load_module("EDC/webnlg", "exact")
    exact_triple_mod = load_module("EDC", "exact_triple")

    partial_scores = partial_mod.partial(pred_path, gold_path)
    strict_scores = strict_mod.strict(pred_path, gold_path)
    exact_scores = exact_mod.exact(pred_path, gold_path)
    et_p, et_r, et_f = exact_triple_mod.f1(pred_path, gold_path)

    return {
        "Partial": partial_scores,
        "Strict": strict_scores,
        "Exact": exact_scores,
        "Exact Triple": {"precision": et_p, "recall": et_r, "f1": et_f},
    }


def _run_sac(pred_path: str, gold_path: str, text_path: Optional[str], api_key: Optional[str]) -> Dict[str, Any]:
    if not text_path:
        raise ValueError("SAC-KG 평가는 원문 텍스트 경로가 필요합니다.")
    llm_mod = load_module("SAC-KG", "llm_precision")
    scores = llm_mod.llm_precision(pred_path, text_path, verbose=True, api_key=api_key)
    return {
        "Precision": scores["precision"],
        "Number of Recalls": scores["number_of_recalls"],
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


def _run_pive(pred_path: str, gold_path: str, text_path: Optional[str], api_key: Optional[str]) -> Dict[str, Any]:
    triple_mod = load_module("PiVe", "triple_f1")
    graph_mod = load_module("PiVe", "graph_f1")
    ged_mod = load_module("PiVe", "ged")
    g_bertscore_mod = load_module("GraphJudge", "g_bertscore")

    triple_scores = triple_mod.triple_f1(pred_path, gold_path)
    graph_scores = graph_mod.graph_f1(pred_path, gold_path)
    ged_score = ged_mod.ged(pred_path, gold_path)
    gbs_scores = g_bertscore_mod.g_bertscore(pred_path, gold_path)

    accuracy = graph_scores.get("accuracy", 0.0)
    graph_block = {
        "G-Precision": accuracy,
        "G-Recall": accuracy,
        "G-F1": accuracy,
    }

    return {
        "Triple Match": {
            "T-Precision": triple_scores["precision"],
            "T-Recall": triple_scores["recall"],
            "T-F1": triple_scores["f1"],
        },
        "Graph Match": graph_block,
        "G-BERTScore (G-BS)": gbs_scores,
        "Graph Edit Distance (GED)": ged_score["ged"],
    }


MODEL_SPECS: List[ModelSpec] = [
    ModelSpec("tree", "Tree-KG", ["tree", "tree-kg", "treekg"], False, _run_tree_kg),
    ModelSpec("edc", "EDC", ["edc"], False, _run_edc),
    ModelSpec("sac", "SAC-KG", ["sac", "sac-kg", "sackg"], True, _run_sac),
    ModelSpec("graphjudge", "GraphJudge", ["graphjudge", "graph-judge", "gj"], False, _run_graphjudge),
    ModelSpec("pive", "PiVe", ["pive"], False, _run_pive),
]

SPEC_BY_KEY = {spec.key: spec for spec in MODEL_SPECS}
ALIAS_TO_KEY = {alias: spec.key for spec in MODEL_SPECS for alias in spec.aliases}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="선행연구별 KG 메트릭 실행기")
    ap.add_argument("--dataset", default=None, help="데이터셋 디렉터리 (선택 사항, 파일명만 지정 시 사용)")
    ap.add_argument("--pred", required=True, help="예측 트리플 파일 경로 (필수)")
    ap.add_argument("--gold", default=None, help="정답 트리플 파일 경로 (SAC-KG 제외한 모델에서 필수)")
    ap.add_argument("--text", default=None, help="원문 텍스트 파일 경로 (SAC-KG 등에서 필요)")
    ap.add_argument("--models", default="all", help="실행할 모델 (콤마 구분, all 사용 가능)")
    ap.add_argument("--api-key", default=os.getenv("GEMINI_API_KEY"), help="LLM 메트릭용 Gemini API 키")
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
        print(f"{pad}[{name}]")
        for sub_key, sub_val in value.items():
            _print_value(sub_key, sub_val, indent + 2)
    else:
        if value is None:
            display = "API 키 필요"
            sys.exit(1)
        elif isinstance(value, float):
            display = f"{value:.4f}"
        else:
            display = str(value)
        print(f"{pad}{name}: {display}")


def _run_model(spec: ModelSpec, dataset_dir: Optional[str], args: argparse.Namespace) -> Dict[str, Any]:
    require_gold = spec.key != "sac"
    if require_gold and not args.gold:
        raise ValueError(f"{spec.label} 평가는 --gold 경로가 필요합니다.")
    
    pred_path, gold_path, text_path = resolve_dataset_paths(
        dataset_dir, args.pred, args.gold, args.text, require_text=spec.require_text, require_gold=require_gold
    )
    return spec.handler(pred_path, gold_path or "", text_path, args.api_key)


def main() -> None:
    args = _parse_args()
    models = _select_models(args.models)

    print("Knowledge Graph Construction Evaluation Suite")
    print(f"Pred file: {args.pred}")
    if args.gold:
        print(f"Gold file: {args.gold}")
    if args.text:
        print(f"Text file: {args.text}")
    if args.dataset:
        print(f"Dataset dir: {args.dataset}")
    print("-" * 80)

    for spec in models:
        print(f"\n[{spec.label}]")
        print("-" * 40)
        results = _run_model(spec, args.dataset, args)
        for key, value in results.items():
            _print_value(key, value, indent=2)


if __name__ == "__main__":
    main()