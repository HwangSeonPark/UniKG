from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from evaluate.construction.common.graph.io import load_flat
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


def _run_tree_kg(pred_path: str, gold_path: str, text_path: Optional[str], api_key: Optional[str]) -> Dict[str, Any]:
    gt = load_flat(gold_path)
    pd = load_flat(pred_path)
    
    # 통합 평가 (임베딩 자동 최적화)
    eval_mod = load_module("Tree-KG", "eval_all")
    res = eval_mod.eval(gt, pd, key=api_key, prep=True)
    
    # 결과 포맷팅
    results: Dict[str, Any] = {
        "Entity Recall (ER)": res.get("er", 0.0),
        "Precision (PC)": res.get("pc", 0.0),
        "F1 Score (F1)": res.get("f1", 0.0),
        "Mapping-based Edge Connectivity (MEC)": res.get("mec", 0.0),
    }
    
    # RS는 키 없으면 None
    if "rs" in res:
        results["Relation Strength (RS)"] = res["rs"]
    
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
    ap.add_argument("--log", default=None, help="로그 파일 경로 (선택, 미지정 시 logs/ 디렉터리에 자동 생성)")
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


def _prep_emb(models: List[ModelSpec], args: argparse.Namespace) -> None:
    """
    평가 전에 임베딩 사전 로드 (효율성 향상)
    - bert-base-uncased 사용 메트릭: GraphJudge, PiVe, Tree-KG
    - 모델 조합에 따라 최적화
    """
    mkeys = {m.key for m in models}
    need_emb = {"tree", "graphjudge", "pive"} & mkeys
    
    # 임베딩 필요한 메트릭이 없으면 스킵
    if not need_emb:
        log.info("임베딩 사전 로드 불필요 (임베딩 사용 메트릭 없음)")
        return
    
    # Tree-KG용 엔티티 사전 로드
    if "tree" in need_emb and args.gold:
        try:
            log.step("임베딩 사전 로드", "Tree-KG 엔티티 추출 및 임베딩")
            gt = load_flat(args.gold)
            pd = load_flat(args.pred)
            
            # 엔티티 개수 추정
            from evaluate.construction.common.preload import _exte
            gtes = _exte(gt)
            pdes = _exte(pd)
            ntot = len(gtes | pdes)
            log.est(ntot, "개 엔티티")
            
            prep_mod = load_module("common", "preload")
            prep_mod.prep_tree(gt, pd)
            log.done("임베딩 사전 로드", f"{ntot}개 엔티티 완료")
        except Exception as e:
            log.info(f"임베딩 사전 로드 실패: {e}")


def _run_model(spec: ModelSpec, dataset_dir: Optional[str], args: argparse.Namespace) -> Dict[str, Any]:
    require_gold = spec.key != "sac"
    if require_gold and not args.gold:
        raise ValueError(f"{spec.label} 평가는 --gold 경로가 필요합니다.")
    
    pred_path, gold_path, text_path = resolve_dataset_paths(
        dataset_dir, args.pred, args.gold, args.text, require_text=spec.require_text, require_gold=require_gold
    )
    return spec.handler(pred_path, gold_path or "", text_path, args.api_key)


def _extr(res: Dict[str, Any]) -> Dict[str, float]:
    """
    결과에서 최종 결과표용 값 추출
    
    Returns:
        지정 순서에 맞춘 결과 딕셔너리
    """
    out = {}
    
    # GraphJudge 결과 추출 (precision이 accuracy 역할)
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
    
    # PiVe 결과 추출
    if "Triple Match" in res:
        tm = res["Triple Match"]
        out["Triple Match (T-Recall)"] = tm.get("T-Recall", 0.0)
        out["Triple Match (T-Precision)"] = tm.get("T-Precision", 0.0)
        out["Triple Match F1 (T-F1)"] = tm.get("T-F1", 0.0)
    
    if "Graph Match" in res:
        gm = res["Graph Match"]
        out["Graph Match (G-Recall)"] = gm.get("G-Recall", 0.0)
        out["Graph Match (G-Precision)"] = gm.get("G-Precision", 0.0)
        out["Graph Match F1 (G-F1)"] = gm.get("G-F1", 0.0)
    
    if "G-BERTScore (G-BS)" in res:
        gbs2 = res["G-BERTScore (G-BS)"]
        out["G-BERTScore (G-BS)"] = gbs2.get("precision", 0.0)
    
    if "Graph Edit Distance (GED)" in res:
        out["Graph Edit Distance (GED)"] = res["Graph Edit Distance (GED)"]
    
    # EDC 결과 추출
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
    
    if "Exact Triple" in res:
        et = res["Exact Triple"]
        out["Exact Triple (Precision)"] = et.get("precision", 0.0)
        out["Exact Triple (Recall)"] = et.get("recall", 0.0)
        out["Exact Triple (F1)"] = et.get("f1", 0.0)
    
    return out


def _tbl(data: Dict[str, float]):
    """
    최종 결과표 출력
    
    Args:
        data: 지표별 값
    """
    # 지정된 순서
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
        "Triple Match (T-Recall)",
        "Triple Match (T-Precision)",
        "Triple Match F1 (T-F1)",
        "Graph Match (G-Recall)",
        "Graph Match (G-Precision)",
        "Graph Match F1 (G-F1)",
        "G-BERTScore (G-BS)",
        "Graph Edit Distance (GED)",
        "Partial (Precision)",
        "Partial (Recall)",
        "Partial (F1)",
        "Strict (Precision)",
        "Strict (Recall)",
        "Strict (F1)",
        "Exact (Precision)",
        "Exact (Recall)",
        "Exact (F1)",
        "Exact Triple (Precision)",
        "Exact Triple (Recall)",
        "Exact Triple (F1)",
    ]
    
    log.info("\n" + "="*120)
    log.info("최종 결과표")
    log.info("="*120)
    
    # 헤더 출력
    hdr = "\t".join(cols)
    log.info(hdr)
    log.info("-"*120)
    
    # 데이터 출력
    vals = []
    for col in cols:
        if col in data:
            vals.append(f"{data[col]:.4f}")
        else:
            vals.append("N/A")
    
    row = "\t".join(vals)
    log.info(row)
    log.info("="*120)


def main() -> None:
    args = _parse_args()
    
    # 로거 초기화
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
        
        # 임베딩 사전 로드 최적화
        _prep_emb(models, args)
        
        # 전체 결과 저장
        all_r = {}
        
        # 각 메트릭 실행
        for spec in models:
            log.step(f"{spec.label} 평가", "시작")
            results = _run_model(spec, args.dataset, args)
            
            # 결과 출력
            for key, value in results.items():
                log.res(key, value)
            
            # 전체 결과에 병합
            all_r.update(results)
            
            log.done(f"{spec.label} 평가", "완료")
        
        # 최종 결과표 출력
        fdata = _extr(all_r)
        _tbl(fdata)
        
    finally:
        # 로거 종료
        lgr.close()


if __name__ == "__main__":
    main()