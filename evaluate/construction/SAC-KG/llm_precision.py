import time
from typing import Dict, List, Tuple, Optional
from evaluate.construction.common.graph.io import load_lines
from evaluate.construction.common.graph.common import _llm


def _evaluate_triple_from_text(pred_triple: List[str], source_text: str, api_key: Optional[str]) -> Tuple[bool, str]:
    pred_str = f"({pred_triple[0]}, {pred_triple[1]}, {pred_triple[2]})"
    
    prompt = f"""You are evaluating the correctness of a knowledge graph triple extracted from text.

Source Text: "{source_text}"

Extracted Triple: {pred_str}

Determine if the extracted triple is correct. Consider:
1. Are the entities (head and tail) present in the text?
2. Does the text support the stated relation between the entities?
3. Is the triple factually correct based on the text?

First explain your reasoning briefly, then respond with "YES" if correct, or "NO" if incorrect."""

    try:
        response = _llm(prompt, temp=0.1, key=api_key)
        is_correct = "YES" in response.upper()
        return is_correct, response
    except ValueError as e:
        if "NO_KEY" in str(e):
            raise ValueError("OPENROUTER_API_KEY is required for LLM-based evaluation")
        return False, "Error: No API key"
    except Exception as e:
        return False, f"Error: {str(e)}"


def llm_precision(pred_path: str, text_path: str, verbose: bool = True, api_key: Optional[str] = None) -> Dict[str, float]:
    pred_triples = load_lines(pred_path)
    
    with open(text_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    if len(texts) == 0:
        return {"precision": 0.0, "number_of_recalls": 0.0}
    
    if len(pred_triples) == 0:
        return {"precision": 0.0, "number_of_recalls": 0.0}
    
    if len(texts) != len(pred_triples):
        raise ValueError(f"텍스트 수({len(texts)})와 트리플 그룹 수({len(pred_triples)})가 일치하지 않습니다.")
    
    total_correct = 0
    total_pred = 0
    
    if verbose:

        print(f"SAC-KG LLM Precision")
        print(f"{'='*80}\n")
    
    for idx, (text, triples) in enumerate(zip(texts, pred_triples)):
        if verbose:
            print(f"[text {idx+1}]")
            print(f"source text: {text[:100]}{'...' if len(text) > 100 else ''}")
            print(f"extracted triples: {len(triples)} triples\n")
        
        correct_count = 0
        
        for triple_idx, triple in enumerate(triples):
            total_pred += 1
            if triple_idx > 0:
                time.sleep(3.0)
            correct, reason = _evaluate_triple_from_text(triple, text, api_key)
            
            if verbose:
                print(f"  triple {triple_idx+1}: {triple}")
                print(f"  LLM decision: {'correct' if correct else 'incorrect'}")
                print(f"  reason: {reason}")
                print()
            
            if correct:
                correct_count += 1
                total_correct += 1
        
        if verbose:
            print(f"→ text {idx+1} result: {correct_count}/{len(triples)} correct\n")
            print("-" * 80 + "\n")
    
    precision = total_correct / total_pred if total_pred > 0 else 0.0
    avg_recalls_per_text = total_correct / len(texts) if len(texts) > 0 else 0.0
    
    if verbose:
        print(f"final result")
        print(f"total text: {len(texts)}")
        print(f"total predicted triples: {total_pred}")
        print(f"total correct triples: {total_correct}")
        print(f"Precision: {precision:.4f}")
        print(f"average number of recalls per text (Number of Recalls): {avg_recalls_per_text:.2f}")
    
    return {
        "precision": precision,
        "number_of_recalls": avg_recalls_per_text
    }

