import os
import json
import ast
import sys
from verifier import TpRef


# Paths will be passed as arguments

# Dataset mapping (canonical dataset names)
DATASET_MAPPING = {
    "webnlg20": "webnlg20",
    "carb-expert": "carb-expert",
    "kelm_sub": "kelm_sub",
    "genwiki-hard": "genwiki-hard",
    "scierc": "scierc",
}


DEFAULT_PORT = int(os.getenv("REFINER_PORT"))
DEFAULT_MAX_WORKERS = int(os.getenv("REFINER_MAX_WORKERS"))
DEFAULT_MODEL = os.getenv("REFINER_MODEL")
DEFAULT_MAX_TOKENS = int(os.getenv("REFINER_MAX_TOKENS"))
DEFAULT_HOST = os.getenv("REFINER_HOST", "localhost")

def dedup_row(row, canon):
    """Deduplicate triples within a row (case-insensitive) and reuse canonical casing across rows."""
    if not row or not isinstance(row, list):
        return []
    seen = set()
    out = []
    for tp in row:
        if not isinstance(tp, (list, tuple)) or len(tp) != 3:
            continue
        s, p, o = tp
        s = s if isinstance(s, str) else (str(s) if s is not None else "")
        p = p if isinstance(p, str) else (str(p) if p is not None else "")
        o = o if isinstance(o, str) else (str(o) if o is not None else "")
        key = (s.lower(), p.lower(), o.lower())
        if key in seen:
            continue
        seen.add(key)
        if key not in canon:
            canon[key] = [s, p, o]
        out.append(canon[key])
    return out

def map_articles_to_triples(articles_path, triples_path, output_path=None, dataset_name=None, actual_dir=None):
    if not os.path.exists(articles_path):
        print(f"Warning: {articles_path} not found")
        return {}
    
    if not os.path.exists(triples_path):
        print(f"Warning: {triples_path} not found")
        return {}
    
    # Read articles
    with open(articles_path, 'r', encoding='utf-8') as f:
        articles = [l.strip() for l in f.readlines()]
    
    # Read triples
    with open(triples_path, 'r', encoding='utf-8') as f:
        triples_lines = [l.strip() for l in f.readlines()]
    
    # Create mapping
    mapping = {}
    min_len = min(len(articles), len(triples_lines))
    
    for idx in range(min_len):
        article = articles[idx]
        triple_str = triples_lines[idx]
        
        # Parse triples
        try:
            if triple_str:
                triples = ast.literal_eval(triple_str)
                if not isinstance(triples, list):
                    triples = []
            else:
                triples = []
        except (ValueError, SyntaxError):
            triples = []
        
        mapping[idx] = {
            "article": article,
            "triples": triples,
            "num_triples": len(triples)
        }
    
    # Add metadata
    metadata = {
        "dataset_name": dataset_name,
        "actual_directory": actual_dir,
        "articles_path": articles_path,
        "triples_path": triples_path,
        "total_articles": len(mapping)
    }
    
    # Save to output file
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_data = {
            "metadata": metadata,
            "mapping": mapping
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Mapping saved to: {output_path}")
    
    return mapping

def get_article_for_triple(mapping, triple, article_index=None):

    results = []
    
    if article_index is not None:
        # Search within a specific index only
        if article_index in mapping:
            article_data = mapping[article_index]
            if triple in article_data["triples"]:
                results.append({
                    "index": article_index,
                    "article": article_data["article"],
                    "triple": triple
                })
    else:
        # Search the full mapping
        for idx, article_data in mapping.items():
            if triple in article_data["triples"]:
                results.append({
                    "index": idx,
                    "article": article_data["article"],
                    "triple": triple
                })
    
    return results

def main():
    # Parse command line arguments
    if len(sys.argv) < 5:
        print("Usage: python run.py <model_name> <dataset_name> <input_dir> <output_dir>")
        sys.exit(1)
    
    mdl_nm = sys.argv[1]
    dset_nm = sys.argv[2]
    input_dir = sys.argv[3]
    output_dir = sys.argv[4]
    
    ref = TpRef(host=DEFAULT_HOST, port=DEFAULT_PORT, max_workers=DEFAULT_MAX_WORKERS,
                model=DEFAULT_MODEL, max_tokens=DEFAULT_MAX_TOKENS)
    
    # Read articles from input folder or file
    if os.path.isfile(input_dir):
        src_p = input_dir
    else:
        src_p = os.path.join(input_dir, "articles.txt")
    
    # Read extract_triples.txt
    prd_p = os.path.join(output_dir, dset_nm, "extract_triples.txt")
    
    if not os.path.exists(src_p):
        print(f"Skip: {src_p} not found")
        return
    if not os.path.exists(prd_p):
        print(f"Skip: {prd_p} not found")
        return
    
    with open(src_p, 'r', encoding='utf-8') as f:
        txts = [l.strip() for l in f.readlines()]
    
    with open(prd_p, 'r', encoding='utf-8') as f:
        preds = [l.strip() for l in f.readlines()]
    
    lim = int(os.getenv("LIM", "0") or "0")
    if lim > 0:
        lim = min(lim, len(txts), len(preds))
        txts = txts[:lim]
        preds = preds[:lim]
    
    # Output directory
    out_dir = os.path.join(output_dir, dset_nm)
    os.makedirs(out_dir, exist_ok=True)
    
    # Output file for refined triples
    refined_p = os.path.join(out_dir, "refined_triples.txt")
    
    # Process and save refined triples
    outs = ref.proc_batch(txts, preds)
    canon = {}
    
    with open(refined_p, 'w', encoding='utf-8') as f:
        for tps in outs:
            f.write(str(dedup_row(tps, canon)) + '\n')

if __name__ == "__main__":
    main()
