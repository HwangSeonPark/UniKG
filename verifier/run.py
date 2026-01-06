import os
import json
import ast
import sys
from refiner import TpRef


# Paths will be passed as arguments

# Dataset mapping (for reference)
DATASET_MAPPING = {
    "webnlg20": "webnlg20",
    "CaRB": "CaRB-Expert",
    "KELM-sub": "kelm_sub",
    "GenWiki": "GenWiki-Hard",
    "SCIERC": "SCIERC"
}


DEFAULT_PORT = int(os.getenv("REFINER_PORT"))
DEFAULT_MAX_WORKERS = int(os.getenv("REFINER_MAX_WORKERS"))
DEFAULT_MODEL = os.getenv("REFINER_MODEL")
DEFAULT_MAX_TOKENS = int(os.getenv("REFINER_MAX_TOKENS"))
DEFAULT_HOST = os.getenv("REFINER_HOST", "localhost")

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
    
    # 출력 파일에 저장
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
        # 특정 인덱스에서만 검색
        if article_index in mapping:
            article_data = mapping[article_index]
            if triple in article_data["triples"]:
                results.append({
                    "index": article_index,
                    "article": article_data["article"],
                    "triple": triple
                })
    else:
        # 전체 매핑에서 검색
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
    
    # Output directory
    out_dir = os.path.join(output_dir, dset_nm)
    os.makedirs(out_dir, exist_ok=True)
    
    # Output file for refined triples
    refined_p = os.path.join(out_dir, "refined_triples.txt")
    
    # Process and save refined triples
    outs = ref.proc_batch(txts, preds)
    
    with open(refined_p, 'w', encoding='utf-8') as f:
        for tps in outs:
            f.write(str(tps) + '\n')

if __name__ == "__main__":
    main()
