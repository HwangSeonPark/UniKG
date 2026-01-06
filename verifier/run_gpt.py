import os
import json
import ast
from refiner_gpt import TpRef

def map_articles_to_triples(articles_path, triples_path, output_path=None, dataset_name=None, actual_dir=None):

    if not os.path.exists(articles_path):
        print(f"Warning: {articles_path} not found")
        return {}
    
    if not os.path.exists(triples_path):
        print(f"Warning: {triples_path} not found")
        return {}
    
    # 아티클 읽기
    with open(articles_path, 'r', encoding='utf-8') as f:
        articles = [l.strip() for l in f.readlines()]
    
    # 트리플 읽기
    with open(triples_path, 'r', encoding='utf-8') as f:
        triples_lines = [l.strip() for l in f.readlines()]
    
    # 매핑 생성
    mapping = {}
    min_len = min(len(articles), len(triples_lines))
    
    for idx in range(min_len):
        article = articles[idx]
        triple_str = triples_lines[idx]
        
        # 트리플 파싱
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
    
    # 메타데이터 추가
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
    import sys
    
    # Parse command line arguments
    if len(sys.argv) < 4:
        print("Usage: python run_gpt.py <dataset_name> <input_dir> <output_dir>")
        sys.exit(1)
    
    dset_nm = sys.argv[1]
    input_dir = sys.argv[2]
    output_dir = sys.argv[3]
    
    # Use GPT-5.1 (API key loaded from environment variable OPENAI_API_KEY)
    ref = TpRef(max_workers=5)
    
    # Read articles from input folder or file
    if os.path.isfile(input_dir):
        src_p = input_dir
    else:
        src_p = os.path.join(input_dir, "articles.txt")
    
    # Read extract_triples.txt
    input_triples_p = os.path.join(output_dir, dset_nm, "extract_triples.txt")
    
    if not os.path.exists(src_p):
        print(f"Skip: {src_p} not found")
        return
    
    if not os.path.exists(input_triples_p):
        print(f"Skip: {input_triples_p} not found (run extract_gpt.py first)")
        return
    
    # Output directory
    out_dir = os.path.join(output_dir, dset_nm)
    os.makedirs(out_dir, exist_ok=True)
    
    with open(src_p, 'r', encoding='utf-8') as f:
        txts = [l.strip() for l in f.readlines()]
    
    # Read extract_triples.txt
    with open(input_triples_p, 'r', encoding='utf-8') as f:
        preds = [l.strip() for l in f.readlines()]
    
    outs = ref.proc_batch(txts, preds)
    
    # Output file for refined triples
    refined_p = os.path.join(out_dir, "refined_triples.txt")
    final = [str(o) for o in outs]
    with open(refined_p, 'w', encoding='utf-8') as f:
        f.write('\n'.join(final))

if __name__ == "__main__":
    main()
