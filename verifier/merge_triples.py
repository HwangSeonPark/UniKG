
import json
import sys
import os
import ast
from pathlib import Path


def merge_triples(input_dir: str, output_dir: str, dataset_name: str):

    refined_file = os.path.join(output_dir, dataset_name, "refined_triples.txt")
    output_file = os.path.join(output_dir, dataset_name, "triples.txt")
    
    if not os.path.exists(refined_file):
        print(f"Error: {refined_file} not found")
        return
    
    # Read refined triples
    with open(refined_file, 'r', encoding='utf-8') as f:
        refined_lines = [l.strip() for l in f.readlines()]
    
    # Check if mine dataset (has article_mapping.json)
    article_mapping_file = os.path.join(input_dir, "article_mapping.json")
    if os.path.exists(article_mapping_file) and dataset_name == "mine":
        # Mine dataset: merge by article name
        with open(article_mapping_file, 'r', encoding='utf-8') as f:
            article_mapping = json.load(f)
        
        merged_triples = []
        # Sort article names
        sorted_articles = sorted(article_mapping.keys())
        
        for article_name in sorted_articles:
            split_indices = article_mapping[article_name]
            merged = []
            
            for split_idx in split_indices:
                if split_idx < len(refined_lines):
                    try:
                        triples = ast.literal_eval(refined_lines[split_idx])
                        if isinstance(triples, list):
                            merged.extend(triples)
                    except (ValueError, SyntaxError):
                        pass
            
            merged_triples.append(merged)
        
        print(f"Merged {len(merged_triples)} articles from {len(refined_lines)} split lines")
    else:
        # Regular dataset: merge by line index
        mapping_file = os.path.join(input_dir, "mapping.json")
        if not os.path.exists(mapping_file):
            print(f"Error: {mapping_file} not found")
            return
        
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        
        merged_triples = []
        # Sort keys as integers
        sorted_keys = sorted([int(k) for k in mapping.keys()])
        for orig_idx in sorted_keys:
            split_indices = mapping[str(orig_idx)]
            merged = []
            
            for split_idx in split_indices:
                if split_idx < len(refined_lines):
                    try:
                        triples = ast.literal_eval(refined_lines[split_idx])
                        if isinstance(triples, list):
                            merged.extend(triples)
                    except (ValueError, SyntaxError):
                        pass
            
            merged_triples.append(merged)
        
        print(f"Merged {len(merged_triples)} lines from {len(refined_lines)} split lines")
    
    # Save merged triples
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for triples in merged_triples:
            f.write(str(triples) + '\n')
    
    print(f"Saved merged triples to: {output_file}")
    print(f"Total merged articles/lines: {len(merged_triples)}")


def main():
    if len(sys.argv) < 4:
        print("Usage: python merge_triples.py <input_dir> <output_dir> <dataset_name>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    dataset_name = sys.argv[3]
    
    merge_triples(input_dir, output_dir, dataset_name)


if __name__ == "__main__":
    main()
