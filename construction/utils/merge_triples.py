import json
import sys
import os
import ast
from pathlib import Path


def dedup_row(row, canon):
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


def merge_triples(input_dir: str, output_dir: str, dataset_name: str):
    refined_file = os.path.join(output_dir, dataset_name, "refined_triples.txt")
    output_file = os.path.join(output_dir, dataset_name, "triples.txt")

    if not os.path.exists(refined_file):
        print(f"Error: {refined_file} not found")
        return

    with open(refined_file, 'r', encoding='utf-8') as f:
        refined_lines = [l.strip() for l in f.readlines()]

    article_mapping_file = os.path.join(input_dir, "article_mapping.json")
    if os.path.exists(article_mapping_file):
        with open(article_mapping_file, 'r', encoding='utf-8') as f:
            article_mapping = json.load(f)

        canon = {}
        article_triples = {}
        for article_name in sorted(article_mapping.keys()):
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

            article_triples[article_name] = dedup_row(merged, canon)

        os.makedirs(os.path.join(output_dir, dataset_name), exist_ok=True)

        for article_name, triples in article_triples.items():
            article_file = os.path.join(output_dir, dataset_name, f"{article_name}.txt")
            with open(article_file, 'w', encoding='utf-8') as f:
                f.write(str(triples) + '\n')

        with open(output_file, 'w', encoding='utf-8') as f:
            for triples in article_triples.values():
                f.write(str(triples) + '\n')

        print(f"Merged {len(article_triples)} articles from {len(refined_lines)} split lines")
        print(f"Saved {len(article_triples)} article files to: {os.path.join(output_dir, dataset_name)}/")
        print(f"Saved combined triples to: {output_file}")
    else:
        mapping_file = os.path.join(input_dir, "mapping.json")
        if not os.path.exists(mapping_file):
            print(f"Error: {mapping_file} not found")
            return

        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping = json.load(f)

        canon = {}
        merged_triples = []
        for orig_idx in sorted([int(k) for k in mapping.keys()]):
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

            merged_triples.append(dedup_row(merged, canon))

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for triples in merged_triples:
                f.write(str(triples) + '\n')

        print(f"Merged {len(merged_triples)} lines from {len(refined_lines)} split lines")
        print(f"Saved merged triples to: {output_file}")


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
