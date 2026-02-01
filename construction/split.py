#!/usr/bin/env python3
"""
Split article files line by line into chunks of approximately 2048 characters each.
Files are split at sentence boundaries to avoid cutting words or sentences.
All chunks are saved to input folder with mapping information.
For mine dataset: handles directory of .txt files, each file is one article.
"""

import nltk
import json
import sys
import os
from pathlib import Path

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK."""
    sentences = nltk.sent_tokenize(text)
    return [' '.join(s.split()) for s in sentences if s.strip()]


def split_line_by_chars(line: str, max_chars: int = 2048) -> list[str]:
    """
    Split a line into chunks of approximately max_chars.
    Splits at sentence boundaries to avoid cutting words or sentences.
    """
    line = line.strip()
    if len(line) <= max_chars:
        return [line]
    
    sentences = split_into_sentences(line)
    chunks = []
    curr_chunk = []
    curr_chars = 0
    
    for sentence in sentences:
        sent_chars = len(sentence)
        
        if curr_chars + sent_chars > max_chars and curr_chunk:
            chunks.append(' '.join(curr_chunk))
            curr_chunk = [sentence]
            curr_chars = sent_chars
        else:
            curr_chunk.append(sentence)
            curr_chars += sent_chars
    
    if curr_chunk:
        chunks.append(' '.join(curr_chunk))
    
    return chunks


def split_articles_from_dir(input_dir: str, output_dir: str, max_chars: int = 2048) -> dict:
    """
    Split articles from directory (for mine dataset).
    Each .txt file is one article. Returns mapping with article names.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    mapping = {}
    article_mapping = {}  # {article_name: [split_line_indices]}
    split_lines = []
    split_idx = 0
    
    # Get all .txt files sorted
    txt_files = sorted([f for f in input_path.glob("*.txt") if f.is_file()])
    
    for article_file in txt_files:
        article_name = article_file.stem  # filename without extension
        article_split_indices = []
        
        # Read entire file content as one article
        with open(article_file, 'r', encoding='utf-8') as f:
            article_content = f.read().strip()
        
        if not article_content:
            continue
        
        # Split article into chunks
        chunks = split_line_by_chars(article_content, max_chars)
        
        for chunk in chunks:
            split_file = output_path / f"article_{split_idx:06d}.txt"
            with open(split_file, 'w', encoding='utf-8') as f:
                f.write(chunk)
            article_split_indices.append(split_idx)
            split_lines.append(chunk)
            split_idx += 1
        
        article_mapping[article_name] = article_split_indices
    
    # Save article mapping (for mine dataset merging)
    article_mapping_file = output_path / "article_mapping.json"
    with open(article_mapping_file, 'w', encoding='utf-8') as f:
        json.dump(article_mapping, f, ensure_ascii=False, indent=2)
    
    # Save line-based mapping (for compatibility)
    line_mapping = {}
    for orig_idx in range(len(split_lines)):
        line_mapping[orig_idx] = [orig_idx]
    mapping_file = output_path / "mapping.json"
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(line_mapping, f, ensure_ascii=False, indent=2)
    
    # Save split articles as single file for processing
    articles_file = output_path / "articles.txt"
    with open(articles_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(split_lines))
    
    return article_mapping


def split_articles(input_path: str, output_dir: str, max_chars: int = 2048) -> dict:
    """
    Split articles line by line and save to output directory.
    If input_path is a directory, treat each .txt file as one article (mine dataset).
    Returns mapping: {original_line_idx: [split_line_indices]} or {article_name: [split_line_indices]}
    """
    input_file = Path(input_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if input is a directory (mine dataset)
    if input_file.is_dir():
        return split_articles_from_dir(str(input_file), output_dir, max_chars)
    
    # Regular file processing
    mapping = {}
    split_lines = []
    split_idx = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines()]
    
    for orig_idx, line in enumerate(lines):
        if not line:
            continue
        
        chunks = split_line_by_chars(line, max_chars)
        split_indices = []
        
        for chunk in chunks:
            split_file = output_path / f"article_{split_idx:06d}.txt"
            with open(split_file, 'w', encoding='utf-8') as f:
                f.write(chunk)
            split_indices.append(split_idx)
            split_lines.append(chunk)
            split_idx += 1
        
        mapping[orig_idx] = split_indices
    
    # Save mapping file
    mapping_file = output_path / "mapping.json"
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    
    # Save split articles as single file for processing
    articles_file = output_path / "articles.txt"
    with open(articles_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(split_lines))
    
    return mapping


def main():
    if len(sys.argv) < 3:
        print("Usage: python split.py <input_articles.txt or input_dir> <output_dir>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Max characters per chunk: 2048")
    print()
    
    mapping = split_articles(input_path, output_dir, max_chars=2048)
    
    if isinstance(mapping, dict) and any(isinstance(k, str) for k in mapping.keys()):
        # Mine dataset: article-based mapping
        total_articles = len(mapping)
        total_split = sum(len(indices) for indices in mapping.values())
        split_count = sum(1 for indices in mapping.values() if len(indices) > 1)
        
        print()
        print(f"Processing complete!")
        print(f"  Original articles: {total_articles}")
        print(f"  Split lines: {total_split}")
        print(f"  Articles that were split: {split_count}")
        print(f"  Output directory: {output_dir}")
    else:
        # Regular dataset: line-based mapping
        total_orig = len(mapping)
        total_split = sum(len(indices) for indices in mapping.values())
        split_count = sum(1 for indices in mapping.values() if len(indices) > 1)
        
        print()
        print(f"Processing complete!")
        print(f"  Original lines: {total_orig}")
        print(f"  Split lines: {total_split}")
        print(f"  Lines that were split: {split_count}")
        print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
