"""
Common base module for triple extraction.
Contains shared prompts and postprocessing functions used by both extractor.py and extractor_gpt.py
"""

import re
import ast
from typing import List

from pathlib import Path


def read_txt(name: str) -> str:
    """Read a UTF-8 prompt file from construction/prompts."""
    pth = Path(__file__).resolve().parent / "prompts" / name
    return pth.read_text(encoding="utf-8")


FEW_SHOT_PROMPT = read_txt("extator_fewshot.txt")
SYSTEM_PROMPT = """You are an Open Information Extraction system.
Extract factual triplets from the given text.

Rules:
- Extract ALL possible factual triplets stated in the text.
- Relations must be verb or verb phrases with spaces (not underscores).
- Head and tail must be noun phrases with spaces (not underscores).
- Do not infer facts that are not explicitly stated.
- Output only a Python list of [head, relation, tail].
- Use spaces, not underscores, in all triple components.
"""

# =========================
# Postprocessing functions
# =========================
def safe_str(value) -> str:
    """Safely convert value to string"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def convert_underscore_to_space(text: str) -> str:
    """Convert underscores to spaces"""
    text = safe_str(text)
    if not text:
        return ""
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_relation(rel: str) -> str:
    """Normalize relation string"""
    rel = safe_str(rel)
    if not rel:
        return ""
    rel = convert_underscore_to_space(rel)
    rel = rel.lower()
    rel = re.sub(r"\b(was|were|is|are|been|being)\b", "", rel)
    rel = re.sub(r"\s+", " ", rel).strip()
    return rel


def normalize_argument(arg: str, is_tail: bool = False) -> str:
    """Normalize argument string"""
    arg = safe_str(arg)
    if not arg:
        return ""
    arg = convert_underscore_to_space(arg)
    arg = arg.strip()
    arg = re.sub(r"^(the|a|an)\s+", "", arg, flags=re.I)
    arg = re.sub(r"[.,;:]$", "", arg)

    if is_tail and arg:
        arg = re.split(r"\b(who|which|that)\b", arg)[0].strip()
    return arg


def postprocess_triplets(triplets: List, debug_index: int = None) -> List:
    """Postprocess extracted triplets"""
    cleaned = []
    seen = set()

    for i, t in enumerate(triplets):
        try:
            if not isinstance(t, list) or len(t) != 3:
                continue

            h, r, ta = t
            
            # Convert to string if needed
            if not isinstance(h, str):
                h = str(h) if h is not None else ""
            if not isinstance(r, str):
                r = str(r) if r is not None else ""
            if not isinstance(ta, str):
                ta = str(ta) if ta is not None else ""
            
            h = normalize_argument(h, is_tail=False)
            r = normalize_relation(r)
            ta = normalize_argument(ta, is_tail=True)

            if not h or not r or not ta:
                continue

            if len(r.split()) > 9:
                continue

            key = (h.lower(), r.lower(), ta.lower())
            if key in seen:
                continue
            seen.add(key)

            cleaned.append([h, r, ta])
            
        except Exception as e:
            if debug_index is not None:
                print(f"Warning: index {debug_index}, triplet {i}: {str(e)}")
            continue

    return cleaned


# =========================
# Prompt builder
# =========================
def build_prompt(text: str) -> str:
    """Build extraction prompt"""
    return f"""{FEW_SHOT_PROMPT}


Text: {text}
Triplets:
"""


# =========================
# Safe parsing
# =========================
def safe_parse_response(content: str) -> List:
    """Safely parse response content"""
    content = content.strip()

    if "```" in content:
        parts = content.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("python"):
                part = part[6:].strip()
            if part.startswith("[") or part.startswith("("):
                content = part
                break

    start_idx = content.find("[")
    if start_idx != -1:
        content = content[start_idx:]

    end_idx = content.rfind("]")
    if end_idx != -1:
        content = content[: end_idx + 1]

    try:
        result = ast.literal_eval(content)
        if isinstance(result, list):
            return result
    except (ValueError, SyntaxError):
        lines = content.split("\n")
        parsed_items = []
        for line in lines:
            line = line.strip()
            if not line or not (line.startswith("[") or line.startswith("(")):
                continue
            try:
                item = ast.literal_eval(line)
                if isinstance(item, list) and len(item) == 3:
                    parsed_items.append(item)
            except Exception:
                continue
        if parsed_items:
            return parsed_items

    return []

