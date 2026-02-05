import os
import json
import asyncio
import sys
import re
import ast
from typing import List, Tuple
from pathlib import Path
from openai import AsyncOpenAI
from openai import APIConnectionError, APIError


def read_jsonl(name: str) -> List[dict]:
    """Read a UTF-8 jsonl prompt file from construction/prompts."""
    pth = Path(__file__).resolve().parent / "prompts" / name
    out = []
    for ln in pth.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        out.append(json.loads(ln))
    return out


def mk_fs(exs: List[dict]) -> str:
    """Build few-shot prompt text from jsonl examples."""
    parts = []
    for i, ex in enumerate(exs, 1):
        txt = str(ex.get("text", "")).strip()
        tps = ex.get("triplets", [])
        parts.append(f"Example {i}:\nText: {txt}\nTriplets: {tps}")
    return "\n\n".join(parts).strip()


FEW_SHOT_PROMPT = mk_fs(read_jsonl("extator_fewshot.jsonl"))
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


def safe_str(value) -> str:
    """Safely convert value to string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def convert_underscore_to_space(text: str) -> str:
    """Convert underscores to spaces."""
    text = safe_str(text)
    if not text:
        return ""
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_relation(rel: str) -> str:
    """Normalize relation string."""
    rel = safe_str(rel)
    if not rel:
        return ""
    rel = convert_underscore_to_space(rel)
    rel = rel.lower()
    rel = re.sub(r"\b(was|were|is|are|been|being)\b", "", rel)
    rel = re.sub(r"\s+", " ", rel).strip()
    return rel


def normalize_argument(arg: str, is_tail: bool = False) -> str:
    """Normalize argument string."""
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
    """Postprocess extracted triplets."""
    cleaned = []
    seen = set()

    for i, t in enumerate(triplets):
        try:
            if not isinstance(t, list) or len(t) != 3:
                continue

            h, r, ta = t

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


def build_prompt(text: str) -> str:
    """Build extraction prompt."""
    return f"""{FEW_SHOT_PROMPT}


Text: {text}
Triplets:
"""


def safe_parse_response(content: str) -> List:
    """Safely parse response content."""
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


VLLM_HOST = os.getenv("VLLM_HOST", "localhost")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "none")
# Port numbers can be set via environment variables
# Format: MODEL_NAME_PORT (e.g., QWEN_PORT, MISTRAL_PORT)
def env_int(name: str):
    val = os.getenv(name)
    if not val:
        return None
    try:
        return int(val)
    except ValueError:
        return None


QWEN_PORT = env_int("QWEN_PORT")
MISTRAL_PORT = env_int("MISTRAL_PORT")
MODEL_PORT_MAP = {
    "Qwen/Qwen2.5-7B-Instruct": QWEN_PORT,
    "mistralai/Mistral-7B-Instruct-v0.3": MISTRAL_PORT
}

# Batch settings
BATCH_SIZE = 50
MAX_RETRIES = 3
RETRY_DELAY = 5
MAX_TOKENS = 10000



async def extract_triplets_with_index(
    client: AsyncOpenAI, 
    index: int,
    text: str,
    model_name: str,
    max_retries: int = MAX_RETRIES,
) -> Tuple[int, List]:
    """Extract triplets asynchronously"""
    text_len = len(text)
    
    for attempt in range(max_retries):
        try:
            estimated_timeout = max(60, min(600, (text_len / 1000) * 100))
            
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model_name,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_prompt(text)},
                    ],
                    max_tokens=MAX_TOKENS
                ),
                timeout=estimated_timeout + 30
            )
            
            raw_content = response.choices[0].message.content.strip()
            raw_triplets = safe_parse_response(raw_content)
            triplets = postprocess_triplets(raw_triplets, debug_index=index)
            
            return (index, triplets)
            
        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                await asyncio.sleep(RETRY_DELAY)
            else:
                print(f"[Error] Index {index}: request timeout", flush=True)
                return (index, [])
                
        except APIConnectionError:
            if attempt < max_retries - 1:
                wait_time = RETRY_DELAY * (attempt + 1)
                await asyncio.sleep(wait_time)
            else:
                print(f"[Error] Index {index}: failed to connect to server", flush=True)
                return (index, [])
                
        except APIError as e:
            print(f"[Error] Index {index}: API call failed - {str(e)}", flush=True)
            return (index, [])
            
        except Exception as e:
            print(f"[Error] Index {index}: unexpected error - {str(e)}", flush=True)
            return (index, [])
    
    return (index, [])


# =========================
# Main
# =========================
async def process_dataset_async(dataset_name: str, client: AsyncOpenAI, model_name: str,
                                 input_dir: str, output_dir: str):
    """Process single dataset (read from input folder or file, save to extract_triples.txt)"""
    # Check if input_dir is a file or directory
    if os.path.isfile(input_dir):
        input_path = input_dir
    else:
        input_path = os.path.join(input_dir, "articles.txt")
    
    output_path = os.path.join(output_dir, dataset_name, "extract_triples.txt")
    
    if not os.path.exists(input_path):
        print(f"Skip: {input_path} not found")
        return
    

    odir = os.path.dirname(output_path)
    os.makedirs(odir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"{'='*60}")
    

    with open(input_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    
    lim = int(os.getenv("LIM", "0") or "0")
    if lim > 0:
        texts = texts[:lim]
    
    total_lines = len(texts)
    print(f"Total {total_lines} lines to process...")
    print(f"Batch size: {BATCH_SIZE} (concurrent requests)\n")
    
    with open(output_path, "w", encoding="utf-8") as fout:
        for batch_start in range(0, total_lines, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_lines)
            
            # Create batch tasks
            tasks = [
                extract_triplets_with_index(client, idx, texts[idx], model_name)
                for idx in range(batch_start, batch_end)
            ]
            
            # Execute batch
            batch_results = await asyncio.gather(*tasks)
            
            # Sort by index and save immediately
            batch_results.sort(key=lambda x: x[0])
            
            for idx, triplets in batch_results:
                fout.write(json.dumps(triplets, ensure_ascii=False) + "\n")
            
            # Print progress
            progress = batch_end
            print(f"Progress: {progress}/{total_lines} ({progress/total_lines*100:.1f}%)", flush=True)
    
    print(f"{dataset_name} processing complete!", flush=True)


async def process_mine_dataset_async(client: AsyncOpenAI, model_name: str, 
                                      input_dir: str, output_dir: str):
    """Process mine dataset (read from input folder)"""
    if not os.path.exists(input_dir):
        print(f"Warning: {input_dir} directory does not exist. Skipping.")
        return
    
    # Read articles from input folder
    articles_file = os.path.join(input_dir, "articles.txt")
    if not os.path.exists(articles_file):
        print(f"Warning: {articles_file} not found. Skipping.")
        return
    
    # Output directory
    out_dir = os.path.join(output_dir, "mine")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "extract_triples.txt")
    
    print(f"\n{'='*60}", flush=True)
    print(f"Dataset: mine", flush=True)
    print(f"Input: {articles_file}", flush=True)
    print(f"Output: {output_path}", flush=True)
    print(f"{'='*60}", flush=True)
    
    # Read all texts
    with open(articles_file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    
    lim = int(os.getenv("LIM", "0") or "0")
    if lim > 0:
        texts = texts[:lim]
    
    total_lines = len(texts)
    print(f"Total {total_lines} lines to process...")
    print(f"Batch size: {BATCH_SIZE} (concurrent requests)\n")
    
    # Process in batches
    with open(output_path, "w", encoding="utf-8") as fout:
        for batch_start in range(0, total_lines, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_lines)
            
            # Create batch tasks
            tasks = [
                extract_triplets_with_index(client, idx, texts[idx], model_name)
                for idx in range(batch_start, batch_end)
            ]
            
            # Execute batch
            batch_results = await asyncio.gather(*tasks)
            
            # Sort by index and save immediately
            batch_results.sort(key=lambda x: x[0])
            
            for idx, triplets in batch_results:
                fout.write(json.dumps(triplets, ensure_ascii=False) + "\n")
            
            # Progress output
            progress = batch_end
            print(f"Progress: {progress}/{total_lines} ({progress/total_lines*100:.1f}%)", flush=True)
    
    print("mine dataset processing complete!", flush=True)


async def main():
    # Parse command line arguments
    if len(sys.argv) < 5:
        print("Usage: python extractor.py <model_name> <dataset_name> <input_dir> <output_dir>")
        sys.exit(1)
    
    mdl_nm = sys.argv[1]
    dset_nm = sys.argv[2]
    input_dir = sys.argv[3]
    output_dir = sys.argv[4]

    mdl_l = mdl_nm.lower()
    gpt = mdl_l.startswith("gpt")

    if gpt:
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            print("Error: OPENAI_API_KEY is not set.", flush=True)
            raise SystemExit(1)

        model_name = os.getenv("OPENAI_MODEL", "gpt-5.1") if mdl_l == "gpt" else mdl_nm
        client = AsyncOpenAI(api_key=key)
    else:
        mdl_map = {
            "qwen": "Qwen/Qwen2.5-7B-Instruct",
            "mistral": "mistralai/Mistral-7B-Instruct-v0.3"
        }
        model_name = mdl_map.get(mdl_nm) or mdl_nm
        if "/" not in model_name and model_name not in mdl_map.values():
            model_name = mdl_map["qwen"]

        vllm_port = MODEL_PORT_MAP.get(model_name)
        if not vllm_port:
            dft = os.getenv("DEFAULT_VLLM_PORT") or os.getenv("QWEN_PORT")
            if not dft:
                print("Error: DEFAULT_VLLM_PORT (or QWEN_PORT) is not set (needed for vLLM extraction).", flush=True)
                raise SystemExit(1)
            vllm_port = int(dft)
        vllm_base_url = f"http://{VLLM_HOST}:{vllm_port}/v1"

        client = AsyncOpenAI(
            base_url=vllm_base_url,
            api_key=VLLM_API_KEY,
        )

    try:
        await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5,
        )
    except APIConnectionError:
        if gpt:
            print("Error: Failed to connect to OpenAI API.", flush=True)
        else:
            print(f"Error: Failed to connect to vLLM server at {vllm_base_url}", flush=True)
            print(f"Please check if {model_name} server is running on port {vllm_port}.", flush=True)
        raise SystemExit(1)
    except Exception as e:
        print(f"Error during connection test: {str(e)}", flush=True)
        raise SystemExit(1)
    
    # Process dataset
    if dset_nm == "mine":
        await process_mine_dataset_async(client, model_name, input_dir, output_dir)
    else:
        await process_dataset_async(dset_nm, client, model_name, input_dir, output_dir)
    
    print("All datasets processed!", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
        sys.exit(0)  # Explicitly return exit code 0 when normal exit
    except SystemExit as e:
        sys.exit(e.code if e.code is not None else 1)
    except Exception as e:
        print(f"Fatal error: {str(e)}", flush=True)
        sys.exit(1)
