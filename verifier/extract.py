import os
import json
import asyncio
import sys
import time
from typing import List, Tuple
from openai import AsyncOpenAI
from openai import APIConnectionError, APIError
from extract_base import (
    SYSTEM_PROMPT, build_prompt, safe_parse_response, postprocess_triplets
)


VLLM_HOST = os.getenv("VLLM_HOST", "localhost")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "none")
# Port numbers can be set via environment variables
# Format: MODEL_NAME_PORT (e.g., QWEN_PORT, MISTRAL_PORT)
QWEN_PORT = int(os.getenv("QWEN_PORT"))
MISTRAL_PORT = int(os.getenv("MISTRAL_PORT"))
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
    file_name: str = ""
) -> Tuple[int, List]:
    """Extract triplets asynchronously"""
    text_len = len(text)
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()
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
                
        except APIConnectionError as e:
            if attempt < max_retries - 1:
                wait_time = RETRY_DELAY * (attempt + 1)
                await asyncio.sleep(wait_time)
            else:
                print(f"[Error] Index {index}: failed to connect to vLLM server", flush=True)
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
async def process_dataset_async(dataset_name: str, client: AsyncOpenAI, model_name: str, mdl_nm: str, 
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
    

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"{'='*60}")
    

    with open(input_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    
    total_lines = len(texts)
    print(f"Total {total_lines} lines to process...")
    print(f"Batch size: {BATCH_SIZE} (concurrent requests)\n")
    
    results_dict = {}
    
    
    overall_start_time = time.time()
    with open(output_path, "w", encoding="utf-8") as fout:
        for batch_start in range(0, total_lines, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_lines)
            batch_start_time = time.time()
            
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
    
    print(f"✓ {dataset_name} processing complete!", flush=True)


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
    
    total_lines = len(texts)
    print(f"Total {total_lines} lines to process...")
    print(f"Batch size: {BATCH_SIZE} (concurrent requests)\n")
    
    # Process in batches
    overall_start_time = time.time()
    with open(output_path, "w", encoding="utf-8") as fout:
        for batch_start in range(0, total_lines, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_lines)
            batch_start_time = time.time()
            
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
    
    print(f"✓ mine dataset processing complete!", flush=True)


async def main():
    # Parse command line arguments
    if len(sys.argv) < 4:
        print("Usage: python extract.py <model_name> <dataset_name> <input_dir> <output_dir>")
        sys.exit(1)
    
    mdl_nm = sys.argv[1]
    dset_nm = sys.argv[2]
    input_dir = sys.argv[3]
    output_dir = sys.argv[4]
    
    # Model name mapping
    mdl_map = {
        "qwen": "Qwen/Qwen2.5-7B-Instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3"
    }
    model_name = mdl_map.get(mdl_nm, mdl_map["qwen"])
    
    # Get port for model (default to QWEN_PORT if model not in map)
    default_port = int(os.getenv("DEFAULT_VLLM_PORT", os.getenv("QWEN_PORT")))
    vllm_port = MODEL_PORT_MAP.get(model_name, default_port)
    vllm_base_url = f"http://{VLLM_HOST}:{vllm_port}/v1"
    
    # Create AsyncOpenAI client
    client = AsyncOpenAI(
        base_url=vllm_base_url,
        api_key=VLLM_API_KEY,
    )
    
    # Test connection
    try:
        await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5,
        )
    except APIConnectionError as e:
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
        await process_dataset_async(dset_nm, client, model_name, mdl_nm, input_dir, output_dir)
    
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
