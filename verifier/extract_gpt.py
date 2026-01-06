import os
import json
import asyncio
import time
import sys
from typing import List, Tuple
from openai import AsyncOpenAI
from openai import APIConnectionError, APIError
from extract_base import (
    SYSTEM_PROMPT, build_prompt, safe_parse_response, postprocess_triplets
)


# OpenAI GPT-5.1 settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
MODEL_NAME = "gpt-5.1"

# Batch settings
BATCH_SIZE = 50
MAX_RETRIES = 3
RETRY_DELAY = 5

async def extract_triplets_with_index(
    client: AsyncOpenAI, 
    index: int,
    text: str, 
    max_retries: int = MAX_RETRIES,
    stats: dict = None
) -> Tuple[int, List]:
    for attempt in range(max_retries):
        try:
            st = time.time()
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(text)},
                ],

            )
            et = time.time()
            
            if stats is not None:
                stats['calls'] = stats.get('calls', 0) + 1
                stats['time'] = stats.get('time', 0.0) + (et - st)
                if hasattr(response, 'usage'):
                    stats['tokens'] = stats.get('tokens', 0) + (response.usage.total_tokens if response.usage else 0)
                stats['chars'] = stats.get('chars', 0) + len(text)

            raw_content = response.choices[0].message.content.strip()
            raw_triplets = safe_parse_response(raw_content)
            triplets = postprocess_triplets(raw_triplets, debug_index=index)
            return (index, triplets)
            
        except APIConnectionError as e:
            if attempt < max_retries - 1:
                wait_time = RETRY_DELAY * (attempt + 1)
                await asyncio.sleep(wait_time)
            else:
                print(f"\nError: index {index}: OpenAI API connection failed")
                return (index, [])
                
        except APIError as e:
            print(f"\nError: index {index}: API call failed - {str(e)}")
            return (index, [])
            
        except Exception as e:
            print(f"\nError: index {index}: unexpected error - {str(e)}")
            return (index, [])
    
    return (index, [])


# =========================
# Main
# =========================
async def process_dataset_async(dataset_name: str, client: AsyncOpenAI, input_dir: str, output_dir: str):
    """Process single dataset (read from input folder or file, save to extract_triples.txt)"""
    # Check if input_dir is a file or directory
    if os.path.isfile(input_dir):
        input_path = input_dir
    else:
        input_path = os.path.join(input_dir, "articles.txt")
    
    output_path = os.path.join(output_dir, dataset_name, "extract_triples.txt")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not os.path.exists(input_path):
        print(f"Warning: {input_path} file does not exist. Skipping.")
        return
    
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
    
    stats = {'calls': 0, 'tokens': 0, 'time': 0.0, 'chars': 0}
    total_time = time.time()
    
    with open(output_path, "w", encoding="utf-8") as fout:
        for batch_start in range(0, total_lines, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_lines)
            
            tasks = [
                extract_triplets_with_index(client, idx, texts[idx], MAX_RETRIES, stats)
                for idx in range(batch_start, batch_end)
            ]
        
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            valid_results = []
            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"\nError: batch processing exception: {str(result)}")
                    continue
                valid_results.append(result)
            
   
            valid_results.sort(key=lambda x: x[0])
            
            for idx, triplets in valid_results:
                fout.write(json.dumps(triplets, ensure_ascii=False) + "\n")
            
            progress = batch_end
            print(f"Progress: {progress}/{total_lines} ({progress/total_lines*100:.1f}%)")
    
    total_time = time.time() - total_time
    



async def main():
    import sys
    if len(sys.argv) < 4:
        print("Usage: python extract_gpt.py <dataset_name> <input_dir> <output_dir>")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    input_dir = sys.argv[2]
    output_dir = sys.argv[3]
    if OPENAI_API_KEY == "your-api-key-here" or not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY is not set.")
        print("\nSolution:")
        print("1. Set environment variable: export OPENAI_API_KEY='your-actual-api-key'")
        print("2. Or set directly in code: OPENAI_API_KEY = 'your-actual-api-key'")
        return

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    try:
        await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "test"}],
        )
    except APIConnectionError as e:
        print(f"Error: Failed to connect to OpenAI API: {str(e)}")
        return
    except Exception as e:
        print(f"Error during connection test: {str(e)}")
        return
    await process_dataset_async(dataset_name, client, input_dir, output_dir)
    
    print(f"\n{dataset_name} dataset processing complete!")


if __name__ == "__main__":
    asyncio.run(main())
