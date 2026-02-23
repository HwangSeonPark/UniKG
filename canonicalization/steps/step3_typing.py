import json
from steps.utils import MODEL_MAP, get_client

MAX_INPUT_TOKENS = 1400
BASE_PROMPT_TOKENS = 750
MAX_BATCH_SIZE = 16
MIN_BATCH_SIZE = 1

def step3(input_path, output_path, prompt_dir, model_key, api_base=None):
    client = get_client(api_base)
    MODEL_NAME = MODEL_MAP[model_key]

    ENTITY_TYPE_PATH = f"{prompt_dir}/entity_types_v1.json"
    FEWSHOT_PATH = f"{prompt_dir}/fewshot_entity_typing_v1.jsonl"

    with open(ENTITY_TYPE_PATH, "r", encoding="utf-8") as f:
        ont = json.load(f)["labels"]

    labels_block = "\n".join(
        [f"{i+1}. {k}: {v}" for i, (k, v) in enumerate(ont.items())]
    )

    fewshot_msgs = []
    with open(FEWSHOT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue

            fewshot_msgs.append({"role": "user", "content": ex["text"]})
            fewshot_msgs.append({
                "role": "assistant",
                "content": json.dumps({"label": ex["label"]})
            })

    SYSTEM_PROMPT = f"""
You are an expert classifier for a knowledge-graph triple tagging task.

Each input contains:
- [HEAD] ... [/HEAD]
- [REL] ... [/REL]
- [TAIL] ... [/TAIL]
- [FOCUS] ... [/FOCUS]

Your task:
- If FOCUS=HEAD: classify HEAD
- If FOCUS=TAIL: classify TAIL
- Prefer relation cues when ambiguous

Allowed labels:
{labels_block}

Return a JSON array of objects:
{{"id": <int>, "label": "<entity type>"}}

Rules:
- Classify EACH item independently
- No extra text
- One label per item
""".strip()

    def estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def flush_batch(batch_items):
        if not batch_items:
            return {}

        user_block = "\n".join(
            [f'{it["id"]}. {it["text"]}' for it in batch_items]
        )

        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + fewshot_msgs
            + [{"role": "user", "content": user_block}]
        )

        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0.0,
                messages=messages
            )
            content = resp.choices[0].message.content
            results = json.loads(content)
        except Exception:
            return {it["id"]: "Unknown" for it in batch_items}

        out = {}
        for r in results:
            if isinstance(r, dict) and "id" in r and "label" in r:
                out[r["id"]] = r["label"]

        for it in batch_items:
            out.setdefault(it["id"], "Unknown")

        return out

    cache = {}
    batch = []
    batch_meta = []
    current_tokens = BASE_PROMPT_TOKENS
    next_id = 1

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            key = item["text"]

            if key in cache:
                item["label"] = cache[key]
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                continue

            item_text = item["text"]
            item_tokens = estimate_tokens(item_text)

            if (
                batch and
                (
                    current_tokens + item_tokens > MAX_INPUT_TOKENS
                    or len(batch) >= MAX_BATCH_SIZE
                )
            ):
                labels = flush_batch(batch)

                for orig_item, k, i_id in batch_meta:
                    label = labels.get(i_id, "Unknown")
                    cache[k] = label
                    orig_item["label"] = label
                    fout.write(json.dumps(orig_item, ensure_ascii=False) + "\n")

                batch = []
                batch_meta = []
                current_tokens = BASE_PROMPT_TOKENS

            batch.append({
                "id": next_id,
                "text": item_text
            })
            batch_meta.append((item, key, next_id))
            current_tokens += item_tokens
            next_id += 1

        if batch:
            labels = flush_batch(batch)
            for orig_item, k, i_id in batch_meta:
                label = labels.get(i_id, "Unknown")
                cache[k] = label
                orig_item["label"] = label
                fout.write(json.dumps(orig_item, ensure_ascii=False) + "\n")
