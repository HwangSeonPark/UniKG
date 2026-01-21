import json
from sentence_transformers import SentenceTransformer

def step5(input_path, output_path):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    entity_cache = {}
    relation_cache = {}

    def get_embedding(text, cache):
        if text in cache:
            return cache[text]
        emb = model.encode(text, convert_to_numpy=True).tolist()
        cache[text] = emb
        return emb

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            head_text = item["head"]["text"]
            rel_text = item["relation"]
            tail_text = item["tail"]["text"]

            item["head"]["embedding"] = get_embedding(head_text, entity_cache)
            item["relation_embedding"] = get_embedding(rel_text, relation_cache)
            item["tail"]["embedding"] = get_embedding(tail_text, entity_cache)

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
