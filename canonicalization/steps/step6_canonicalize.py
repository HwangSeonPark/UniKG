import json, numpy as np
from collections import defaultdict
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from steps.utils import MODEL_MAP, get_client

TOP_K = 16
BM25_WEIGHT = 0.5
EMB_WEIGHT = 0.5

def step6(input_path, out_path, model_key, api_base=None):
    client = get_client(api_base)
    MODEL_NAME = MODEL_MAP[model_key]

    def load_jsonl(path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(l) for l in f]

    data = load_jsonl(input_path)

    entities = defaultdict(dict)
    relations = defaultdict(dict)

    for row in data:
        h, t = row["head"], row["tail"]
        r = row["relation"]

        entities[h["type"]][h["text"]] = np.array(h["embedding"])
        entities[t["type"]][t["text"]] = np.array(t["embedding"])

        key = (h["type"], t["type"])
        relations[key][r] = np.array(row["relation_embedding"])

    def get_topk(query_text, query_emb, texts, embs, k=TOP_K, min_sim=0.75):
        tokenized = [t.lower().split() for t in texts]
        bm25 = BM25Okapi(tokenized)
        bm25_scores = bm25.get_scores(query_text.lower().split())

        emb_scores = cosine_similarity(
            query_emb.reshape(1, -1),
            np.stack(embs)
        )[0]

        scores = BM25_WEIGHT * bm25_scores + EMB_WEIGHT * emb_scores
        idx = np.argsort(scores)[::-1]

        filtered = [texts[i] for i in idx if emb_scores[i] >= min_sim]
        if not filtered:
            return []
        return filtered[:k]

    def ask_llm(item, candidates, item_type):
        if not candidates:
            return [], None

        cand_text = "\n".join(f"- {c}" for c in candidates)

        prompt = f"""
Find duplicate {item_type} for the item and an alias that best
represents the duplicates. Duplicates are those that are the same
in meaning, such as with variation in tense, plural form, stem form,
case, abbreviation, shorthand.

If semantic equivalence is clear, merge.
If uncertain, do NOT merge, but ALWAYS return valid JSON.

Item:
{item}

Candidates:
{cand_text}

Return JSON only. Do not include explanations or text outside JSON.
If there are no duplicates, return:
{{ "duplicates": [], "canonical": null }}

If duplicates is non-empty, canonical MUST be one of [Item or Candidates].
""".strip()

        response = client.responses.create(
            model=MODEL_NAME,
            input=prompt,
            max_output_tokens=256,
            temperature=0.0,
        )

        text = response.output_text.strip()
        s, e = text.index("{"), text.rindex("}") + 1
        j = json.loads(text[s:e])
        return j.get("duplicates", []), j.get("canonical")

    entity_clusters = {}

    for etype, pool in entities.items():
        remaining = dict(pool)
        while remaining:
            a, a_emb = next(iter(remaining.items()))
            texts = list(remaining.keys())
            embs = list(remaining.values())

            topk = get_topk(a, a_emb, texts, embs)
            candidates = [t for t in topk if t != a]

            dups, canon = ask_llm(a, candidates, "entity")
            dups = [d for d in dups if isinstance(d, str)]

            if dups and canon:
                cluster = set([a] + dups)
                entity_clusters[canon] = cluster
                for x in cluster:
                    remaining.pop(x, None)
            else:
                remaining.pop(a)

    edge_clusters = {}

    for key, pool in relations.items():
        remaining = dict(pool)
        while remaining:
            a, a_emb = next(iter(remaining.items()))
            texts = list(remaining.keys())
            embs = list(remaining.values())

            topk = get_topk(a, a_emb, texts, embs)
            candidates = [t for t in topk if t != a]

            dups, canon = ask_llm(a, candidates, "relation")
            dups = [d for d in dups if isinstance(d, str)]

            if dups and canon:
                cluster = set([a] + dups)
                edge_clusters[canon] = cluster
                for x in cluster:
                    remaining.pop(x, None)
            else:
                remaining.pop(a)

    entity_map = {alias: canon for canon, aliases in entity_clusters.items() for alias in aliases}
    relation_map = {alias: canon for canon, aliases in edge_clusters.items() for alias in aliases}

    for row in data:
        row["head"]["text"] = entity_map.get(row["head"]["text"], row["head"]["text"])
        row["tail"]["text"] = entity_map.get(row["tail"]["text"], row["tail"]["text"])
        row["relation"] = relation_map.get(row["relation"], row["relation"])

    with open(out_path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
