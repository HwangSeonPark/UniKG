import json

def step7(input_path, output_path):
    def strip_embeddings(item: dict) -> dict:
        out = {}
        for k in ["triple_id", "row_id", "relation"]:
            if k in item:
                out[k] = item[k]

        out["head"] = {"text": item["head"]["text"], "type": item["head"]["type"]}
        out["tail"] = {"text": item["tail"]["text"], "type": item["tail"]["type"]}
        return out

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            item = json.loads(line)
            clean = strip_embeddings(item)
            fout.write(json.dumps(clean, ensure_ascii=False) + "\n")
