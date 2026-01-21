import json, re
from collections import defaultdict

def step4(input_path, output_path):
    def parse_text(text):
        h = re.search(r"\[HEAD\](.*?)\[/HEAD\]", text).group(1).strip()
        r = re.search(r"\[REL\](.*?)\[/REL\]", text).group(1).strip()
        t = re.search(r"\[TAIL\](.*?)\[/TAIL\]", text).group(1).strip()
        return h, r, t

    groups = defaultdict(dict)

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            row_id = item["row_id"]
            base_id = item["id"].rsplit("_", 1)[0]
            focus = item["id"].split("_")[-1]

            h, r, t = parse_text(item["text"])

            g = groups[(row_id, base_id)]
            g["row_id"] = row_id
            g["triple_id"] = base_id
            g["relation"] = r

            if focus == "H":
                g["head"] = {"text": h, "type": item["label"]}
                g.setdefault("tail", {"text": t})
            else:
                g["tail"] = {"text": t, "type": item["label"]}
                g.setdefault("head", {"text": h})

    with open(output_path, "w", encoding="utf-8") as f:
        for g in groups.values():
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
