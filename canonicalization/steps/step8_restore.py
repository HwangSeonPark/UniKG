import json
from collections import defaultdict
import os

def step8(input_path, output_path):
    rows = defaultdict(list)

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            row_id = obj["row_id"]
            h = obj["head"]["text"]
            r = obj["relation"]
            t = obj["tail"]["text"]
            rows[row_id].append([h, r, t])

    min_row = min(rows.keys())
    max_row = max(rows.keys())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row_id in range(min_row, max_row + 1):
            f.write(json.dumps(rows.get(row_id, []), ensure_ascii=False) + "\n")
