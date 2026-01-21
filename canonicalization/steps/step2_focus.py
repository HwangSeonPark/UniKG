import ast, json

def step2(input_path, output_path):
    idx = 1

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for row_id, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue

            rows = ast.literal_eval(line)
            for h, r, t in rows:
                base_id = f"triple_{idx:06d}"

                fout.write(json.dumps({
                    "row_id": row_id,
                    "id": f"{base_id}_H",
                    "text": f"[HEAD] {h} [/HEAD] [REL] {r} [/REL] [TAIL] {t} [/TAIL] [FOCUS] HEAD [/FOCUS]"
                }, ensure_ascii=False) + "\n")

                fout.write(json.dumps({
                    "row_id": row_id,
                    "id": f"{base_id}_T",
                    "text": f"[HEAD] {h} [/HEAD] [REL] {r} [/REL] [TAIL] {t} [/TAIL] [FOCUS] TAIL [/FOCUS]"
                }, ensure_ascii=False) + "\n")

                idx += 1
