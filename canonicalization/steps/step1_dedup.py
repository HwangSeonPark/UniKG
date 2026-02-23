import ast

def step1(input_path, output_path):
    canonical = {}
    output_lines = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            row = ast.literal_eval(line)
            if not row:
                output_lines.append([])
                continue

            row_seen = set()
            new_row = []

            for s, p, o in row:
                key = (s.lower(), p.lower(), o.lower())

                if key in row_seen:
                    continue
                row_seen.add(key)

                if key not in canonical:
                    canonical[key] = [s, p, o]

                new_row.append(canonical[key])

            output_lines.append(new_row)

    with open(output_path, "w", encoding="utf-8") as f:
        for row in output_lines:
            f.write(f"{row}\n")
