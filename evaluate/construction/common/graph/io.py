import ast
import re
from typing import List


def _clean_line(line: str) -> str:

    line = re.sub(r'",\s*"([^"]+)"\s+([a-zA-Z]+)\s+"([^"]+)"\]', r'", "\1 \2 \3"]', line)
    return line.strip()


def load_flat(path: str) -> List[List[str]]:

    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                cleaned = _clean_line(line)

                if not cleaned:
                    continue
                if not cleaned.startswith('['):
                    continue
                
  
                if cleaned.startswith('[['):
                    grp = ast.literal_eval(cleaned)
                else:
             
                    cleaned = '[' + cleaned + ']'
                    grp = ast.literal_eval(cleaned)
                
   
                if grp and isinstance(grp[0], list):
                    for trip in grp:
                        if isinstance(trip, list) and len(trip) == 3:
                            data.append(trip)
                elif isinstance(grp, list) and len(grp) == 3:
                    data.append(grp)
            except (SyntaxError, ValueError) as e:
     
                continue
    return data


def load_lines(path: str) -> List[List[List[str]]]:

    res = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            res.append(ast.literal_eval(line))
    return res


def load_lines_safe(path: str) -> List[List[List[str]]]:
    res: List[List[List[str]]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 빈 줄 처리
            if not line:
                res.append([["error", "error", "error"]])
                continue
            try:
                data = ast.literal_eval(line)
                # 빈 트리플 [] 처리
                if not data or data == []:
                    res.append([["error", "error", "error"]])
                else:
                    res.append(data)
            except Exception:
                # 파싱 오류 처리
                res.append([["error", "error", "error"]])
    return res


