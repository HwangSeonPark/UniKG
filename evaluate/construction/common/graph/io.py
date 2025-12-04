import ast
from typing import List


def load_flat(path: str) -> List[List[str]]:
    """
    파일에서 한 줄의 트리플 배열을 파싱하여 전체를 평탄화해 반환.
    [['h','r','t'], ...] 형태의 리스트를 모두 이어붙여 반환
    """
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            grp = ast.literal_eval(line)
            for trip in grp:
                data.append(trip)
    return data


def load_lines(path: str) -> List[List[List[str]]]:
    """
    파일에서 한 줄당 한 배열의 트리플을 파싱하여
    라인 단위 리스트로 반환
    """
    res = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            res.append(ast.literal_eval(line))
    return res


def load_lines_safe(path: str) -> List[List[List[str]]]:
    """
    GraphJudge와 동일하게 라인 파싱 중 오류가 발생할 경우
    해당 라인은 [["Null","Null","Null"]] 로 대체하여 반환
    """
    res: List[List[List[str]]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                res.append(ast.literal_eval(line))
            except Exception:
                res.append([["Null", "Null", "Null"]])
    return res


