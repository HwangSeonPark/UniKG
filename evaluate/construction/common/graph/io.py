import ast
import re
from typing import List


def _clean_line(line: str) -> str:
    """
    라인에서 주석을 제거하고 파싱 가능한 형태로 정리.
    괄호로 감싸진 주석 (예: (Assuming ...), (Note: ...)) 제거
    """
    # 괄호로 감싸진 주석 제거 (예: (Assuming ...), (Note: ...))
    line = re.sub(r'\s*\([^)]*\)\s*$', '', line)
    # 따옴표 오류 수정 (예: "text1" word "text2" -> "text1 word text2")
    # 리스트 내부에서 발생하는 패턴: ", "text1" word "text2"]
    line = re.sub(r'",\s*"([^"]+)"\s+([a-zA-Z]+)\s+"([^"]+)"\]', r'", "\1 \2 \3"]', line)
    return line.strip()


def load_flat(path: str) -> List[List[str]]:
    """
    파일에서 한 줄의 트리플 배열을 파싱하여 전체를 평탄화해 반환.
    [['h','r','t'], ...] 형태의 리스트를 모두 이어붙여 반환
    """
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                cleaned = _clean_line(line)
                # 빈 라인이 되면 건너뛰기
                if not cleaned:
                    continue
                # 리스트 형태가 아니면 건너뛰기
                if not cleaned.startswith('['):
                    continue
                
                # 이미 리스트의 리스트 형태인지 확인
                if cleaned.startswith('[['):
                    grp = ast.literal_eval(cleaned)
                else:
                    # 여러 리스트가 콤마로 구분된 경우 전체를 리스트로 감싸기
                    cleaned = '[' + cleaned + ']'
                    grp = ast.literal_eval(cleaned)
                
                # grp가 리스트의 리스트인지 확인
                if grp and isinstance(grp[0], list):
                    for trip in grp:
                        if isinstance(trip, list) and len(trip) == 3:
                            data.append(trip)
                elif isinstance(grp, list) and len(grp) == 3:
                    data.append(grp)
            except (SyntaxError, ValueError) as e:
                # 파싱 오류 발생 시 해당 라인 건너뛰기
                continue
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
    라인별로 트리플 배열 파싱
    - 파싱 오류, 빈 줄, 빈 트리플([])은 [["error","error","error"]]로 대체
    - 완전 오답 처리를 위해 매칭 불가능한 값 사용
    """
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


