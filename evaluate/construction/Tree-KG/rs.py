import json
import re
from typing import List, Optional
from .common import _llm


def rs(pd: List[List[str]], key: str = None) -> float:
    """
    RS 계산 (LLM 기반 관계 강도 평가)

    Args:
        pd: PD 트리플 [[h,r,t], ...]
        key: OpenRouter API 키(없으면 None 반환)

    Returns:
        평균 RS 값 [0,10] 또는 None(키 없음)
    """
    if len(pd) == 0:
        return 0.0

    totl = 0.0
    for h, r, t in pd:
        pmt = f"""You are a relationship extraction expert.

Task: Given two entities and their relationship, evaluate the relationship strength.

Entity 1 (Head): {h}
Relationship Type: {r}
Entity 2 (Tail): {t}

Constraints:
- Output must be valid JSON
- Relationship strength must be 0-10
- Provide scoring rationale

Output Template:
{{
"is_relevant": true/false,
"type": "{r}",
"strength": 0-10,
"reason": "Scoring rationale"
}}

Respond with ONLY valid JSON.
"""
        scor = 5.0
        try:
            resp = _llm(pmt, temp=0.1, key=key)
            jstr = resp
            if '```' in resp:
                parts = resp.split('```')
                if len(parts) >= 3:
                    jstr = parts[1].strip().replace('json', '', 1).strip()
                else:
                    jstr = resp.replace('```json', '').replace('```', '').strip()
            else:
                jstr = resp.strip()
            jstr = jstr.replace('\n', '').replace('\r', '').strip()
            data = json.loads(jstr)
            scor = float(data.get('strength', 5.0))
            scor = min(10.0, max(0.0, scor))
        except ValueError as e:
            if "NO_KEY" in str(e):
                # API 키가 없으면 None 반환
                return None
            _raw = locals().get('resp', '')
            nums = re.findall(r'[0-9]{1,2}(?:\.\d+)?', _raw)
            if nums:
                scor = float(nums[0])
                scor = min(10.0, max(0.0, scor))
            else:
                scor = 5.0
        except Exception:
            _raw = locals().get('resp', '')
            nums = re.findall(r'[0-9]{1,2}(?:\.\d+)?', _raw)
            if nums:
                scor = float(nums[0])
                scor = min(10.0, max(0.0, scor))
            else:
                scor = 5.0
        totl += scor

    return totl / len(pd)


