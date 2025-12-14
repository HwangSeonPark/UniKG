# denoise.py
import asyncio
import json
from pathlib import Path
from tqdm.asyncio import tqdm

from api_client import call_api


DENOISE_PROMPT = """Goal:
Given the original text and the entity list, extract and rewrite a clean, short sentence that only contains factual content useful for relation triple extraction.
- Keep meaning faithful to the text.
- Prefer a single sentence.
- Remove irrelevant clauses, boilerplate, and noisy fragments.
- Do NOT introduce new entities not in the Entity List.
- If the text does not support any relation among entities, output an empty string.

Output format:
Output ONLY the denoised sentence (or empty string). No explanation.

Text: {text}
Entity List: {ents}
Output:"""


def _ents(ents):
    """
    엔티티 리스트를 모델 프롬프트에 안정적으로 전달하기 위해 JSON 문자열로 변환
    """
    try:
        return json.dumps(ents, ensure_ascii=False)
    except Exception:
        return str(ents)


async def denoise_text(texts, ents, model_info, max_c, out_dir: Path):
    """
    texts, ents는 동일 길이여야 하며, 결과는 denoise.txt에 '입력 순서 그대로' 저장한다.
    """
    sema = asyncio.Semaphore(max_c)
    out_f = out_dir / "denoise.txt"

    ress = [""] * len(texts)

    async def _dnz(idx, text, ent):
        async with sema:
            prompt = DENOISE_PROMPT.format(text=text, ents=_ents(ent))
            res = await call_api(prompt, model_info)

            # 파일 저장 안정성을 위해 개행 제거(한 줄 1 샘플)
            res = (res or "").strip().replace("\n", " ").strip()
            ress[idx] = res
            return res

    out_f.write_text("", encoding="utf-8")

    tasks = [_dnz(i, t, e) for i, (t, e) in enumerate(zip(texts, ents))]
    await tqdm.gather(*tasks, desc="Denoising")

    with open(out_f, "w", encoding="utf-8") as f:
        for r in ress:
            f.write(f"{r}\n")
