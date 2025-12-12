# extract_entities.py
import asyncio
from tqdm.asyncio import tqdm
from api_client import call_api


ENTITY_PROMPT = """Goal:
Transform the text into a list of entities. Ensure the comprehensiveness and accuracy of the extracted entities, which should be related to the topic of the tex. *YOU MUST INCLUDE ALL ENTITIES RELATED TO THE TOPIC OF THE TEXT DO NOT PROVIDE EMPTY *.
Here are two examples:
Example#1:
Text: "Shotgate Thickets is a nature reserve in the United Kingdom operated by the Essex Wildlife Trust."
List of entities: ["Shotgate Thickets", "Nature reserve", "United Kingdom", "Essex Wildlife Trust"]

Example#2:
Text: "Garczynski Nunatak is a cone-shaped nunatak, the highest in a cluster of nunataks close west of Mount Brecher, lying at the north flank of Quonset Glacier in the Wisconsin Range of the Horlick Mountains of Antarctica."
List of entities: ["Garczynski Nunatak", "nunatak", "Wisconsin Range", "Mount Brecher", "Quonset Glacier", "Horlick Mountains", "Antarctica"]

Example#3:
Text: "She seems happy with this arrangement."
List of entities: ["she", "arrangement"]


Refer to the examples and here is the question:
Text: {text}
List of entities:"""


async def extract_entities(texts, model_info, max_c, out_dir):
    sema = asyncio.Semaphore(max_c)
    out_f = out_dir / "entities.txt"
    
    # 순서 보장을 위한 결과 저장 리스트
    ents = [None] * len(texts)
    
    async def _extract(idx, text):
        async with sema:
            prompt = ENTITY_PROMPT.format(text=text)
            res = await call_api(prompt, model_info)
            clean = res.strip().replace('\n', ' ')
            ents[idx] = clean
            return clean
    
    out_f.write_text("")
    
    tasks = [_extract(i, t) for i, t in enumerate(texts)]
    await tqdm.gather(*tasks, desc="Extracting entities")
    
    # 순서대로 파일에 기록
    with open(out_f, "w", encoding="utf-8") as f:
        for e in ents:
            if e is not None:
                f.write(f"{e}\n")
    
    return ents
