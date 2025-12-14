# api_client.py
import asyncio
import os
from openai import AsyncOpenAI


_clients = {}


# 각 모델별 시스템 프롬프트 정의
SYSTEM_PROMPTS = {
    "llama": "You are an expert at knowledge graph construction. Follow instructions carefully and output only the list format without explanation.",
    "mistral": "You are an expert at knowledge graph construction. Output only the list format without any explanation or markdown.",
    "qwen": "You are an expert at knowledge graph construction. Return only the list format without explanation.",
    "gpt": "You are an expert at knowledge graph construction. Follow the output format strictly.",
}


def get_sys(name):
    """모델명 기반 시스템 프롬프트 반환"""
    if "llama" in name.lower():
        return SYSTEM_PROMPTS["llama"]
    elif "mistral" in name.lower():
        return SYSTEM_PROMPTS["mistral"]
    elif "qwen" in name.lower():
        return SYSTEM_PROMPTS["qwen"]
    else:
        return SYSTEM_PROMPTS["gpt"]


async def call_api(prompt, model_info, max_retries=5):
    name = model_info.get("name", "")
    sys = get_sys(name)
    
    # api_client.py (call_api 안의 OpenAI 모델인 경우 블록만 수정)

    # OpenAI 모델인 경우
    if model_info.get("provider") == "openai":
        key = "openai"

        if key not in _clients:
            api = model_info.get("api_key") or os.getenv("OPENAI_API_KEY")
            _clients[key] = AsyncOpenAI(api_key=api)

        for retry in range(max_retries):
            try:
                # 한국어 주석:
                # gpt-5 계열은 max_tokens 대신 max_completion_tokens를 요구함. [web:150][web:165]
                par = {
                    "model": name,
                    "messages": [
                        {"role": "system", "content": sys},
                        {"role": "user", "content": prompt},
                    ]
                }

                if "gpt-5" in name.lower():
                    par["max_completion_tokens"] = 2048
                else:
                    par["max_tokens"] = 2048

                res = await _clients[key].chat.completions.create(**par)
                return res.choices[0].message.content

            except Exception:
                if retry == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** retry)

    
    # 로컬 vLLM 서버인 경우
    else:
        port = model_info["port"]
        
        if port not in _clients:
            _clients[port] = AsyncOpenAI(
                api_key="dummy",
                base_url=f"http://localhost:{port}/v1"
            )
        
        for retry in range(max_retries):
            try:
                res = await _clients[port].chat.completions.create(
                    model=name,
                    messages=[
                        {"role": "system", "content": sys},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=2048
                )
                return res.choices[0].message.content
            except Exception as e:
                if retry == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** retry)
