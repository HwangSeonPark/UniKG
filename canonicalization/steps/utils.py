from openai import OpenAI

MODEL_MAP = {
    "gpt": "gpt-5.1-2025-11-13",
    "qwen": "Qwen2.5-7B-Instruct",
    "mistral": "Mistral-7B-Instruct-v0.3",
}

def get_client(api_base=None):
    """
    GPT  : OPENAI_API_KEY env var
    vLLM : api_base + dummy key
    """
    if api_base:
        return OpenAI(base_url=api_base, api_key="EMPTY")
    return OpenAI()
