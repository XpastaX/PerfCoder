from pprint import pprint, pformat
import requests
import json
from typing import List, Dict, Union, Tuple
import asyncio
import aiohttp


class VLLMClient:
    def __init__(self, model: str, host: str = "localhost", port: int = None, temperature=0, top_p=1, top_k=-1, max_tokens=4096, seed=42):
        if port is None:
            port = 8001

        self.model = model
        self.port = port
        self.url = f"http://{host}:{port}/v1/chat/completions"
        self.DEFAULT_OPTIONS = {
            'temperature': temperature,
            'max_tokens': max_tokens,
            'top_p': top_p,
            'top_k': top_k,
            'seed': seed
        }
        self.timeout = 180  # seconds

    def generate(self, prompt: str) -> Union[str, Tuple[str, List[Dict]]]:
        try:
            headers = {"Content-Type": "application/json"}
            messages = [{"role": "user", "content": prompt}]
            data = {
                "model": self.model,
                "messages": messages,
            }
            data.update(self.DEFAULT_OPTIONS)

            response = requests.post(self.url, headers=headers, data=json.dumps(data), timeout=self.timeout)
            content = response.json()
            choice = content['choices'][0]
            gen_text = choice['message']['content']
            return gen_text

        except Exception as e:
            raise Exception(f'Failed in LLM request! {e}')

    async def inference(self, prompt):
        try:
            headers = {"Content-Type": "application/json"}
            messages = [{"role": "user", "content": prompt}]
            data = {
                "model": self.model,
                "messages": messages,
            }
            data.update(self.DEFAULT_OPTIONS)

            timeout = aiohttp.ClientTimeout(total=self.timeout)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.url, json=data, headers=headers) as resp:
                    result = await resp.json()
                    return result['choices'][0]['message']['content']

        except Exception as e:
            raise Exception(f'Failed in LLM request! {e}')
