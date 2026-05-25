
from pie.vllm_client import VLLMClient

class CoSOptimizer:
    def __init__(self, client: VLLMClient, template_file: str):
        self.client = client
        with open(template_file, 'r') as f:
            self.template = f.read()

    async def optimize(self, code, suggestion):
        base_prompt = self.template.replace("{{  src_code  }}", code).replace("{{  suggestion  }}", suggestion)
        return await self.client.inference(base_prompt)