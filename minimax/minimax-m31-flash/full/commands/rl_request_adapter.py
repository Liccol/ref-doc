"""Render text completion prompts using this model's local chat template."""
from functools import lru_cache

@lru_cache(maxsize=1)
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained('/mnt/weight/MiniMax-M3.1-Flash-BF16/',trust_remote_code=True,local_files_only=True)

def adapt_request(request_fn):
    def request(base,method,endpoint,payload=None,*,timeout=1800,headers=None):
        if endpoint=='/v1/completions' and method=='POST' and payload is not None:
            payload=dict(payload)
            prompt=payload['prompt']
            if not isinstance(prompt,str):raise ValueError('Task harness expects a single string completion prompt')
            payload['prompt']=tokenizer().apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
        kwargs={'timeout':timeout}
        if headers is not None:kwargs['headers']=headers
        return request_fn(base,method,endpoint,payload,**kwargs)
    return request
