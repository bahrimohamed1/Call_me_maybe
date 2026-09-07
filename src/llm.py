from typing import List
from llm_sdk import Small_LLM_Model
import time
import numpy as np
from .parser import FunctionDefinition
from enum import Enum, auto

start = time.time()

model = Small_LLM_Model()


def generate_tokens(prompt: str, max_tokens: int) -> None:
    raw_tensor = model.encode(prompt)
    token_ids: List[int] = [int(x) for x in raw_tensor.squeeze(0).tolist()]
    for _ in range(0, max_tokens):
        allowed_ids = [1, 2, 3, 5]
        output_result: str =  ""
        raw_logits: list[float] = model.get_logits_from_input_ids(token_ids)
        raw_mask = [-np.inf] * len(raw_logits)
        for i in allowed_ids:
            raw_mask[i] = 0
        raw_logits = np.add(raw_logits, raw_mask) 
        max_index: int = int(np.argmax(raw_logits))
        token_ids.append(max_index)
        output_result += model.decode([max_index])

    print(model.decode(token_ids))


generate_tokens("hello", 10)

end = time.time() - start
print(f"{end:.2f} s")
