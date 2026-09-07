from typing import List
from llm_sdk import Small_LLM_Model
import time
import numpy as np

start = time.time()

model = Small_LLM_Model()


def generate_tokens(prompt: str, max_tokens: int) -> None:
    raw_tensor = model.encode(prompt)
    token_ids: List[int] = [int(x) for x in raw_tensor.squeeze(0).tolist()]
    for _ in range(0, max_tokens):
        raw_logits: list[float] = model.get_logits_from_input_ids(token_ids)
        max_index: int = int(np.argmax(raw_logits))
        token_ids.append(max_index)

    print(model.decode(token_ids))


end = time.time() - start
print(f"{end:.2f} s")
