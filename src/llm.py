import time
start = time.time()

from llm_sdk import Small_LLM_Model
from typing import List


small = Small_LLM_Model()

raw_tensor = small.encode("hello")
token_ids: List[int] = [int(x) for x in raw_tensor]
logits: list[float] = small.get_logits_from_input_ids(token_ids)


print(len(logits))
end = time.time() - start
print(f"{end:.2f}ms")