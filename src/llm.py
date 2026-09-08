from typing import List, Dict
from llm_sdk import Small_LLM_Model
import time
import numpy as np
from .parser import FunctionDefinition, load_definitions
from enum import Enum, auto
import json

start = time.time()

model = Small_LLM_Model()


class States(Enum):
    START = auto()
    PROMPT_OPEN_QUOTE = auto()
    PROMPT = auto()
    PROMPT_CLOSE_QUOTE = auto()
    PROMPT_COLON = auto()
    PROMPT_VALUE_OPEN_QUOTE = auto()
    PROMPT_VALUE = auto()
    PROMPT_VALUE_CLOSE_QUOTE = auto()
    PROMPT_COMA = auto()
    KEY_OPEN_QUOTE = auto()
    NAME = auto()
    KEY_CLOSE_QUOTE = auto()
    COLON = auto()
    VALUE_OPEN_QUOTE = auto()
    FUNCTION_NAME = auto()
    COMA = auto()
    PARAM_OPEN_QUOTE = auto()
    PARAMETERS = auto()
    PARAM_CLOSE_QUOTE = auto()
    PARAM_COLON = auto()
    PARAM_VALUE_OPEN_QUOTE = auto()

    END = auto()


def mask_logits(raw_logits: List[float], legal_token: List[int]) -> np.ndarray:
    logits = np.array(raw_logits, dtype=np.float32)

    mask = np.zeros(logits.shape, dtype=bool)
    mask[legal_token] = True

    logits[~mask] = -np.inf
    return logits


def generate_tokens(prompt: str, max_tokens: int, path: str) -> str:
    definitions = load_definitions(path)
    context = "Available functions:\n"
    for function in definitions:
        context += f"\nFunction: {function.name}\n"
        context += f"Description: {function.description}\n"
    context += f"\nUser request:\n{prompt}\n"

    raw_tensor = model.encode(context)
    token_ids: List[int] = [int(x) for x in raw_tensor.squeeze(0).tolist()]
    output_result: str = ""
    current_state: States = States.START

    available_functions: List[str] = [
        definition.name for definition in definitions]

    while current_state != States.END:
        if current_state == States.START:
            legal_pieces = ['{']
            current_state = States.PROMPT_OPEN_QUOTE

        elif current_state == States.PROMPT_OPEN_QUOTE:
            legal_pieces = ['"']
            current_state = States.PROMPT

        elif current_state == States.PROMPT:
            legal_pieces = ['prompt']
            current_state = States.PROMPT_CLOSE_QUOTE

        elif current_state == States.PROMPT_CLOSE_QUOTE:
            legal_pieces = ['"']
            current_state = States.PROMPT_COLON

        elif current_state == States.PROMPT_COLON:
            legal_pieces = [':']
            current_state = States.PROMPT_VALUE_OPEN_QUOTE

        elif current_state == States.PROMPT_VALUE_OPEN_QUOTE:
            legal_pieces = ['"']
            current_state = States.PROMPT_VALUE

        elif current_state == States.PROMPT_VALUE:
            legal_pieces = ["prompt"]
            current_state = States.PROMPT_VALUE_CLOSE_QUOTE

        elif current_state == States.PROMPT_VALUE_CLOSE_QUOTE:
            legal_pieces = ['"']
            current_state = States.PROMPT_COMA

        elif current_state == States.PROMPT_COMA:
            legal_pieces = [',']
            current_state = States.KEY_OPEN_QUOTE

        elif current_state == States.KEY_OPEN_QUOTE:
            legal_pieces = ['"']
            current_state = States.NAME

        elif current_state == States.NAME:
            legal_pieces = ['name']
            current_state = States.KEY_CLOSE_QUOTE

        elif current_state == States.KEY_CLOSE_QUOTE:
            legal_pieces = ['"']
            current_state = States.COLON

        elif current_state == States.COLON:
            legal_pieces = [':']
            current_state = States.VALUE_OPEN_QUOTE

        elif current_state == States.VALUE_OPEN_QUOTE:
            legal_pieces = ['"']
            current_state = States.FUNCTION_NAME

        elif current_state == States.FUNCTION_NAME:
            position: int = 0
            quote_raw = model.encode('"')
            quote_token: List[int] = [int(x)
                                      for x in quote_raw.squeeze(0).tolist()]

            candidates: Dict[str, List[int]] = {}
            for func in available_functions:
                raw = model.encode(func)
                token: List[int] = [int(x)
                                    for x in raw.squeeze(0).tolist()]
                candidates[func] = token

            while True:
                legal_tokens = []

                for candidate in candidates.values():
                    if position == len(candidate):
                        legal_tokens.append(quote_token[0])

                    elif position < len(candidate):
                        legal_tokens.append(candidate[position])

                raw_logits: list[float] = model.get_logits_from_input_ids(
                    token_ids)
                masked_logits: np.ndarray = mask_logits(raw_logits, legal_tokens)
                max_index: int = int(np.argmax(masked_logits))
                token_ids.append(max_index)
                output_result += model.decode([max_index])

                if max_index == quote_token[0]:
                    for name, token in candidates.items():
                        if len(token) == position:
                            selected_function = name
                            print(selected_function)
                            current_state = States.COMA
                    break

                candidates = {name: token for name,
                              token in candidates.items() if (
                                  len(token) > position and
                                  token[position] == max_index)}
                position += 1

            continue

        elif current_state == States.COMA:
            legal_pieces = [',']
            current_state = States.PARAM_OPEN_QUOTE

        elif current_state == States.PARAM_OPEN_QUOTE:
            legal_pieces = ['"']
            current_state = States.PARAMETERS

        elif current_state == States.PARAMETERS:
            legal_pieces = ['parameters']
            current_state = States.PARAM_CLOSE_QUOTE

        elif current_state == States.PARAM_CLOSE_QUOTE:
            legal_pieces = ['"']
            current_state = States.PARAM_COLON

        elif current_state == States.PARAM_COLON:
            legal_pieces = [':']
            current_state = States.END

        for piece in legal_pieces:
            raw_token = model.encode(piece)
            legal_tokens = [int(x)
                                       for x in raw_token.squeeze(0).tolist()]
            raw_logits = model.get_logits_from_input_ids(
                token_ids)
            masked_logits = mask_logits(raw_logits, legal_tokens)
            max_index = int(np.argmax(masked_logits))
            token_ids.append(max_index)
            output_result += model.decode([max_index])

    return output_result


output = generate_tokens("sum of 3 and 2", 10,
                         'data/input/functions_definition.json')
output += '""}'
data = json.loads(output)
with open('data/output/function_calls.json', 'w') as f:
    json.dump(data, f, indent=4)
end = time.time() - start
print(f"{end:.2f} s")
