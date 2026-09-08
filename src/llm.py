from typing import List, Dict
from llm_sdk import Small_LLM_Model
import time
import numpy as np
from .parser import FunctionDefinition, load_definitions
from enum import Enum, auto

start = time.time()

model = Small_LLM_Model()


class States(Enum):
    START = auto()
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
    END = auto()


def mask_logits(raw_logits, legal_token) -> List[float]:
    masked_logits: List[float] = []
    for i, logit in enumerate(raw_logits, 0):
        if i not in legal_token:
            masked_logits.append(float('-inf'))
        else:
            masked_logits.append(logit)

    return masked_logits


def generate_tokens(prompt: str, max_tokens: int, path: str) -> None:
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
                masked_logits = mask_logits(raw_logits, legal_tokens)
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
            current_state = States.END

        for piece in legal_pieces:
            raw_token = model.encode(piece)
            legal_tokens: List[int] = [int(x)
                                       for x in raw_token.squeeze(0).tolist()]
            raw_logits: list[float] = model.get_logits_from_input_ids(
                token_ids)
            masked_logits = mask_logits(raw_logits, legal_tokens)
            max_index: int = int(np.argmax(masked_logits))
            token_ids.append(max_index)
            output_result += model.decode([max_index])

    print(output_result)


generate_tokens("greet simo", 10, 'data/input/functions_definition.json')
end = time.time() - start
print(f"{end:.2f} s")
