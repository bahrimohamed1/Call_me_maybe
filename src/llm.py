from typing import List, Dict, Optional
from llm_sdk import Small_LLM_Model
import time
import numpy as np
from .parser import FunctionDefinition, load_definitions, ValueSchema
from enum import Enum, auto
import json

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
    PARAM_COLON = auto()
    PARAM_OBJECT_OPEN = auto()
    PARAM_NAME_OPEN_QUOTE = auto()
    PARAM_NAME = auto()
    PARAM_NAME_CLOSE_QUOTE = auto()
    PARAM_NAME_COLON = auto()
    PARAM_VALUE = auto()
    PARAM_COMA = auto()
    PARAM_OBJECT_CLOSE = auto()
    OBJECT_CLOSE = auto()
    END = auto()


def mask_logits(raw_logits: List[float], legal_token: List[int]) -> np.ndarray:
    logits = np.array(raw_logits, dtype=np.float32)

    mask = np.zeros(logits.shape, dtype=bool)
    mask[legal_token] = True

    logits[~mask] = -np.inf
    return logits


def generate_tokens(prompt: str, max_tokens: int, path: str) -> str:
    params: list[tuple[str, ValueSchema]] = []
    definitions = load_definitions(path)

    context = "Available functions:\n"
    for function in definitions:
        context += f"\nFunction: {function.name}\n"
        context += f"Description: {function.description}\n"
        context += "Parameters:\n"
        for param_name, param_schema in function.parameters.items():
            context += f"- {param_name}: {param_schema.type}\n"
    context += f"\nUser request:\n{prompt}\n"
    context += "Select the single function that best matches the user's request. Use the function names, descriptions, and parameters above. Then provide the required arguments from the request.\n"

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
                masked_logits: np.ndarray = mask_logits(
                    raw_logits, legal_tokens)
                max_index: int = int(np.argmax(masked_logits))
                token_ids.append(max_index)
                output_result += model.decode([max_index])

                if max_index == quote_token[0]:
                    for name, token in candidates.items():
                        if len(token) == position:
                            selected_function = name
                            for definition in definitions:
                                if definition.name == selected_function:
                                    selected_definition = definition
                            params = list(
                                selected_definition.parameters.items())
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
            current_state = States.PARAM_OBJECT_OPEN

        elif current_state == States.PARAM_OBJECT_OPEN:
            legal_pieces = ['{']
            current_state = States.PARAM_NAME_OPEN_QUOTE

        elif current_state == States.PARAM_NAME_OPEN_QUOTE:
            legal_pieces = ['"']
            current_state = States.PARAM_NAME

        elif current_state == States.PARAM_NAME:
            param: tuple[str, ValueSchema] = params.pop(0)
            legal_pieces = [param[0]]
            current_state = States.PARAM_NAME_CLOSE_QUOTE

        elif current_state == States.PARAM_NAME_CLOSE_QUOTE:
            legal_pieces = ['"']
            current_state = States.PARAM_NAME_COLON

        elif current_state == States.PARAM_NAME_COLON:
            legal_pieces = [':']
            current_state = States.PARAM_VALUE

        elif current_state == States.PARAM_COMA:
            legal_pieces = [',']
            current_state = States.PARAM_NAME_OPEN_QUOTE

        elif current_state == States.PARAM_VALUE:
            if param[1].type == 'boolean':
                legal_tokens: List[int] = []
                legal_pieces = ['true', 'false']
                for piece in legal_pieces:
                    raw = model.encode(piece)
                    token = [int(x) for x in raw.squeeze(0).tolist()]
                    legal_tokens += token
                raw_logits = model.get_logits_from_input_ids(token_ids)
                logits = mask_logits(raw_logits, legal_tokens)
                max_index = int(np.argmax(logits))
                token_ids.append(max_index)
                output_result += model.decode([max_index])

            elif param[1].type == 'number':
                number_text = ""
                finished = False

                while not finished:
                    legal_pieces: List[str] = []

                    if number_text == "":
                        legal_pieces = [
                            '-', '0', '1', '2', '3', '4',
                            '5', '6', '7', '8', '9'
                        ]

                    elif number_text == "-":
                        legal_pieces = [
                            '0', '1', '2', '3', '4',
                            '5', '6', '7', '8', '9'
                        ]

                    elif number_text.endswith('.'):
                        legal_pieces = [
                            '0', '1', '2', '3', '4',
                            '5', '6', '7', '8', '9'
                        ]

                    elif '.' in number_text:
                        legal_pieces = [
                            '0', '1', '2', '3', '4',
                            '5', '6', '7', '8', '9',
                            ',', '}'
                        ]

                    else:
                        legal_pieces = [
                            '0', '1', '2', '3', '4',
                            '5', '6', '7', '8', '9',
                            '.', ',', '}'
                        ]

                    legal_tokens: List[int] = []
                    stop_tokens: List[int] = []

                    for piece in legal_pieces:
                        raw = model.encode(piece)
                        token = [int(x) for x in raw.squeeze(0).tolist()]
                        legal_tokens += token

                        if piece in [',', '}']:
                            stop_tokens += token

                    raw_logits = model.get_logits_from_input_ids(token_ids)
            
                    logits = mask_logits(raw_logits, legal_tokens)
                    print('2:', logits[17])
                    print('-:', logits[12])
                    max_index = int(np.argmax(logits))
                    if max_index in stop_tokens:
                        finished = True
                        continue

                    token_ids.append(max_index)

                    decoded_piece = model.decode([max_index])
                    output_result += decoded_piece
                    number_text += decoded_piece
            if params:
                current_state = States.PARAM_COMA
            else:
                current_state = States.PARAM_OBJECT_CLOSE

            continue

        elif current_state == States.PARAM_OBJECT_CLOSE:
            legal_pieces = ['}']
            current_state = States.OBJECT_CLOSE

        elif current_state == States.OBJECT_CLOSE:
            legal_pieces = ['}']
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


# output = generate_tokens("enable the feature", 10,
#                          'data/input/functions_definition.json')
# print(output)
output = generate_tokens("sum of -2 and 5", 10,
                         'data/input/functions_definition.json')
print(output)
# output = generate_tokens("greet simo", 10,
#                          'data/input/functions_definition.json')
# print(output)
# output = generate_tokens("sum of 2 and 5", 10,
#                          'data/input/functions_definition.json')
# print(output)
# output = generate_tokens("sum of 2 and 5", 10,
#                          'data/input/functions_definition.json')
# print(output)
# output = generate_tokens("sum of 2 and 5", 10,
#                          'data/input/functions_definition.json')
# print(output)
print(model.encode('-2'))
end = time.time() - start
print(f"{end:.2f} s")
