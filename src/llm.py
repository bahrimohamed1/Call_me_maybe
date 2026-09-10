from typing import List
from llm_sdk import Small_LLM_Model
import json
import re
import numpy as np
from .parser import load_definitions, ValueSchema, FunctionDefinition
from enum import Enum, auto


model = Small_LLM_Model()


with open(model.get_path_to_vocab_file(), "r") as file:
    vocab: dict[str, int] = json.load(file)


vocab_ids: List[int] = list(vocab.values())


decoded_vocab: dict[int, str] = {
    token_id: model.decode([token_id])
    for token_id in vocab_ids
}


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


def mask_logits(
    raw_logits: List[float],
    legal_tokens: List[int]
) -> np.ndarray:
    logits = np.array(
        raw_logits,
        dtype=np.float32
    )

    mask = np.zeros(
        logits.shape,
        dtype=bool
    )

    mask[legal_tokens] = True
    logits[~mask] = -np.inf

    return logits


def decode_token(token_id: int) -> str:
    return decoded_vocab[token_id]


def get_prefix_tokens(
    current_text: str,
    targets: List[str]
) -> List[int]:
    legal_tokens: List[int] = []

    for token_id in vocab_ids:
        token_text = decode_token(
            token_id
        )

        if not token_text:
            continue

        candidate = (
            current_text + token_text
        )

        for target in targets:
            if target.startswith(candidate):
                legal_tokens.append(
                    token_id
                )
                break

    return legal_tokens


def generate_fixed_text(
    text: str,
    token_ids: List[int],
    output_result: str,
    generated_count: List[int]
) -> str:
    generated_text = ""

    while generated_text != text:
        legal_tokens = get_prefix_tokens(
            generated_text,
            [text]
        )

        if not legal_tokens:
            raise RuntimeError(
                "No valid token found while "
                f"generating '{text}'"
            )

        raw_logits = (
            model.get_logits_from_input_ids(
                token_ids
            )
        )

        logits = mask_logits(
            raw_logits,
            legal_tokens
        )

        max_index = int(
            np.argmax(logits)
        )

        token_text = decode_token(
            max_index
        )

        token_ids.append(
            max_index
        )

        output_result += token_text
        generated_text += token_text
        generated_count[0] += 1

    return output_result


def is_complete_json_number(
    text: str
) -> bool:
    pattern = (
        r"-?(0|[1-9]\d*)"
        r"(\.\d+)?"
        r"([eE][+-]?\d+)?"
    )

    return (
        re.fullmatch(
            pattern,
            text
        )
        is not None
    )


def is_valid_json_number_prefix(
    text: str
) -> bool:
    if text == "":
        return True

    position = 0
    length = len(text)

    if text[position] == "-":
        position += 1

        if position == length:
            return True

    if (
        position >= length
        or not text[position].isdigit()
    ):
        return False

    if text[position] == "0":
        position += 1

        if (
            position < length
            and text[position].isdigit()
        ):
            return False

    else:
        if text[position] not in "123456789":
            return False

        position += 1

        while (
            position < length
            and text[position].isdigit()
        ):
            position += 1

    if position == length:
        return True

    if text[position] == ".":
        position += 1

        if position == length:
            return True

        if not text[position].isdigit():
            return False

        while (
            position < length
            and text[position].isdigit()
        ):
            position += 1

    if position == length:
        return True

    if text[position] in "eE":
        position += 1

        if position == length:
            return True

        if text[position] in "+-":
            position += 1

            if position == length:
                return True

        if (
            position >= length
            or not text[position].isdigit()
        ):
            return False

        while (
            position < length
            and text[position].isdigit()
        ):
            position += 1

    return position == length


def is_safe_string_content(
    text: str
) -> bool:
    if not text:
        return False

    if "\\" in text:
        return False

    if '"' in text:
        return False

    for char in text:
        if ord(char) < 0x20:
            return False

    return True


def classify_string_tokens(
) -> tuple[List[int], List[int]]:
    continue_tokens: List[int] = []
    close_tokens: List[int] = []

    for token_id in vocab_ids:
        token_text = decode_token(
            token_id
        )

        if not token_text:
            continue

        if "\\" in token_text:
            continue

        if any(
            ord(char) < 0x20
            for char in token_text
        ):
            continue

        quote_count = token_text.count(
            '"'
        )

        if quote_count == 0:
            continue_tokens.append(
                token_id
            )
            continue

        if (
            quote_count == 1
            and token_text.endswith('"')
        ):
            content = token_text[:-1]

            if (
                not content
                or is_safe_string_content(
                    content
                )
            ):
                close_tokens.append(
                    token_id
                )

    return (
        continue_tokens,
        close_tokens
    )


STRING_CONTINUE_TOKENS, STRING_CLOSE_TOKENS = (
    classify_string_tokens()
)


def constrained_log_probabilities(
    raw_logits: List[float],
    legal_tokens: List[int]
) -> dict[int, float]:
    values = np.array(
        [
            raw_logits[token_id]
            for token_id in legal_tokens
        ],
        dtype=np.float64
    )

    maximum = float(
        np.max(values)
    )

    shifted = (
        values - maximum
    )

    log_sum = (
        maximum
        + float(
            np.log(
                np.exp(
                    shifted
                ).sum()
            )
        )
    )

    probabilities: dict[int, float] = {}

    for index, token_id in enumerate(
        legal_tokens
    ):
        probabilities[token_id] = (
            float(values[index])
            - log_sum
        )

    return probabilities


def top_token_ids(
    token_scores: dict[int, float],
    allowed_tokens: List[int],
    count: int
) -> List[int]:
    if not allowed_tokens:
        return []

    return sorted(
        allowed_tokens,
        key=lambda token_id: (
            token_scores[token_id]
        ),
        reverse=True
    )[:count]


def generate_string_value(
    token_ids: List[int],
    beam_width: int = 2,
    max_steps: int = 32
) -> tuple[List[int], str]:
    base_token_ids = list(
        token_ids
    )

    beams: list[
        tuple[List[int], str, float]
    ] = [
        (
            [],
            "",
            0.0
        )
    ]

    completed: list[
        tuple[List[int], str, float]
    ] = []

    legal_tokens = (
        STRING_CONTINUE_TOKENS
        + STRING_CLOSE_TOKENS
    )

    for _ in range(max_steps):
        next_beams: list[
            tuple[List[int], str, float]
        ] = []

        for (
            generated_ids,
            generated_text,
            score
        ) in beams:
            context_ids = (
                base_token_ids
                + generated_ids
            )

            raw_logits = (
                model.get_logits_from_input_ids(
                    context_ids
                )
            )

            token_scores = (
                constrained_log_probabilities(
                    raw_logits,
                    legal_tokens
                )
            )

            continuation_ids = top_token_ids(
                token_scores,
                STRING_CONTINUE_TOKENS,
                beam_width
            )

            closing_ids = top_token_ids(
                token_scores,
                STRING_CLOSE_TOKENS,
                1
            )

            for token_id in continuation_ids:
                token_text = decode_token(
                    token_id
                )

                next_beams.append(
                    (
                        generated_ids
                        + [token_id],
                        generated_text
                        + token_text,
                        score
                        + token_scores[
                            token_id
                        ]
                    )
                )

            for token_id in closing_ids:
                token_text = decode_token(
                    token_id
                )

                completed.append(
                    (
                        generated_ids
                        + [token_id],
                        generated_text
                        + token_text,
                        score
                        + token_scores[
                            token_id
                        ]
                    )
                )

        if not next_beams:
            break

        next_beams.sort(
            key=lambda candidate: (
                candidate[2]
            ),
            reverse=True
        )

        beams = next_beams[
            :beam_width
        ]

        if completed:
            completed.sort(
                key=lambda candidate: (
                    candidate[2]
                ),
                reverse=True
            )

            best_completed_score = (
                completed[0][2]
            )

            best_unfinished_score = (
                beams[0][2]
            )

            if (
                best_completed_score
                >= best_unfinished_score
            ):
                best = completed[0]

                return (
                    best[0],
                    best[1]
                )

    if completed:
        completed.sort(
            key=lambda candidate: (
                candidate[2]
            ),
            reverse=True
        )

        best = completed[0]

        return (
            best[0],
            best[1]
        )

    raise RuntimeError(
        "String generation did not terminate"
    )


def generate_tokens(
    prompt: str,
    path: str
) -> str:
    params: list[
        tuple[str, ValueSchema]
    ] = []

    definitions: List[FunctionDefinition] = load_definitions(
        path
    )

    context = (
        "Available functions:\n"
    )

    for function in definitions:
        context += (
            f"\nFunction: "
            f"{function.name}\n"
        )

        context += (
            "Description: "
            f"{function.description}\n"
        )

        context += (
            "Parameters:\n"
        )

        for (
            param_name,
            param_schema
        ) in function.parameters.items():
            context += (
                f"- {param_name}: "
                f"{param_schema.type}\n"
            )

    context += (
        f"\nUser request:\n"
        f"{prompt}\n"
    )

    context += (
        "\nGenerate the function call for "
        "the user request. "
        "The parameter values must be copied "
        "from the information provided by "
        "the user, not calculated or invented. "
        "Preserve the exact value of each "
        "argument, including its sign, "
        "decimal point, capitalization, "
        "and boolean value. "
        "For multiple arguments, assign each "
        "value to the parameter that it "
        "corresponds to in the request. "
        "Generate the function call now:\n"
    )

    raw_tensor = model.encode(
        context
    )

    token_ids: List[int] = [
        int(x)
        for x in (
            raw_tensor
            .squeeze(0)
            .tolist()
        )
    ]

    output_result = ""
    generated_count = [0]

    current_state = States.START

    available_functions: List[str] = [
        definition.name
        for definition in definitions
    ]

    while (
        current_state
        != States.END
    ):

        if current_state == States.START:
            output_result = (
                generate_fixed_text(
                    "{",
                    token_ids,
                    output_result,
                    generated_count
                )
            )

            current_state = (
                States.KEY_OPEN_QUOTE
            )

        elif (
            current_state
            == States.KEY_OPEN_QUOTE
        ):
            output_result = (
                generate_fixed_text(
                    '"',
                    token_ids,
                    output_result,
                    generated_count
                )
            )

            current_state = (
                States.NAME
            )

        elif current_state == States.NAME:
            output_result = (
                generate_fixed_text(
                    "name",
                    token_ids,
                    output_result,
                    generated_count
                )
            )

            current_state = (
                States.KEY_CLOSE_QUOTE
            )

        elif (
            current_state
            == States.KEY_CLOSE_QUOTE
        ):
            output_result = (
                generate_fixed_text(
                    '"',
                    token_ids,
                    output_result,
                    generated_count
                )
            )

            current_state = (
                States.COLON
            )

        elif current_state == States.COLON:
            output_result = (
                generate_fixed_text(
                    ":",
                    token_ids,
                    output_result,
                    generated_count
                )
            )

            current_state = (
                States.VALUE_OPEN_QUOTE
            )

        elif (
            current_state
            == States.VALUE_OPEN_QUOTE
        ):
            output_result = (
                generate_fixed_text(
                    '"',
                    token_ids,
                    output_result,
                    generated_count
                )
            )

            current_state = (
                States.FUNCTION_NAME
            )

        elif (
            current_state
            == States.FUNCTION_NAME
        ):
            generated_name = ""

            while True:
                legal_tokens = (
                    get_prefix_tokens(
                        generated_name,
                        available_functions
                    )
                )

                complete = (
                    generated_name
                    in available_functions
                )

                quote_tokens: List[int] = []

                if complete:
                    quote_tokens = (
                        get_prefix_tokens(
                            "",
                            ['"']
                        )
                    )

                    legal_tokens += (
                        quote_tokens
                    )

                if not legal_tokens:
                    raise RuntimeError(
                        "No legal token available "
                        "for function name"
                    )

                raw_logits = (
                    model
                    .get_logits_from_input_ids(
                        token_ids
                    )
                )

                logits = mask_logits(
                    raw_logits,
                    legal_tokens
                )

                max_index = int(
                    np.argmax(logits)
                )

                token_text = decode_token(
                    max_index
                )

                if (
                    complete
                    and max_index
                    in quote_tokens
                ):
                    token_ids.append(
                        max_index
                    )

                    output_result += (
                        token_text
                    )

                    generated_count[0] += 1

                    selected_function = (
                        generated_name
                    )

                    selected_definition = (
                        next(
                            definition
                            for definition
                            in definitions
                            if (
                                definition.name
                                == selected_function
                            )
                        )
                    )

                    params = list(
                        selected_definition
                        .parameters
                        .items()
                    )

                    current_state = (
                        States.COMA
                    )

                    break

                token_ids.append(
                    max_index
                )

                output_result += (
                    token_text
                )

                generated_name += (
                    token_text
                )

                generated_count[0] += 1

        elif current_state == States.COMA:
            output_result = (
                generate_fixed_text(
                    ",",
                    token_ids,
                    output_result,
                    generated_count
                )
            )

            current_state = (
                States.PARAM_OPEN_QUOTE
            )

        elif (
            current_state
            == States.PARAM_OPEN_QUOTE
        ):
            output_result = (
                generate_fixed_text(
                    '"',
                    token_ids,
                    output_result,
                    generated_count
                )
            )

            current_state = (
                States.PARAMETERS
            )

        elif (
            current_state
            == States.PARAMETERS
        ):
            output_result = (
                generate_fixed_text(
                    "parameters",
                    token_ids,
                    output_result,
                    generated_count
                )
            )

            current_state = (
                States.PARAM_CLOSE_QUOTE
            )

        elif (
            current_state
            == States.PARAM_CLOSE_QUOTE
        ):
            output_result = (
                generate_fixed_text(
                    '"',
                    token_ids,
                    output_result,
                    generated_count
                )
            )

            current_state = (
                States.PARAM_COLON
            )

        elif (
            current_state
            == States.PARAM_COLON
        ):
            output_result = (
                generate_fixed_text(
                    ":",
                    token_ids,
                    output_result,
                    generated_count
                )
            )

            current_state = (
                States.PARAM_OBJECT_OPEN
            )

        elif (
            current_state
            == States.PARAM_OBJECT_OPEN
        ):
            output_result = (
                generate_fixed_text(
                    "{",
                    token_ids,
                    output_result,
                    generated_count
                )
            )

            if params:
                current_state = (
                    States
                    .PARAM_NAME_OPEN_QUOTE
                )
            else:
                current_state = (
                    States
                    .PARAM_OBJECT_CLOSE
                )

        elif (
            current_state
            == States.PARAM_NAME_OPEN_QUOTE
        ):
            output_result = (
                generate_fixed_text(
                    '"',
                    token_ids,
                    output_result,
                    generated_count
                )
            )

            current_state = (
                States.PARAM_NAME
            )

        elif (
            current_state
            == States.PARAM_NAME
        ):
            param: tuple[
                str,
                ValueSchema
            ] = params.pop(0)

            output_result = (
                generate_fixed_text(
                    param[0],
                    token_ids,
                    output_result,
                    generated_count
                )
            )

            current_state = (
                States.PARAM_NAME_CLOSE_QUOTE
            )

        elif (
            current_state
            == States.PARAM_NAME_CLOSE_QUOTE
        ):
            output_result = (
                generate_fixed_text(
                    '"',
                    token_ids,
                    output_result,
                    generated_count
                )
            )

            current_state = (
                States.PARAM_NAME_COLON
            )

        elif (
            current_state
            == States.PARAM_NAME_COLON
        ):
            output_result = (
                generate_fixed_text(
                    ":",
                    token_ids,
                    output_result,
                    generated_count
                )
            )

            current_state = (
                States.PARAM_VALUE
            )

        elif (
            current_state
            == States.PARAM_VALUE
        ):

            if (
                param[1].type
                == "boolean"
            ):
                value_text = ""

                targets = [
                    "true",
                    "false"
                ]

                while (
                    value_text
                    not in targets
                ):
                    legal_tokens = (
                        get_prefix_tokens(
                            value_text,
                            targets
                        )
                    )

                    if not legal_tokens:
                        raise RuntimeError(
                            "No legal boolean "
                            "token available"
                        )

                    raw_logits = (
                        model
                        .get_logits_from_input_ids(
                            token_ids
                        )
                    )

                    logits = mask_logits(
                        raw_logits,
                        legal_tokens
                    )

                    max_index = int(
                        np.argmax(logits)
                    )

                    token_text = (
                        decode_token(
                            max_index
                        )
                    )

                    token_ids.append(
                        max_index
                    )

                    output_result += (
                        token_text
                    )

                    value_text += (
                        token_text
                    )

                    generated_count[0] += 1

            elif (
                param[1].type
                == "number"
                or param[1].type
                == "integer"
            ):
                number_text = ""

                comma_tokens = (
                    get_prefix_tokens(
                        "",
                        [","]
                    )
                )

                brace_tokens = (
                    get_prefix_tokens(
                        "",
                        ["}"]
                    )
                )

                stop_tokens = (
                    comma_tokens
                    + brace_tokens
                )

                while True:
                    legal_tokens = []

                    for token_id in vocab_ids:
                        token_text = (
                            decode_token(
                                token_id
                            )
                        )

                        if not token_text:
                            continue

                        candidate = (
                            number_text
                            + token_text
                        )

                        if number_text == "":
                            candidate = (
                                candidate
                                .lstrip()
                            )

                        if any(
                            char
                            not in (
                                "0123456789"
                                "-.eE+"
                            )
                            for char
                            in candidate
                        ):
                            continue

                        if (
                            param[1].type
                            == "integer"
                            and any(
                                char in ".eE"
                                for char
                                in candidate
                            )
                        ):
                            continue

                        if (
                            is_valid_json_number_prefix(
                                candidate
                            )
                        ):
                            legal_tokens.append(
                                token_id
                            )

                    if (
                        is_complete_json_number(
                            number_text
                        )
                    ):
                        legal_tokens += (
                            stop_tokens
                        )

                    if not legal_tokens:
                        raise RuntimeError(
                            "No legal number "
                            "token available"
                        )

                    raw_logits = (
                        model
                        .get_logits_from_input_ids(
                            token_ids
                        )
                    )

                    logits = mask_logits(
                        raw_logits,
                        legal_tokens
                    )

                    max_index = int(
                        np.argmax(logits)
                    )

                    if (
                        is_complete_json_number(
                            number_text
                        )
                        and max_index
                        in stop_tokens
                    ):
                        break

                    token_text = (
                        decode_token(
                            max_index
                        )
                    )

                    token_ids.append(
                        max_index
                    )

                    output_result += (
                        token_text
                    )

                    number_text += (
                        token_text
                    )

                    if (
                        number_text
                        and number_text[0]
                        .isspace()
                    ):
                        number_text = (
                            number_text
                            .lstrip()
                        )

                    generated_count[0] += 1

            elif (
                param[1].type
                == "string"
            ):
                output_result = (
                    generate_fixed_text(
                        '"',
                        token_ids,
                        output_result,
                        generated_count
                    )
                )

                (
                    generated_ids,
                    generated_text
                ) = generate_string_value(
                    token_ids
                )

                token_ids.extend(
                    generated_ids
                )

                output_result += (
                    generated_text
                )

                generated_count[0] += (
                    len(generated_ids)
                )

            else:
                raise ValueError(
                    "Unsupported parameter "
                    f"type: {param[1].type}"
                )

            if params:
                current_state = (
                    States.PARAM_COMA
                )
            else:
                current_state = (
                    States
                    .PARAM_OBJECT_CLOSE
                )

        elif (
            current_state
            == States.PARAM_COMA
        ):
            output_result = (
                generate_fixed_text(
                    ",",
                    token_ids,
                    output_result,
                    generated_count
                )
            )

            current_state = (
                States
                .PARAM_NAME_OPEN_QUOTE
            )

        elif (
            current_state
            == States.PARAM_OBJECT_CLOSE
        ):
            output_result = (
                generate_fixed_text(
                    "}",
                    token_ids,
                    output_result,
                    generated_count
                )
            )

            current_state = (
                States.OBJECT_CLOSE
            )

        elif (
            current_state
            == States.OBJECT_CLOSE
        ):
            output_result = (
                generate_fixed_text(
                    "}",
                    token_ids,
                    output_result,
                    generated_count
                )
            )

            current_state = (
                States.END
            )

    return output_result
