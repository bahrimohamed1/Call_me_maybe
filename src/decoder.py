"""Generate function calls using only the subject's public SDK interface."""

import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from src.grammar import ScalarGrammar
from src.schema import FunctionDefinition, Result, normalize_numbers, read_json


def vocabulary_bytes(path: Path) -> dict[int, bytes]:
    """Interpret Qwen's byte-level BPE vocabulary; tokenization stays in SDK.

    The vocabulary encodes each byte as a printable Unicode character. Undo
    that representation, retaining incomplete UTF-8 pieces for the grammar.
    """
    visible = list(range(33, 127)) + list(range(161, 173))
    visible += list(range(174, 256))
    characters = visible.copy()
    extra = 0
    for byte in range(256):
        if byte not in visible:
            visible.append(byte)
            characters.append(256 + extra)
            extra += 1
    byte_map = {chr(char): byte for char, byte in zip(characters, visible)}
    raw = TypeAdapter(dict[str, int]).validate_python(read_json(path))
    vocabulary: dict[int, bytes] = {}
    for text, token_id in raw.items():
        if text and not text.startswith("<|"):
            try:
                vocabulary[token_id] = bytes(byte_map[char] for char in text)
            except KeyError as error:
                message = "Unsupported vocabulary byte encoding."
                raise ValueError(message) from error
    if not vocabulary or min(vocabulary) < 0:
        raise ValueError("The SDK returned an empty or invalid vocabulary.")
    return vocabulary


class Decoder(BaseModel):
    """Mask invalid logits and greedily generate schema-valid scalar values."""

    model_config = ConfigDict(extra="forbid", strict=True)
    sdk: Any = Field(repr=False, exclude=True)
    vocabulary: dict[int, bytes] = Field(repr=False)
    visualize: bool = False
    generated_tokens: int = 0

    def encode(self, text: str) -> list[int]:
        """Convert the public encode result into the SDK's input ID list."""
        rows = TypeAdapter(list[list[int]]).validate_python(
            self.sdk.encode(text).tolist(), strict=True,
        )
        if len(rows) != 1:
            raise ValueError("SDK encode must return one row of token IDs.")
        return rows[0]

    def select(
        self, context: list[int], allowed: list[int], label: str,
    ) -> int:
        """Set invalid logits to negative infinity before greedy selection."""
        if len(context) > 8192:
            raise ValueError("Request exceeds the 8192-token context limit.")
        logits = np.asarray(
            self.sdk.get_logits_from_input_ids(context), dtype=np.float64,
        )
        if logits.ndim != 1 or not logits.size:
            raise ValueError("SDK returned an invalid logits vector.")
        valid_ids = np.asarray(
            [index for index in allowed if 0 <= index < logits.size],
            dtype=np.int64,
        )
        masked = np.full(logits.shape, -np.inf)
        masked[valid_ids] = logits[valid_ids]
        masked[~np.isfinite(masked)] = -np.inf
        if not np.any(np.isfinite(masked)):
            raise ValueError(f"No valid finite token for {label}.")
        token_id = int(np.argmax(masked))
        context.append(token_id)
        self.generated_tokens += 1
        if self.visualize:
            fragment = self.sdk.decode([token_id])
            print(
                f"  {label:20} | {len(valid_ids):6} allowed | "
                f"{token_id:6} | {fragment!r}", file=sys.stderr,
            )
        return token_id

    def choose_function(
        self, context: list[int], functions: list[FunctionDefinition],
    ) -> FunctionDefinition:
        """Choose among supplied names using constrained LLM token scores."""
        candidates = [
            (self.encode(json.dumps(function.name)), function)
            for function in functions
        ]
        position = 0
        while candidates:
            allowed = sorted({ids[position] for ids, _ in candidates})
            token_id = self.select(context, allowed, "function name")
            candidates = [
                (ids, function) for ids, function in candidates
                if ids[position] == token_id
            ]
            position += 1
            completed = [
                function for ids, function in candidates
                if len(ids) == position
            ]
            if completed:
                return completed[0]
        raise ValueError("No function name matches the constrained tokens.")

    def value(
        self, context: list[int], grammar: ScalarGrammar, label: str,
    ) -> bytes:
        """Generate one scalar plus delimiter, rejecting unfinished values."""
        state = "start"
        output = bytearray()
        for _ in range(256):
            transitions: dict[int, str] = {}
            for token_id, fragment in self.vocabulary.items():
                next_state = grammar.advance(state, fragment)
                if next_state is not None:
                    transitions[token_id] = next_state
            token_id = self.select(context, list(transitions), label)
            state = transitions[token_id]
            output.extend(self.vocabulary[token_id])
            if state == "done":
                return bytes(output)
        raise ValueError(f"Generation limit reached for argument {label!r}.")

    def generate(
        self, prompt: str, functions: list[FunctionDefinition],
    ) -> Result:
        """Choose a function, constrain arguments, and validate the call."""
        definitions = json.dumps(
            [item.model_dump(exclude_defaults=True) for item in functions],
            ensure_ascii=True, separators=(",", ":"),
        )
        instruction = (
            "Select the function fulfilling the user's request. Extract its "
            "arguments; do not execute the function or compute its result. "
            "Copy string arguments exactly, preserving spelling and case. "
            "Never add leading or trailing spaces to argument strings. "
            "Preserve negative signs and decimal values in numeric arguments. "
            "Write numbers without leading zeros, as JSON requires. "
            "Return only JSON with name and parameters. Available functions:\n"
            + definitions
        )
        chat = (
            "<|im_start|>system\n" + instruction + "<|im_end|>\n"
            "<|im_start|>user\n" + prompt + "<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            '{"name": '
        )
        context = self.encode(chat)
        function = self.choose_function(context, functions)
        prefix = ', "parameters": {'
        context.extend(self.encode(prefix))
        parameters = bytearray(b"{")
        items = list(function.parameters.items())
        for index, (name, parameter) in enumerate(items):
            key = json.dumps(name, ensure_ascii=True) + ":"
            context.extend(self.encode(key))
            parameters.extend(key.encode("ascii"))
            grammar = ScalarGrammar(
                kind=parameter.type,
                delimiter="," if index + 1 < len(items) else "}",
            )
            parameters.extend(self.value(context, grammar, name))
        if not items:
            parameters.extend(b"}")
        arguments = cast(dict[str, Any], json.loads(parameters))
        result = Result(
            prompt=prompt, name=function.name, parameters=arguments,
        )
        return normalize_numbers(result, function)
