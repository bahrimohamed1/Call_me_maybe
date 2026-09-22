"""Byte-level finite-state grammar for flat JSON scalar arguments."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from src.schema import ScalarType


class ScalarGrammar(BaseModel):
    """Recognize a scalar value followed by its exact schema delimiter.

    States are immutable strings. A rejected byte returns None; a completed
    value and delimiter returns 'done'. Raw string bytes must form UTF-8.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    kind: ScalarType
    delimiter: Literal[",", "}"]

    def advance(self, state: str, token: bytes) -> str | None:
        """Return the state after all token bytes, or reject the token."""
        for byte in token:
            next_state = self.step(state, byte)
            if next_state is None:
                return None
            state = next_state
        return state

    def step(self, state: str, byte: int) -> str | None:
        """Consume one byte of a JSON value without accepting dead prefixes."""
        char = chr(byte)
        if state == "done":
            return None
        if state in ("start", "end") and char in " \t\r\n":
            return state
        if state == "end":
            return "done" if char == self.delimiter else None
        if state == "start":
            if self.kind == "string":
                return "string" if char == '"' else None
            if self.kind in ("number", "integer"):
                return number_step("start", char, self.kind, self.delimiter)
            choices = (
                ("true", "false") if self.kind == "boolean" else ("null",)
            )
            for choice in choices:
                if choice.startswith(char):
                    return "literal:" + choice[1:]
            return None
        if state.startswith("literal:"):
            remaining = state[8:]
            if char != remaining[0]:
                return None
            return "literal:" + remaining[1:] if len(remaining) > 1 else "end"
        if self.kind in ("number", "integer"):
            return number_step(state, char, self.kind, self.delimiter)
        return string_step(state, byte)


def number_step(
    state: str, char: str, kind: str, delimiter: str,
) -> str | None:
    """Recognize JSON numbers, excluding leading zeros and incomplete forms."""
    if state in ("zero", "digits", "fraction", "exponent_digits"):
        if char == delimiter:
            return "done"
        if char in " \t\r\n":
            return "end"
    if state == "start" and char == "-":
        return "sign"
    if state in ("start", "sign"):
        if char == "0":
            return "zero"
        return "digits" if char in "123456789" else None
    if state == "digits" and char in "0123456789":
        return "digits"
    if kind == "integer":
        return None
    if state in ("zero", "digits") and char == ".":
        return "dot"
    if state in ("dot", "fraction") and char in "0123456789":
        return "fraction"
    if state in ("zero", "digits", "fraction") and char in "eE":
        return "exponent"
    if state == "exponent" and char in "+-":
        return "exponent_sign"
    if state in ("exponent", "exponent_sign", "exponent_digits"):
        return "exponent_digits" if char in "0123456789" else None
    return None


def string_step(state: str, byte: int) -> str | None:
    """Recognize JSON escapes and well-formed, possibly split UTF-8 bytes."""
    if state == "escape":
        if chr(byte) in '"\\/bfnrt':
            return "string"
        return "hex4" if byte == ord("u") else None
    if state.startswith("hex"):
        if chr(byte) not in "0123456789abcdefABCDEF":
            return None
        count = int(state[-1]) - 1
        return f"hex{count}" if count else "string"
    if state.startswith("utf"):
        _, count_text, low_text, high_text = state.split(":")
        if not int(low_text) <= byte <= int(high_text):
            return None
        count = int(count_text) - 1
        return f"utf:{count}:128:191" if count else "string"
    if state != "string":
        return None
    if byte == 34:
        return "end"
    if byte == 92:
        return "escape"
    if 32 <= byte < 128:
        return "string"
    if 194 <= byte <= 223:
        return "utf:1:128:191"
    if 224 <= byte <= 239:
        low, high = (160, 191) if byte == 224 else (128, 191)
        if byte == 237:
            high = 159
        return f"utf:2:{low}:{high}"
    if 240 <= byte <= 244:
        low, high = (144, 191) if byte == 240 else (128, 191)
        if byte == 244:
            high = 143
        return f"utf:3:{low}:{high}"
    return None
