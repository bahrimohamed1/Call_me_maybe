"""Validate the subject's flat function definitions and result format."""

import json
import math
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


Scalar = Union[str, int, float, bool, None]
ScalarType = Literal["string", "number", "integer", "boolean", "null"]


class Parameter(BaseModel):
    """Describe one required, scalar function argument."""

    model_config = ConfigDict(extra="forbid", strict=True)
    type: ScalarType
    description: str = ""


class ReturnType(BaseModel):
    """Describe a return type; the function itself is never executed."""

    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal[
        "string", "number", "integer", "boolean", "null", "array", "object"
    ]


class FunctionDefinition(BaseModel):
    """Validate a named function and its required arguments."""

    model_config = ConfigDict(extra="forbid", strict=True)
    name: Annotated[str, Field(min_length=1)]
    description: str
    parameters: dict[str, Parameter]
    returns: ReturnType


class Request(BaseModel):
    """Preserve a natural-language request verbatim, including empty text."""

    model_config = ConfigDict(extra="forbid", strict=True)
    prompt: str


class Result(BaseModel):
    """Represent exactly the three output keys required by the subject."""

    model_config = ConfigDict(extra="forbid", strict=True)
    prompt: str
    name: str
    parameters: dict[str, Scalar]


def reject_constant(value: str) -> None:
    """Reject JavaScript non-finite constants that are not valid JSON."""
    raise ValueError(f"Invalid JSON constant: {value}")


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject repeated JSON keys instead of silently discarding data."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def read_json(path: Path) -> object:
    """Read strict UTF-8 JSON, reporting file and syntax errors together."""
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(
                stream, parse_constant=reject_constant,
                object_pairs_hook=unique_object,
            )
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read {path}: {error}") from error


def load_inputs(
    definitions_path: Path, requests_path: Path,
) -> tuple[list[FunctionDefinition], list[Request]]:
    """Load both files and reject ambiguous or unsupported definitions."""
    functions = TypeAdapter(list[FunctionDefinition]).validate_python(
        read_json(definitions_path), strict=True,
    )
    requests = TypeAdapter(list[Request]).validate_python(
        read_json(requests_path), strict=True,
    )
    if not functions:
        raise ValueError("At least one function definition is required.")
    names = [function.name for function in functions]
    if len(set(names)) != len(names):
        raise ValueError("Function names must be unique.")
    return functions, requests


def validate_result(result: Result, function: FunctionDefinition) -> None:
    """Check exact argument keys and types without coercing model output."""
    if result.name != function.name:
        raise ValueError("Generated an unknown function name.")
    if result.parameters.keys() != function.parameters.keys():
        raise ValueError("Generated argument keys do not match the schema.")
    for name, parameter in function.parameters.items():
        value = result.parameters[name]
        kind = parameter.type
        valid = (
            (kind == "string" and type(value) is str)
            or (kind == "boolean" and type(value) is bool)
            or (kind == "null" and value is None)
            or (kind == "integer" and type(value) is int)
            or (kind == "number" and type(value) in (int, float))
        )
        if not valid or (type(value) is float and not math.isfinite(value)):
            raise ValueError(f"Invalid {kind} argument: {name!r}")


def normalize_numbers(
    result: Result, function: FunctionDefinition,
) -> Result:
    """Validate a call and represent number arguments as lossless floats.

    Integer, boolean, string, and null parameters retain their declared types.
    Reject overflow or precision loss instead of silently changing a value.
    """
    validate_result(result, function)
    parameters = result.parameters.copy()
    for name, parameter in function.parameters.items():
        value = parameters[name]
        if parameter.type == "number" and isinstance(value, (int, float)):
            try:
                number = float(value)
            except OverflowError as error:
                message = f"Number argument {name!r} is too large."
                raise ValueError(message) from error
            if not math.isfinite(number) or number != value:
                raise ValueError(
                    f"Number argument {name!r} cannot be represented "
                    "exactly as a finite float."
                )
            parameters[name] = number
    return Result(
        prompt=result.prompt, name=result.name, parameters=parameters,
    )
