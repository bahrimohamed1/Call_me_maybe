from pydantic import BaseModel, TypeAdapter, ValidationError, ConfigDict
from typing import List, Dict, Literal
import sys


class ValueSchema(BaseModel):
    """Represent the supported schema for a single value."""
    model_config = ConfigDict(extra="forbid")
    type: Literal['number', 'string', 'boolean', 'integer']


class FunctionDefinition(BaseModel):
    """Represent one available function and its parameter schema."""
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: Dict[str, ValueSchema]
    returns: ValueSchema


class FunctionCallingTest(BaseModel):
    """Represent one function-calling test prompt."""
    model_config = ConfigDict(extra='forbid')
    prompt: str


function_adapter = TypeAdapter(List[FunctionDefinition])
function_calling_adapter = TypeAdapter(List[FunctionCallingTest])


def load_definitions(path: str) -> List[FunctionDefinition]:
    """Load and validate function definitions from a JSON file."""
    try:
        with open(path, 'rb') as file:
            definitions = function_adapter.validate_json(file.read())
            return definitions

    except (ValidationError, FileNotFoundError, OSError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def load_calling_tests(path: str) -> List[FunctionCallingTest]:
    """Load and validate function-calling tests from a JSON file."""
    try:
        with open(path, 'rb') as file:
            calling_tests = function_calling_adapter.validate_json(file.read())
            return calling_tests

    except (FileNotFoundError, ValidationError, OSError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)
