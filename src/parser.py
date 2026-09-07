from pydantic import BaseModel, TypeAdapter, ValidationError, ConfigDict
from typing import List, Dict, Literal
import sys


class ValueSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal['number', 'string', 'boolean', 'integer']


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: Dict[str, ValueSchema]
    returns: ValueSchema


class FunctionCallingTest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prompt: str


function_adapter = TypeAdapter(List[FunctionDefinition])
function_calling_adapter = TypeAdapter(List[FunctionCallingTest])


def load_definitions(path: str) -> List[FunctionDefinition]:
    try:
        with open(path, 'rb') as file:
            definitions = function_adapter.validate_json(file.read())
            return definitions

    except (ValidationError, FileNotFoundError, OSError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def load_calling_tests(path: str) -> List[FunctionCallingTest]:
    try:
        with open(path, 'rb') as file:
            calling_tests = function_calling_adapter.validate_json(file.read())
            return calling_tests

    except (FileNotFoundError, ValidationError, OSError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)
