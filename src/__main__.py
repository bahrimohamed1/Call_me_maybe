import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
from .llm import generate_tokens
from .parser import FunctionCallingTest, load_calling_tests, load_definitions


DEFAULT_FUNCTIONS_DEFINITION = "data/input/functions_definition.json"
DEFAULT_INPUT_FILE = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT_FILE = "data/output/function_calls.json"


def main() -> None:
    """Run function calling generation for every input prompt."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--functions_definition",
        help="Select a functions definition file",
        default=DEFAULT_FUNCTIONS_DEFINITION,
    )
    parser.add_argument(
        "--input",
        help="Select an input file",
        default=DEFAULT_INPUT_FILE,
    )
    parser.add_argument(
        "--output",
        help="Select an output file",
        default=DEFAULT_OUTPUT_FILE,
    )

    args = parser.parse_args()

    function_definitions_path: str = args.functions_definition
    input_file_path: str = args.input
    output_file_path: str = args.output

    load_definitions(function_definitions_path)
    tests: List[FunctionCallingTest] = load_calling_tests(input_file_path)

    results: List[Dict[str, Any]] = []

    for test in tests:
        generated_call = json.loads(
            generate_tokens(
                test.prompt,
                function_definitions_path,
            )
        )
        results.append(
            {
                "prompt": test.prompt,
                "name": generated_call["name"],
                "parameters": generated_call["parameters"],
            }
        )

    output_path = Path(output_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
