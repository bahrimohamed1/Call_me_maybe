import argparse
from .parser import load_calling_tests, load_definitions
from .parser import FunctionCallingTest, FunctionDefinition
from typing import List


def main() -> None:
    DEFAULT_FUNCTIONS_DEFINITION: str = "data/input/functions_definition.json"
    DEFAULT_INPUT_FILE: str = "data/input/function_calling_tests.json"
    DEFAULT_OUTPUT_FILE: str = "data/output/function_calls.json"

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--functions_definition',
        help="Select a functions definiton file",
        default=DEFAULT_FUNCTIONS_DEFINITION)
    parser.add_argument(
        '--input',
        help="Select an input file",
        default=DEFAULT_INPUT_FILE)
    parser.add_argument(
        '--output',
        help="Select an output file",
        default=DEFAULT_OUTPUT_FILE)

    args = parser.parse_args()
    function_definitions_path: str = args.functions_definition
    input_file_path: str = args.input
    output_file_path: str = args.output

    definitions: List[FunctionDefinition] = load_definitions(
        function_definitions_path)
    tests: List[FunctionCallingTest] = load_calling_tests(input_file_path)


if __name__ == '__main__':
    main()
