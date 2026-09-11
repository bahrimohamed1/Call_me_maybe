"""Command-line entry point with clear errors and atomic JSON output."""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from src.schema import Result, load_inputs


def parse_arguments() -> argparse.Namespace:
    """Parse the subject's file options and optional token visualization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--functions_definition", type=Path,
        default=Path("data/input/functions_definition.json"),
    )
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/input/function_calling_tests.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/output/function_calling_results.json"),
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="show each selected token and allowed-token count on stderr",
    )
    return parser.parse_args()


def write_results(path: Path, results: list[Result]) -> None:
    """Replace output only after all results have validated and serialized."""
    payload = json.dumps(
        [result.model_dump() for result in results],
        indent=2, ensure_ascii=True, allow_nan=False,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=".function-calls-", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(arguments: argparse.Namespace) -> None:
    """Validate inputs, load the model, and process prompts in order."""
    output = Path(arguments.output)
    sources = [Path(arguments.functions_definition), Path(arguments.input)]
    if output.resolve() in {path.resolve() for path in sources}:
        raise ValueError("Output must not overwrite either input file.")
    functions, requests = load_inputs(sources[0], sources[1])
    started = time.perf_counter()
    results: list[Result] = []
    if requests:
        from llm_sdk import Small_LLM_Model
        from src.decoder import Decoder, vocabulary_bytes

        print("Loading Qwen/Qwen3-0.6B...", file=sys.stderr)
        sdk = Small_LLM_Model(trust_remote_code=False)
        decoder = Decoder(
            sdk=sdk,
            vocabulary=vocabulary_bytes(Path(sdk.get_path_to_vocab_file())),
            visualize=arguments.visualize,
        )
        for index, request in enumerate(requests, start=1):
            try:
                result = decoder.generate(request.prompt, functions)
            except Exception as error:
                raise ValueError(f"Prompt {index}: {error}") from error
            results.append(result)
            print(
                f"[{index}/{len(requests)}] {result.name}", file=sys.stderr,
            )
    write_results(output, results)
    elapsed = time.perf_counter() - started
    print(f"Wrote {len(results)} calls to {output} in {elapsed/60:.2f}m.")


def main() -> int:
    """Turn operational failures into a readable error and nonzero status."""
    try:
        run(parse_arguments())
    except KeyboardInterrupt:
        print("Error: interrupted; output was not replaced.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
