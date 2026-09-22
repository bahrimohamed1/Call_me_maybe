*This project has been created as part of the 42 curriculum by mbahri.*

# Call me maybe

## Description

Translate natural-language requests into function calls using the provided
`llm_sdk` and **Qwen/Qwen3-0.6B**. Each result contains exactly `prompt`, `name`,
and `parameters`. The program selects a function and extracts its arguments;
it does not execute the function or return the result of the requested operation.

Constrained decoding restricts the model's choices before token selection.
The supplied function definitions determine the available names, argument keys,
and argument types. There are no prompt-to-function heuristics or hardcoded
answers. Supported arguments are flat `string`, `number`, `integer`, `boolean`,
and `null` values. Nested arguments are outside this implementation's scope.

## Instructions

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
these commands from the project root:

```sh
uv sync
uv run python -m src
```

The code uses Python 3.10-compatible syntax; `.python-version` selects Python
3.12 for the development environment. `uv sync` creates `.venv` and installs
the locked dependencies, including NumPy, Pydantic, and the local SDK.
Keep `llm_sdk/` alongside `src/`. The first model run needs internet access
to download Qwen's weights and tokenizer into the SDK's normal model cache.
Subsequent runs can reuse those files. Device selection is left to the supplied
SDK: MPS on a supported Mac, then CUDA, otherwise CPU.

Default paths:

| Purpose | Path |
| --- | --- |
| Functions | `data/input/functions_definition.json` |
| Prompts | `data/input/function_calling_tests.json` |
| Generated output | `data/output/function_calling_results.json` |

The default output name follows the subject's mandatory output specification
(V.4). Its usage example also shows `function_calls.json`; use `--output` to
select that filename explicitly. The provided reference file is not read by
the implementation.

### Example usage

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json

uv run python -m src --visualize
```

`--visualize` is the only bonus feature. It prints the current field, number of
allowed tokens, selected token ID, and readable token fragment to stderr. It
does not change the JSON output. A fragment containing part of a UTF-8 character
may display as a replacement character; the grammar preserves its original
bytes until the complete character is generated.

```text
  message              | 147089 allowed |   2403 | 'All'
  message              | 147089 allowed |   2797 | ' clear'
```

Make targets:

```sh
make install          # uv sync
make run              # main command; accepts ARGS="--input ... --output ..."
make debug            # Python's pdb debugger; accepts the same ARGS
make lint             # flake8 and the subject's exact required mypy flags
make lint-strict      # flake8 and mypy --strict
make clean            # Python and checker caches; keeps model data and .venv
```

Missing files, invalid JSON, invalid schemas, model failures, generation limits,
and output errors produce `Error: ...` on stderr with exit status 1. Ctrl-C
returns 130. Argument usage errors use argparse's status 2. An empty prompt
array writes `[]` without loading the model. An empty function list is rejected.
Individual empty prompt strings are accepted, although they cannot establish
an unambiguous intended call.

## Algorithm explanation

1. **Validate inputs.** Strict Pydantic models reject wrong types and unexpected
   keys. JSON parsing rejects duplicate keys and nonstandard `NaN`/`Infinity`
   constants. Function names must be unique. Every declared parameter is required.
2. **Build model context.** Describe all functions using the input definitions
   and append the user's request using Qwen's ChatML format. The documented
   empty thinking block starts a direct answer. Tokenization uses `sdk.encode`.
3. **Constrain the function name.** Encode each JSON-quoted function name.
   At each position, only tokens continuing one of the remaining name sequences
   are allowed. Shared prefixes remain possible until the LLM chooses a branch.
   All other logits become negative infinity before NumPy's `argmax` selects
   the next token. This is a token trie represented by a list of remaining paths.
4. **Constrain arguments.** Emit the known object punctuation and required keys
   in definition order. For each value, examine every vocabulary token against
   a finite-state grammar for that parameter's type and its following delimiter.
   A token is legal only if *every byte* preserves a valid prefix. Invalid tokens
   are masked to negative infinity before selection; the selected ID is appended
   to the model context for the next SDK call.
5. **Check and serialize.** Parse the completed argument object, verify exact
   keys and strict types against the chosen definition, and preserve the original
   prompt in the result. Normalize declared `number` arguments to finite floats
   for the moulinette (`16` becomes `16.0`); keep `integer` arguments as integers.
   Reject overflow or integer-to-float precision loss rather than changing a
   value. Serialize the complete batch with `allow_nan=False` and atomically
   replace the destination file.

The string grammar tracks quotes, backslash escapes, four-digit `\u` escapes,
and UTF-8 continuation bytes. The number grammar tracks sign, integer digits,
fraction, and exponent; it rejects leading zeros and unfinished values such as
`1e+`. Integers exclude decimal and exponent notation. Booleans and null allow
only their exact JSON literals. A value cannot finish before the required comma
or closing argument brace. Tokens crossing beyond that delimiter are rejected.

Vocabulary entries use a byte-to-printable-character representation. The code
reverses this representation to inspect token bytes, including partial Unicode
characters. It does **not** implement tokenization or BPE merges: `encode` and
optional `decode` remain SDK calls. Structural punctuation and keys have only
one permitted value and are inserted directly; all function-name choices and
argument content are selected from masked LLM logits.

## Design decisions

- Every project class inherits Pydantic `BaseModel`. Strict validation avoids
  silently converting `"3"` to a number or `true` to an integer.
- After validation, schema-driven numeric formatting ensures `number` arguments
  are Python floats when the moulinette reads the JSON. This is independent of
  the function name and of whether the model generated a decimal point.
- Only the SDK constructor, `encode`, `decode`,
  `get_logits_from_input_ids`, and `get_path_to_vocab_file` are used.
  No private SDK attributes or methods are accessed.
- The supplied SDK is unchanged. Its internal Torch/Transformers/Hugging Face
  dependencies are installed transitively; project code never imports them.
  Lint/type checks exclude this externally provided SDK, not project source.
- No additional models, nested schemas, recoded tokenizer, retries, or inference
  caches are implemented. Measured performance determines whether caching is
  warranted; the provided batch is below the subject's five-minute limit.
- A value has a 256-generated-token limit, and model contexts are limited to
  8192 tokens. Exceeding either produces a clear error, rather than truncating
  JSON or inventing missing arguments.
- A failed batch does not replace an existing output file or publish partial
  results. Output cannot overwrite either of the selected input files.

## Performance analysis

Validation is performed with the real Qwen/Qwen3-0.6B through the supplied SDK,
on this Apple Silicon Mac with model files already downloaded. Timing includes
model initialization, generation, validation, and output writing, but excludes
the initial model download. CPU-only hardware and longer batches may take more
time; there is no claim that arbitrary workloads finish within five minutes.

The recorded supplied-data run, using the exact default command, took **118.53
seconds** for 11 prompts. All 11 function choices were correct, all 11 results
were schema-valid, and all 11 matched the requested behavior. Nine results
matched the supplied reference literally (JSON numeric equality treats `2`
and `2.0` alike). Two regex arguments differed: `34|233` replaces the numbers
in the given source correctly, and `a|e|i|o|u` correctly replaces its vowels,
where the reference's negated character class does not. Behavioral checking
compares the actual substitutions on the supplied source; it does not claim
those regexes are equivalent for every possible source string.

The recorded new-function run took **92.07 seconds** for 13 prompts with
visualization enabled. All 11 unambiguous cases matched their independently
written expected calls exactly. Two additional ambiguous/empty requests were
checked for structural validity only, since neither specifies one correct call.
All 13 results were schema-valid. These two timing runs used the same source
revision before the number-format fix. After that fix, a fresh model run was
graded by the supplied public moulinette: **11/11 (100%)**, including square-root
arguments serialized as `16.0` and `144.0`.

| Final check | Result |
| --- | --- |
| Provided requests: function selection | 11/11 |
| Provided requests: requested behavior | 11/11 |
| Provided requests: literal reference equality | 9/11; regex differences explained above |
| New functions: exact function and argument equality | 11/11 |
| Public moulinette after number-format fix | 11/11 (100%) |
| Output schema validity across both batches | 24/24 |
| Provided / new-function batch times | 118.53s / 92.07s |
| Mandatory lint and strict mypy | Passed |
| Local grammar, validation, error-path, SDK-boundary checks | Passed |

Caching was not added because both measured batches complete comfortably
within five minutes using only the supplied public SDK methods.

Schema validity and semantic accuracy are different: the grammar enforces the
format and types, while the small model can still misunderstand an ambiguous
request or extract the wrong value. No malformed or schema-invalid result is
written, even on an error path.

## Challenges faced

- **Multi-character tokens:** accepting a token by its first character would
  allow it to cross a JSON boundary. The grammar checks every byte instead.
- **Split Unicode:** decoding isolated tokens can lose incomplete UTF-8 bytes.
  Grammar transitions operate on vocabulary bytes and validate continuations.
- **Incorrect reference regex:** the supplied reference for “replace vowels”
  uses `[^aeiouAEIOU]`, which matches non-vowels. Evaluation checks substitution
  behavior independently and preserves the supplied reference unchanged.
- **String padding:** an initial model run added a space to a replacement
  string. Fixed argument keys now end at the colon, so the model selects its
  own whitespace-bearing tokens. Ending a separately encoded key with a space
  created an unnatural token boundary. A general extraction instruction also
  forbids added padding; there is no prompt-specific correction. Numeric output
  formatting follows the declared schema for all functions.
- **Numeric extraction:** explicit numeric instructions preserve values and
  remove leading zeros when writing JSON, without converting them into strings.
  Numeric signs and zero-padded input are checked with new function definitions.
- **Typing across Python versions:** NumPy is constrained below 2.3 so its
  installed annotations remain compatible with the Python 3.10 mypy target.

## Testing strategy

Local development checks cover valid and invalid scalar grammar transitions,
every split point in Unicode/escaped tokens, 1,000 randomized string encodings,
strict argument typing, malformed and missing input files, atomic output,
invalid-logit masking, empty batches, and forbidden SDK access/imports.
Development test programs are excluded from submission, as requested in IV.3.

Real-model checks use all 11 supplied prompts and a separate batch with new
function definitions: booleans, integer times, concatenation, empty strings,
quotes, backslashes, multilingual text, large and decimal numbers, null, and
zero-parameter functions, plus two ambiguous/empty requests. Function names,
arguments, exact output keys, and
schema validity are checked independently. The visualization is exercised in
the new-function run. Both mandatory lint and strict mypy checks are run.

For a quick check of the default output after running the program:

```sh
uv run python -m json.tool data/output/function_calling_results.json
```

| Subject requirement | Implementation / verification |
| --- | --- |
| Python, annotations, PEP 257, flake8, mypy | `src/`, `make lint`, `make lint-strict` |
| Pydantic for all project classes | `schema.py`, `grammar.py`, `decoder.py` |
| Provided SDK and default Qwen model | Lazy SDK construction in `__main__.py` |
| LLM function choice and constrained decoding | Name paths and scalar masks in `decoder.py` |
| Required arguments and strict JSON types | `grammar.py`, final `validate_result` |
| Exact output keys and original prompt | `Result` and `write_results` |
| Custom paths and required defaults | CLI options in `__main__.py` |
| Error/resource handling | Context managers, bounded generation, atomic output |
| Dependency and Makefile requirements | `pyproject.toml`, `uv.lock`, `Makefile` |
| No output or test files in submission | `.gitignore` excludes `output/` and `tests/` |
| Limited bonus scope | `--visualize`; no caching needed |

## Resources

- [Python JSON documentation](https://docs.python.org/3/library/json.html):
  JSON parsing, serialization, and error behavior.
- [Pydantic strict mode](https://docs.pydantic.dev/latest/concepts/strict_mode/):
  validation without implicit type conversion.
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B): model use and
  the non-thinking response format. The supplied tokenizer configuration was
  also checked against the prompt format.
- [uv locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/):
  reproducible dependency installation.

AI assistance was used to read the subject, create local validation programs, analyze real-model results.