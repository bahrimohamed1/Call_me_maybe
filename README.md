*This project has been created as part of the 42 curriculum by mbahri.*

# Call Me Maybe

## Description

Call Me Maybe is an introduction to function calling with Large Language
Models.

The goal of the project is to transform a natural-language user request
into a structured function call while guaranteeing that the generated
output respects the expected JSON format and the schema of the selected
function.

The project uses the provided LLM SDK and implements constrained
decoding instead of relying on the model to spontaneously generate valid
JSON.

The program: - reads function definitions from a JSON file; - reads user
prompts from an input JSON file; - asks the LLM to select the
appropriate function and generate its arguments; - constrains token
generation according to the expected JSON structure and parameter
types; - writes the generated function calls to a JSON output file.

## Instructions

### Requirements

The project requires Python 3.10 or later and uses `uv` for dependency
management.

Install the dependencies with:

``` bash
make install
```

Run the project with:

``` bash
make run
```

The program can also be executed directly with:

``` bash
uv run python -m src
```

Custom files can be provided with:

``` bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calls.json
```

Run the project using Python's debugger with:

``` bash
make debug
```

Run the code-quality checks with:

``` bash
make lint
```

Clean temporary files and caches with:

``` bash
make clean
```

## Algorithm Explanation

The project implements constrained token decoding.

For every generation step:

1.  The current token sequence is sent to the LLM.
2.  The LLM returns the logits for every token in its vocabulary.
3.  The decoder determines which tokens are valid according to the
    current generation state.
4.  Invalid tokens are masked by replacing their logits with negative
    infinity.
5.  The highest-scoring valid token is selected.
6.  The selected token is appended to the generated sequence.
7.  The process continues until the complete function call has been
    generated.

A state machine controls the expected JSON structure, including object
delimiters, JSON keys, the selected function name, parameter names,
parameter values, commas, and colons.

The function name is constrained to the functions provided in the
function-definition file.

Parameter generation depends on the parameter schema. Numbers are
validated as valid JSON-number prefixes while they are generated.
Boolean values are constrained to `true` or `false`. String values use
constrained generation while allowing the LLM to determine their
semantic content.

For string parameters, a small beam search is used so that the decoder
can consider possible closing tokens instead of always following a
single greedy continuation.

## Design Decisions

Pydantic is used to validate the input function definitions and test
cases.

The input parser, command-line interface, LLM generation logic, and
constrained decoding responsibilities are kept separated.

The decoder uses the model vocabulary to determine which token IDs can
legally continue the current output.

A state machine is used because the expected output format is structured
and predictable. Each state is responsible for one part of the generated
JSON.

The project focuses on flat primitive parameter types and avoids
unnecessary complexity in the mandatory implementation.

The constrained decoder controls the JSON structure, while the LLM
remains responsible for understanding the user's request, selecting the
appropriate function, and generating the parameter values.

A compact Logit Masking X-Ray visualization is displayed during
generation. It shows some of the model's highest raw-logit tokens and
the highest-scoring tokens that remain legal after applying the
constraints.

## Performance Analysis

The implementation was tested on the provided function-calling examples.

A complete run of 11 test prompts takes approximately 3.2 minutes on the
development machine.

The generated calls are constrained during decoding so that structurally
invalid tokens cannot be selected.

The implementation successfully generates function calls involving
numerical parameters, negative numbers, strings, square-root requests,
and regular-expression substitution arguments.

The constrained approach improves reliability because JSON structure and
parameter constraints do not depend entirely on the model following
instructions correctly.

## Challenges Faced

One challenge was handling tokenization. A visible value does not
necessarily correspond to one token or one character. The tokenizer can
contain tokens representing multiple characters, spaces, numbers, or
parts of JSON syntax.

For this reason, legal-token detection operates on the model vocabulary
rather than assuming that values are generated character by character.

Another issue occurred with negative numbers. The tokenizer could
produce a token containing both whitespace and the negative sign, such
as `" -"`. Number-prefix validation was adjusted so that valid leading
whitespace from such tokens does not prevent the correct negative value
from being generated.

String generation was another challenge. Greedy decoding could
repeatedly extend some values, particularly regular expressions, instead
of selecting a closing quote.

A small beam search was introduced for string values so that promising
completed strings can compete with continuing candidates while keeping
the rest of the constrained-decoding architecture unchanged.

## Testing Strategy

The implementation was tested using multiple prompts covering the
available function definitions.

The tests include: - addition with positive values; - addition with
negative values; - greeting generation; - string reversal; - square-root
calls; - regular-expression substitutions.

Generated outputs are parsed as JSON and written using the required
structure:

``` json
{
  "prompt": "Greet john",
  "name": "fn_greet",
  "parameters": {
    "name": "john"
  }
}
```

The project is also checked using `flake8` and `mypy`.

Malformed or missing input files are handled through validation and
exception handling.

## Example Usage

With the following prompt:

``` text
What is the sum of -2.0 and 3?
```

The program can generate:

``` json
{
  "prompt": "What is the sum of -2.0 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": -2.0,
    "b": 3.0
  }
}
```

For:

``` text
Greet john
```

the generated call is:

``` json
{
  "prompt": "Greet john",
  "name": "fn_greet",
  "parameters": {
    "name": "john"
  }
}
```

## Resources

Classic references related to the concepts and tools used in this
project:

-   Python Documentation --- language reference, standard library, file
    handling, exceptions, typing, and command-line utilities.
-   Python `json` Documentation --- JSON parsing and serialization.
-   NumPy Documentation --- array manipulation, logits processing,
    masking, and `argmax`.
-   Pydantic Documentation --- data models and validation of function
    definitions and input tests.
-   mypy Documentation --- static type checking.
-   JSON specification (RFC 8259) --- JSON grammar, strings, numbers,
    objects, and structural validity.
-   42 Call Me Maybe subject --- project requirements,
    constrained-decoding objectives, expected input/output format, and
    evaluation constraints.

### AI Usage

AI was used as a learning, debugging, and review assistant during
development. It was not used as a replacement for the
constrained-decoding mechanism: JSON and schema correctness are enforced
by the implementation itself.

AI assistance was used for the following tasks and parts of the project:

-   **Project understanding:** clarifying the subject requirements and
    the distinction between normal LLM generation and constrained
    function calling.
-   **Code review and debugging:** reviewing implementation decisions
    and identifying edge cases while the project was being tested.
-   **Documentation:** helping organize and draft this README according
    to the sections required by the subject.

## Example

Input prompt:

``` text
Replace all numbers in "Hello 34 I'm 233 years old" with NUMBERS
```

Generated function call:

``` json
{
  "prompt": "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS",
  "name": "fn_substitute_string_with_regex",
  "parameters": {
    "source_string": "Hello 34 I'm 233 years old",
    "regex": "([0-9]+)",
    "replacement": "NUMBERS"
  }
}
```
