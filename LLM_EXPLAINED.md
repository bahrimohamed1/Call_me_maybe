# `llm.py` — Detailed Block-by-Block Explanation

This document explains the role of each major block in `llm.py`, why it exists, and how it contributes to constrained function calling.

The goal is not to explain every single line separately, but to make the full architecture understandable enough for a project defense.

---

## 1. Imports

```python
from typing import List
from llm_sdk import Small_LLM_Model
import json
import re
import numpy as np
from .parser import load_definitions, ValueSchema, FunctionDefinition
from enum import Enum, auto
```

### Why this block exists

Each import supports one specific part of the decoder:

- `List` is used for type hints such as lists of token IDs.
- `Small_LLM_Model` is the provided model interface.
- `json` is used to load the model vocabulary.
- `re` is used to validate complete JSON numbers.
- `numpy` is used for logits, masks, `argmax`, and beam-search scores.
- `load_definitions`, `ValueSchema`, and `FunctionDefinition` connect the decoder to the validated function definitions.
- `Enum` and `auto` are used to build the decoding state machine.

The important idea is that `llm.py` combines **LLM predictions** with **deterministic constraints**.

---

## 2. Model and vocabulary initialization

```python
model = Small_LLM_Model()

with open(model.get_path_to_vocab_file(), "r") as file:
    vocab: dict[str, int] = json.load(file)

vocab_ids: List[int] = list(vocab.values())

decoded_vocab: dict[int, str] = {
    token_id: model.decode([token_id])
    for token_id in vocab_ids
}
```

### Why we need this

The model works with **token IDs**, not directly with characters.

For constrained decoding, we need to know what each token ID represents as text.

Example:

```text
Token ID 42   -> "{"
Token ID 91   -> "name"
Token ID 712  -> " -"
```

The vocabulary is therefore essential because the decoder must answer:

> “If I allow this token, will the generated text still be valid?”

`decoded_vocab` is created once so we do not repeatedly call `model.decode()` for the same token.

### Example

If we are generating the exact word:

```text
parameters
```

the tokenizer might contain:

```text
"para"
"parameters"
"meter"
"parameters "
```

The decoder checks which tokens preserve the required prefix.

---

## 3. `States` — the JSON generation state machine

```python
class States(Enum):
    ...
```

### Why we need states

The output has a fixed structure:

```json
{
  "name": "fn_name",
  "parameters": {
    "argument": "value"
  }
}
```

Instead of asking the LLM to freely generate this entire JSON object, the program moves through controlled states.

Conceptually:

```text
START
  ↓
"name"
  ↓
:
  ↓
function name
  ↓
,
  ↓
"parameters"
  ↓
{
  ↓
parameter name
  ↓
parameter value
  ↓
}
```

Each state knows what kind of token is legal next.

This is the core idea behind the constrained decoder.

---

## 4. Logit X-Ray visualization

```python
def show_xray(...):
    ...
```

### What it does

This is the visualization bonus.

The LLM produces one score, called a **logit**, for every vocabulary token.

The X-Ray displays:

```text
RAW:
tokens the model originally prefers

LEGAL:
highest-scoring tokens that survive our constraints
```

### Example

Suppose the decoder currently needs `{`.

The model might prefer:

```text
"Sure"      24.1
"The"       21.8
"{"         18.5
```

But only `{` is legal.

The visualization demonstrates why constrained decoding is different from ordinary generation:

```text
RAW   : 'Sure' (24.1) | 'The' (21.8) | '{' (18.5)
LEGAL : '{' (18.5)
```

The model still provides intelligence, but the decoder controls validity.

---

## 5. `mask_logits()`

```python
def mask_logits(...):
    ...
```

### Why this function is central

The LLM returns logits for every possible token.

We create a boolean mask:

```text
legal token   -> keep original score
illegal token -> -∞
```

Then `np.argmax()` cannot select an illegal token.

### Example

Before masking:

```text
"hello" = 22
"{"     = 17
"name"  = 13
```

If only `{` is legal:

```text
"hello" = -∞
"{"     = 17
"name"  = -∞
```

Now the highest legal token is automatically `{`.

This is the actual **logit masking** required for constrained decoding.

---

## 6. `decode_token()`

```python
def decode_token(token_id: int) -> str:
    return decoded_vocab[token_id]
```

### Why this helper exists

The rest of the code constantly needs to convert:

```text
token ID -> readable token text
```

Instead of directly accessing the dictionary everywhere, this helper gives one clear interface.

Example:

```python
decode_token(123)
```

might return:

```text
"true"
```

---

## 7. `get_prefix_tokens()`

```python
def get_prefix_tokens(...):
    ...
```

### Purpose

This function finds every vocabulary token that can legally continue a known target string.

Example target:

```text
"name"
```

Current generated text:

```text
na
```

Possible token texts:

```text
"me"      ✅ -> "name"
"m"       ✅ -> "nam"
"x"       ❌ -> "nax"
"name"    ❌ -> "naname"
```

The function tests:

```python
target.startswith(current_text + token_text)
```

So it works with tokenizer pieces of different lengths.

### Why this matters

We cannot assume generation happens character by character.

A tokenizer can contain whole words, fragments, punctuation, or combinations.

---

## 8. `generate_fixed_text()`

```python
def generate_fixed_text(...):
    ...
```

### Purpose

Some output parts are completely deterministic:

```text
{
"
name
:
,
parameters
}
```

There is no reason to let the model invent them.

This function repeatedly:

1. determines which tokens can still form the required text;
2. gets model logits;
3. masks every illegal token;
4. picks the best legal token;
5. appends it.

### Example

To generate:

```text
parameters
```

the tokenizer might choose:

```text
"param"
"eters"
```

or:

```text
"parameters"
```

Both are valid as long as the final text is exactly `parameters`.

---

## 9. JSON number validation

Two functions work together:

```python
is_complete_json_number()
is_valid_json_number_prefix()
```

---

### `is_complete_json_number()`

Checks whether a number is already a complete valid JSON number.

Examples:

```text
12      ✅
-2      ✅
3.14    ✅
1e5     ✅
-       ❌
3.      ❌
1e      ❌
```

This matters because the decoder needs to know when it is allowed to stop generating the number.

---

### `is_valid_json_number_prefix()`

This function is more subtle.

During generation, incomplete values must temporarily be accepted.

For example:

```text
-
-2
-2.
-2.5
```

`-` is not a complete JSON number, but it is a valid **prefix** because it can become `-2`.

Likewise:

```text
1e
1e-
1e-3
```

The first two are incomplete but still potentially valid.

### Why we need both functions

```text
prefix validator  -> Can generation continue from here?
complete validator -> Can generation stop here?
```

That distinction is essential for constrained generation.

---

## 10. String token safety

```python
def is_safe_string_content(...):
    ...
```

### Purpose

JSON strings cannot contain arbitrary raw characters.

This helper rejects tokens containing:

- an unhandled backslash;
- an internal `"`;
- control characters.

The goal is to avoid accidentally creating invalid JSON strings.

Example:

```text
hello      ✅
cat        ✅
abc"def    ❌
newline control character ❌
```

---

## 11. `classify_string_tokens()`

```python
def classify_string_tokens(...):
    ...
```

### Why string generation is different

Strings are open-ended.

Unlike `"name"` or `{`, we do not know the target value in advance.

For example, the LLM may need to generate:

```text
john
```

or:

```text
([0-9]+)
```

So the vocabulary is divided into two groups.

### Group 1 — continuation tokens

Tokens that add normal content:

```text
"john"
"abc"
"([0"
```

### Group 2 — closing tokens

Tokens that end the JSON string.

The important detail is that a tokenizer may contain a token such as:

```text
)"
```

rather than separate tokens:

```text
)
"
```

So closing tokens may contain some final content followed by the quote.

This classification solved one of the difficult string/regex termination problems.

---

## 12. Precomputed string-token groups

```python
STRING_CONTINUE_TOKENS, STRING_CLOSE_TOKENS = (
    classify_string_tokens()
)
```

### Why do this once

Classifying the entire vocabulary is expensive.

The classification does not change between prompts, so we calculate it once at startup.

Then string generation can directly reuse both lists.

---

## 13. `constrained_log_probabilities()`

```python
def constrained_log_probabilities(...):
    ...
```

### Purpose

The string decoder uses a small beam search.

Beam search needs scores that can be added across multiple generated tokens.

Raw logits cannot be meaningfully added directly in the same way, so this function converts the legal logits into **log probabilities**.

Conceptually:

```text
raw legal logits
      ↓
normalize only over legal tokens
      ↓
log probabilities
```

### Why log probabilities

For a candidate consisting of multiple tokens:

```text
score = logP(token1)
      + logP(token2)
      + logP(token3)
```

This gives the score of the full candidate sequence.

---

## 14. `top_token_ids()`

```python
def top_token_ids(...):
    ...
```

### Purpose

Beam search does not keep every possible token.

That would explode computationally.

This helper sorts the allowed token IDs by score and returns only the best few.

Example with beam width `2`:

```text
token A = -0.4
token B = -0.8
token C = -2.1
token D = -4.0
```

We keep:

```text
A
B
```

and discard the weaker branches.

---

## 15. `generate_string_value()` — beam search

```python
def generate_string_value(...):
    ...
```

This is the most important special case in the file after the main state machine.

### Why greedy decoding was not enough

For normal fixed structures, choosing the highest legal token works well.

But strings can continue indefinitely.

Example regex:

```text
([0-9]+)
```

A greedy model might prefer:

```text
([0-9]+)|([0-9]+)|([0-9]+)|...
```

instead of closing the string.

### Beam-search idea

Rather than following only one path:

```text
best token
   ↓
best next token
   ↓
...
```

we temporarily keep two candidate paths:

```text
candidate A
candidate B
```

Each gets its own next-token logits.

Completed strings are stored separately from unfinished strings.

### Why `beam_width=2`

It is intentionally small.

The purpose is not to build a large general-purpose search engine. It is just enough to let a reasonable closing candidate compete against a greedy continuation.

### Why `max_steps=32`

This prevents accidental infinite generation.

If no string closes within the limit, the function eventually raises an error instead of hanging forever.

### Important scoring rule

A completed candidate can win when:

```text
best completed score >= best unfinished score
```

At that point the decoder has a strong reason to choose the finished string.

---

# 16. `generate_tokens()` — main constrained decoder

```python
def generate_tokens(prompt: str, path: str) -> str:
    ...
```

This function connects everything together.

---

## 16.1 Load function definitions

```python
definitions = load_definitions(path)
```

The available functions are loaded through the Pydantic-validated parser.

Example:

```json
{
  "name": "fn_greet",
  "parameters": {
    "name": {"type": "string"}
  }
}
```

The decoder therefore knows which function names and parameter schemas are valid.

---

## 16.2 Build the LLM context

The code constructs a prompt containing:

```text
Available functions
Descriptions
Parameter names and types
User request
Generation instruction
```

### Important distinction

The prompt helps the model understand **semantics**.

For example:

```text
"Greet john"
```

should lead toward:

```text
fn_greet
```

But the prompt is **not trusted to guarantee JSON validity**.

The constrained decoder does that.

So responsibilities are separated:

```text
LLM -> what does the user mean?

Decoder -> what outputs are structurally legal?
```

---

## 16.3 Encode the context

```python
raw_tensor = model.encode(context)
```

The textual context becomes model token IDs.

These token IDs form the initial generation context.

Every newly generated token is then appended to this list.

---

## 16.4 Initialize generation state

Important variables include:

```text
output_result
generated_count
current_state
available_functions
```

`output_result` stores the JSON text being constructed.

`current_state` starts at `START` and controls which generation rule is active.

---

# 17. Fixed JSON structure states

States such as:

```text
START
KEY_OPEN_QUOTE
NAME
KEY_CLOSE_QUOTE
COLON
COMA
PARAMETERS
...
```

mostly call:

```python
generate_fixed_text(...)
```

Example:

```text
START -> {
NAME  -> name
COLON -> :
```

This guarantees that structural JSON pieces cannot be hallucinated or misspelled.

---

# 18. Function-name generation

```python
elif current_state == States.FUNCTION_NAME:
```

This is where the model chooses the function.

### How it works

The legal targets are:

```python
available_functions
```

Suppose they are:

```text
fn_add_numbers
fn_greet
fn_reverse_string
```

At every token step, only tokens that remain prefixes of one of these names survive.

Example:

```text
generated: fn_g

fn_greet          ✅ still possible
fn_add_numbers    ❌ impossible
fn_reverse_string ❌ impossible
```

When the generated name exactly matches a function, the closing quote becomes legal.

### Why this satisfies function selection

The LLM still decides which legal path has the highest logits.

There is no manual rule such as:

```text
if "greet" in prompt -> choose fn_greet
```

The selection remains model-driven.

---

# 19. Parameter extraction from the selected function

After the function name is complete:

```python
params = list(
    selected_definition.parameters.items()
)
```

The decoder now knows exactly:

- which parameters must appear;
- their order;
- their schemas.

Example:

```json
"a": {"type": "integer"},
"b": {"type": "number"}
```

becomes conceptually:

```text
[
    ("a", integer),
    ("b", number)
]
```

---

# 20. Parameter-name generation

Parameter names are not invented by the model.

The program pops the next schema-defined parameter:

```python
param = params.pop(0)
```

and uses:

```python
generate_fixed_text(param[0], ...)
```

So if the function requires `source_string`, the output must contain exactly:

```json
"source_string"
```

This is another place where constraints guarantee schema compliance.

---

# 21. Boolean parameter generation

```python
if param[1].type == "boolean":
```

The only allowed targets are:

```text
true
false
```

`get_prefix_tokens()` restricts generation to those two possibilities.

The model still chooses which one has the stronger score based on the user request.

Example:

```text
"enable notifications"
```

might make `true` more probable.

The decoder does not infer the boolean manually.

---

# 22. Number and integer generation

```python
elif (
    param[1].type == "number"
    or param[1].type == "integer"
):
```

This branch scans vocabulary tokens and asks:

> If this token is appended to the current number, is the result still a valid JSON-number prefix?

### Example

Current:

```text
-2
```

Candidate token:

```text
.5
```

For a `number`:

```text
-2.5 ✅
```

For an `integer`:

```text
-2.5 ❌
```

### Stop tokens

Once the number is complete, delimiters such as:

```text
,
}
```

are added as conceptual stop choices.

If a delimiter receives the highest allowed score, number generation stops **without consuming it**.

The normal state machine generates that delimiter afterward.

This keeps parameter-value logic separate from JSON-structure logic.

---

# 23. Leading whitespace in number tokens

The code contains special handling using:

```python
lstrip()
```

### Why

A tokenizer can have a token representing:

```text
" -"
```

instead of simply:

```text
"-"
```

Rejecting that token because of its leading space can accidentally remove the LLM's preferred negative-number token.

So the internal validation ignores initial whitespace while the final output may still legally contain:

```json
"a": -2
```

Whitespace after `:` is valid JSON.

---

# 24. String parameter generation

```python
elif param[1].type == "string":
```

First, the opening quote is generated deterministically.

Then:

```python
generate_string_value(token_ids)
```

handles the unknown string content using the beam-search logic described earlier.

Example:

```text
User:
Greet john

Generated string parameter:
"john"
```

For regex substitution:

```text
"([0-9]+)"
```

can also be generated while allowing the decoder to recognize tokenizer tokens that include the final closing quote.

---

# 25. Unsupported types

```python
else:
    raise ValueError(...)
```

This prevents silent incorrect behavior.

If an unsupported schema somehow reaches the decoder, the program stops clearly rather than pretending it knows how to generate it.

---

# 26. Multiple parameters

After one parameter value:

```python
if params:
    current_state = States.PARAM_COMA
else:
    current_state = States.PARAM_OBJECT_CLOSE
```

This means the schema determines whether another argument must be generated.

Example with two parameters:

```json
{
  "a": 2,
  "b": 3
}
```

After `a`, `params` still contains `b`, so a comma is required.

After `b`, the parameter object closes.

---

# 27. Closing the JSON object

The final states generate:

```text
}
}
```

One closes:

```json
"parameters": { ... }
```

and the second closes the complete function-call object.

Then:

```python
current_state = States.END
```

terminates generation.

---

# 28. Full example

Consider:

```text
What is the sum of -2 and 3?
```

and this function:

```json
{
  "name": "fn_add_numbers",
  "parameters": {
    "a": {"type": "integer"},
    "b": {"type": "number"}
  }
}
```

The decoder conceptually performs:

```text
Generate fixed "{"
        ↓
Generate fixed ""name":"
        ↓
LLM chooses among valid function names
        ↓
fn_add_numbers
        ↓
Generate fixed ","parameters":{"
        ↓
Generate fixed parameter name "a"
        ↓
Constrained integer generation -> -2
        ↓
Generate comma
        ↓
Generate fixed parameter name "b"
        ↓
Constrained number generation -> 3
        ↓
Close parameter object
        ↓
Close function-call object
```

Final result:

```json
{
  "name": "fn_add_numbers",
  "parameters": {
    "a": -2,
    "b": 3
  }
}
```

---

# 29. The most important idea to remember

The architecture can be summarized in one sentence:

> **The LLM decides what the answer should mean, while the constrained decoder decides what it is allowed to look like.**

The model provides logits:

```text
What token does the model prefer?
```

Your code provides constraints:

```text
Which of those tokens are legal right now?
```

Then logit masking combines both:

```text
LLM logits
    +
grammar/schema constraints
    ↓
highest-scoring legal token
```

That is the central mechanism of `llm.py`.

---

# 30. Defense summary

If you need to explain the file quickly during evaluation, remember these six blocks:

1. **Vocabulary** — maps token IDs to text.
2. **State machine** — knows which part of the JSON is currently being generated.
3. **Legal-token calculation** — determines which vocabulary tokens may continue the output.
4. **Logit masking** — changes illegal-token scores to `-∞`.
5. **Type-specific decoding** — separate constraints for booleans, numbers, integers, and strings.
6. **Small beam search for strings** — prevents open-ended string generation from getting stuck while still using LLM scores.

Everything else mainly supports these six ideas.
