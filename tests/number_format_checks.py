"""Regression checks for schema-driven float formatting, without model weights."""
import json
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from src.__main__ import write_results
from src.decoder import Decoder
from src.schema import FunctionDefinition, Result, normalize_numbers


def generate_from_tokens(parameters, values, name='arbitrary_function'):
    """Feed controlled model tokens through the real generation pipeline."""
    definition = FunctionDefinition.model_validate({
        'name': name, 'description': 'Fixture', 'parameters': parameters,
        'returns': {'type': 'number'},
    })
    stream = json.dumps(name)
    for index, value in enumerate(values):
        stream += json.dumps(value) + (',' if index + 1 < len(values) else '}')
    sequence = iter(stream.encode())

    def logits(context):
        """Make the next planned character the highest-scoring token."""
        scores = [-100.0] * 128
        scores[next(sequence)] = 10.0
        return scores

    sdk = SimpleNamespace(
        encode=lambda text: SimpleNamespace(tolist=lambda: [[ord(c) for c in text]]),
        get_logits_from_input_ids=logits,
    )
    decoder = Decoder(sdk=sdk, vocabulary={i: bytes([i]) for i in range(128)})
    return decoder.generate('A request', [definition])


def main():
    """Verify wire types and compatibility with the actual public fixtures."""
    sys.path.insert(0, str(Path('moulinette').resolve()))
    from moulinette.functions_definition import fn_add_numbers, fn_get_square_root

    results = []
    for value in (16, 144, -12, 0, 12.5, 1000000000000):
        result = generate_from_tokens({'a': {'type': 'number'}}, [value])
        assert type(result.parameters['a']) is float, result.parameters
        assert result.parameters['a'] == value
        if value in (16, 144):
            assert fn_get_square_root(**result.parameters) == math.sqrt(value)
        results.append(result)

    addition = generate_from_tokens(
        {'a': {'type': 'number'}, 'b': {'type': 'number'}}, [2, 3])
    assert fn_add_numbers(**addition.parameters) == 5.0
    results.append(addition)

    mixed = generate_from_tokens({
        'amount': {'type': 'number'}, 'count': {'type': 'integer'},
        'enabled': {'type': 'boolean'}, 'text': {'type': 'string'},
        'nothing': {'type': 'null'},
    }, [16, 7, True, '16', None])
    assert [type(v) for v in mixed.parameters.values()] == [
        float, int, bool, str, type(None)]
    results.append(mixed)

    definition = FunctionDefinition.model_validate({
        'name': 'numeric', 'description': 'Fixture',
        'parameters': {'a': {'type': 'number'}}, 'returns': {'type': 'number'},
    })
    for invalid in (True, '16', None, float('inf'), float('nan'),
                    10 ** 400, 2 ** 53 + 1):
        try:
            normalize_numbers(Result(prompt='', name='numeric',
                                     parameters={'a': invalid}), definition)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Unsafe conversion accepted: {invalid!r}')
    original = Result(prompt='', name='numeric', parameters={'a': 16})
    normalized = normalize_numbers(original, definition)
    assert type(original.parameters['a']) is int
    assert type(normalized.parameters['a']) is float

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'results.json'
        write_results(path, results)
        wire = json.loads(path.read_text())
        assert type(wire[0]['parameters']['a']) is float
        assert wire[0]['parameters']['a'] == 16.0
        assert type(wire[-1]['parameters']['count']) is int
        assert type(wire[-1]['parameters']['enabled']) is bool
    print('PASS model integer tokens produce float number arguments on disk')
    print('PASS both reported square-root cases and addition fixtures')
    print('PASS integer, boolean, string, and null arguments retain their types')
    print('PASS wrong types, overflow, and precision loss are rejected')


if __name__ == '__main__':
    main()
