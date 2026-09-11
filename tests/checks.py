"""Local verification only; tests are excluded from submission by .gitignore."""
import ast
import json
import math
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from pydantic import ValidationError

from src.__main__ import write_results
from src.decoder import Decoder, vocabulary_bytes
from src.grammar import ScalarGrammar
from src.schema import (
    FunctionDefinition, Result, load_inputs, read_json, validate_result,
)

ROOT = Path(__file__).resolve().parents[1]


def must_fail(action):
    try:
        action()
    except (ValueError, ValidationError, OSError):
        return
    raise AssertionError('Expected validation failure')


def test_scalar_grammar():
    valid = {
        'string': ['""', '"hello"', '"\\\"\\\\\\n\\t"',
                   '"café 😀 東京"', '"\\u0041"'],
        'number': ['0', '-0', '12345678901234567890', '-12.5',
                   '1E+12', '1e-9', '0.0001'],
        'integer': ['0', '-0', '-100', '12345678901234567890'],
        'boolean': ['true', 'false'], 'null': ['null'],
    }
    invalid = {
        'string': ['"line\nfeed"', '"\\x"', '"\\uZZZZ"', '12', 'true'],
        'number': ['01', '+1', '.1', '1.', '-', '1e', '1e+', 'NaN',
                   'Infinity', 'true', '"12"'],
        'integer': ['1.2', '1e2', '01', 'true'],
        'boolean': ['True', '0', '"false"'], 'null': ['None', '"null"'],
    }
    for kind, values in valid.items():
        for delimiter in (',', '}'):
            grammar = ScalarGrammar(kind=kind, delimiter=delimiter)
            for value in values:
                data = (value + delimiter).encode()
                assert grammar.advance('start', data) == 'done', data
                for split in range(len(data) + 1):
                    state = grammar.advance('start', data[:split])
                    assert state is not None
                    assert grammar.advance(state, data[split:]) == 'done'
                assert grammar.advance('start', data + b'garbage') is None
                assert grammar.advance('start', data + b',') is None
        for value in invalid[kind]:
            assert ScalarGrammar(kind=kind, delimiter='}').advance(
                'start', (value + '}').encode()) is None, (kind, value)
    grammar = ScalarGrammar(kind='string', delimiter='}')
    for raw in (b'"\x80"}', b'"\xc0\xaf"}', b'"\xed\xa0\x80"}',
                b'"\xf4\x90\x80\x80"}', b'"\xff"}'):
        assert grammar.advance('start', raw) is None
    random.seed(42)
    for _ in range(500):
        value = ''.join(chr(random.choice([0, 9, 10, 34, 92, 97, 233,
                                          0x6771, 0x1f600]))
                        for _ in range(random.randrange(30)))
        for escaped in (True, False):
            raw = (json.dumps(value, ensure_ascii=escaped) + '}').encode()
            assert grammar.advance('start', raw) == 'done'


def test_schema_and_files():
    functions, requests = load_inputs(
        ROOT / 'data/input/functions_definition.json',
        ROOT / 'data/input/function_calling_tests.json')
    assert len(functions) == 5 and len(requests) == 11
    definition = FunctionDefinition.model_validate({
        'name': 'arbitrary', 'description': 'Example',
        'parameters': {'x': {'type': 'number'}}, 'returns': {'type': 'null'}})
    validate_result(Result(prompt='', name='arbitrary', parameters={'x': 3}),
                    definition)
    for value in (True, '3', None, math.inf, math.nan):
        must_fail(lambda: validate_result(Result(
            prompt='', name='arbitrary', parameters={'x': value}), definition))
    for args in ({}, {'x': 2, 'extra': 3}):
        must_fail(lambda: validate_result(Result(
            prompt='', name='arbitrary', parameters=args), definition))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'bad.json'
        for content in ('{', '[1,]', 'NaN', '{"a": 1, "a": 2}'):
            path.write_text(content)
            must_fail(lambda: read_json(path))
        must_fail(lambda: read_json(Path(directory) / 'missing.json'))
        output = Path(directory) / 'nested/result.json'
        write_results(output, [])
        assert json.loads(output.read_text()) == []
        assert not list(output.parent.glob('*.tmp'))
        invalid = Result(prompt='', name='arbitrary', parameters={'x': math.inf})
        must_fail(lambda: write_results(output, [invalid]))
        assert output.read_text() == '[]\n'


def test_logit_mask():
    fake = SimpleNamespace(get_logits_from_input_ids=lambda ids: [100, 1, 9])
    decoder = Decoder(sdk=fake, vocabulary={0: b'x', 1: b'0', 2: b'1'})
    context = [99]
    assert decoder.select(context, [1, 2], 'test') == 2
    assert context == [99, 2]
    must_fail(lambda: decoder.select([], [], 'empty mask'))
    fake.get_logits_from_input_ids = lambda ids: [math.nan, math.inf, -math.inf]
    must_fail(lambda: decoder.select([], [0, 1, 2], 'nonfinite'))


def test_cli_errors():
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        output = base / 'result.json'
        output.write_text('preserve me')
        cases = [('missing.json', None), ('bad.json', '{'),
                 ('wrong.json', '[{"prompt": 3}]')]
        for name, content in cases:
            path = base / name
            if content is not None:
                path.write_text(content)
            run = subprocess.run([sys.executable, '-m', 'src', '--input',
                                  str(path), '--output', str(output)],
                                 cwd=ROOT, capture_output=True, text=True)
            assert run.returncode == 1, run
            assert 'Error:' in run.stderr and 'Traceback' not in run.stderr
            assert 'Loading' not in run.stderr
            assert output.read_text() == 'preserve me'
        empty = base / 'empty.json'
        empty.write_text('[]')
        run = subprocess.run([sys.executable, '-m', 'src', '--input', str(empty),
                              '--output', str(output)], cwd=ROOT,
                             capture_output=True, text=True)
        assert run.returncode == 0 and json.loads(output.read_text()) == []
        run = subprocess.run([sys.executable, '-m', 'src', '--input', str(empty),
                              '--output', str(empty)], cwd=ROOT,
                             capture_output=True, text=True)
        assert run.returncode == 1 and empty.read_text() == '[]'


def test_source_boundaries():
    forbidden = {'torch', 'transformers', 'huggingface_hub', 'dspy', 'outlines'}
    sdk_methods = {'encode', 'decode', 'get_path_to_vocab_file',
                   'get_logits_from_input_ids'}
    for path in (ROOT / 'src').glob('*.py'):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(alias.name.split('.')[0] in forbidden
                               for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                assert (node.module or '').split('.')[0] not in forbidden
            if isinstance(node, ast.ClassDef):
                assert any(isinstance(b, ast.Name) and b.id == 'BaseModel'
                           for b in node.bases)
            if isinstance(node, ast.Attribute):
                owner = node.value
                if ((isinstance(owner, ast.Name) and owner.id == 'sdk') or
                    (isinstance(owner, ast.Attribute) and owner.attr == 'sdk')):
                    assert node.attr in sdk_methods, node.attr


if __name__ == '__main__':
    checks = [value for key, value in list(globals().items())
              if key.startswith('test_')]
    for check in checks:
        check()
        print(f'PASS {check.__name__}')
    print(f'{len(checks)} local verification groups passed')
