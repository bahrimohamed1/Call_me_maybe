all: run

install:
	uv sync

run:
	@uv run python3 -m src

debug:
	uv run python3 -m pdb -m src

lint:
	flake8 .
	mypy . --warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

clean:
	rm -rf src/__pycache__
	rm -rf .mypy_cache