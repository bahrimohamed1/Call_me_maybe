all: run

install:
	uv sync

run:
	uv run python3 -m src

lint:
	flake8 src

clean:
	rm -rf src/__pycache__
	rm -rf .mypy_cache
