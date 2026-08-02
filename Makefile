.PHONY: install test lint typecheck gate demo clean

install:
	python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check .

typecheck:
	.venv/bin/mypy --ignore-missing-imports ir.py profiler.py runtime.py passes/ codegen/

gate: test lint typecheck

demo:
	.venv/bin/python examples/decode_demo.py

clean:
	find . -path './.venv' -prune -o -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.ruff_cache' -o -name '.mypy_cache' -o -name '*.egg-info' \) -prune -exec rm -rf {} +
