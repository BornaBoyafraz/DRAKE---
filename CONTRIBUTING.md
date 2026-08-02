# Contributing to DRAKE

## Layout and imports

DRAKE uses a flat import layout. Core modules (`ir.py`, `runtime.py`, and
`profiler.py`) live at the repository root, with passes in `passes/`, code
generation in `codegen/`, tests in `tests/`, examples in `examples/`, and
documentation in `docs/`.

Import directly from those modules, for example:

```python
from ir import Graph, Op
from passes.fusion import FusionPass
from runtime import DrakeEngine
```

Do not use `from drake...` imports.

## Development setup

Create the project virtual environment and install the development extras:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Mandatory quality gate

Run all three commands from the repository root before every commit:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/mypy --ignore-missing-imports ir.py profiler.py runtime.py passes/ codegen/
```

All three commands must pass; never commit a red gate.

## Commits

Keep each commit focused and use a clear subject and explanatory body. End
every commit message with this exact trailer:

```text
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

## Semantics-preserving compiler passes

Fusion, dead-code elimination, and common subexpression elimination must stay
semantics-preserving. Every change to fusion, DCE, or CSE must include a test
that runs equivalent inputs before and after the pass and proves the observable
graph outputs are unchanged.
