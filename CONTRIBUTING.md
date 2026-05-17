# Contributing to bagel-sbsr

## Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

## Clone and install

```bash
git clone https://github.com/hinanohart/bagel-sbsr.git
cd bagel-sbsr
pip install -e ".[dev]"
```

## Run tests

```bash
pytest
```

## Lint

```bash
ruff check .
ruff format --check .
```

## Branch convention

Use `feat/<topic>`, `fix/<topic>`, or `chore/<topic>` branches off `main`.

## Pull request convention

- One logical change per PR.
- Include tests for new behaviour.
- Ensure `pytest` and lint pass locally before opening the PR.
