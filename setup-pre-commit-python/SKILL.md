---
name: setup-pre-commit-python
description: Set up pre-commit hooks for Python repos with Ruff (lint + format), mypy type checking, and pytest. Use when user wants to add pre-commit hooks to a Python project, configure pre-commit framework, set up Ruff/mypy/pytest at commit time, or add commit-time formatting/linting/typechecking/testing for Python.
---

# Setup Pre-Commit Hooks (Python)

## What This Sets Up

- **pre-commit** framework managing the git hook
- **Ruff** for lint + format on staged files (replaces black, isort, flake8)
- **mypy** type checking
- **pytest** running on commit

## Steps

### 1. Detect package manager

Check, in order: `uv.lock` (uv), `poetry.lock` (poetry), `Pipfile.lock` (pipenv), `pyproject.toml` with `[tool.poetry]` (poetry), else fall back to `pip` + active venv. Set `RUN` prefix accordingly:

- uv → `uv run`
- poetry → `poetry run`
- pipenv → `pipenv run`
- pip/venv → `` (empty; assume venv is active)

### 2. Install dev dependencies

Install as dev dependencies: `pre-commit ruff mypy pytest`

- uv: `uv add --dev pre-commit ruff mypy pytest`
- poetry: `poetry add --group dev pre-commit ruff mypy pytest`
- pipenv: `pipenv install --dev pre-commit ruff mypy pytest`
- pip: `pip install pre-commit ruff mypy pytest` and add to `requirements-dev.txt` if it exists

### 3. Create `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: local
    hooks:
      - id: mypy
        name: mypy
        entry: <RUN> mypy
        language: system
        types: [python]
        pass_filenames: false

      - id: pytest
        name: pytest
        entry: <RUN> pytest
        language: system
        pass_filenames: false
        stages: [pre-commit]
```

Replace `<RUN>` with the detected prefix (e.g. `uv run mypy`, or just `mypy` for pip/venv).

**Adapt**: If the repo has no tests directory or no `pytest` config, omit the pytest hook and tell the user. If there's no mypy config and no type hints in use, ask before adding mypy.

### 4. Add Ruff config to `pyproject.toml` (if missing)

Only add if no `[tool.ruff]` section exists:

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

If `pyproject.toml` doesn't exist, create a minimal one. If repo uses `setup.cfg` or `ruff.toml` instead, write to that.

### 5. Install the git hook

```bash
<RUN> pre-commit install
```

This writes `.git/hooks/pre-commit`.

### 6. Verify

- [ ] `.pre-commit-config.yaml` exists
- [ ] `.git/hooks/pre-commit` exists and is executable
- [ ] Ruff config present (in `pyproject.toml` or `ruff.toml`)
- [ ] `pre-commit`, `ruff`, `mypy`, `pytest` listed as dev deps
- [ ] Run `<RUN> pre-commit run --all-files` to verify hooks fire

### 7. Commit

Stage all changed/created files and commit with message: `Add pre-commit hooks (ruff + mypy + pytest)`

This will run through the new hooks — a good smoke test.

## Notes

- Pin `ruff-pre-commit` `rev` to a recent stable tag; user can later run `pre-commit autoupdate`
- mypy and pytest run as `local` hooks via `language: system` so they use the project's venv (and see installed deps), unlike isolated hook envs
- `pre-commit` only runs hooks on staged files by default; mypy/pytest above use `pass_filenames: false` so they run project-wide
- If pytest is too slow for every commit, suggest moving it to `pre-push` stage instead (`stages: [pre-push]` and `pre-commit install --hook-type pre-push`)
