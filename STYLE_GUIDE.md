# Python Style Guide for Resource Simulator

## Overview
This guide documents the coding standards for the Resource Simulator project to prevent indentation issues, style inconsistencies, and common pitfalls.

## Core Rules

### 1. Indentation
- **Use 4 spaces per indent level** (no tabs)
- Never mix tabs and spaces
- Continuation lines should be indented by 4 spaces from the opening delimiter

### 2. Blank Lines
- **No trailing whitespace** on blank lines (W293)
- **2 blank lines** between top-level function/class definitions (E302)
- **1 blank line** between method definitions within a class
- No blank lines with trailing spaces (W293)

### 3. Line Length
- **Maximum 120 characters** per line (E501)
- Break long lines using parentheses for implicit line continuation
- Prefer implicit continuation over explicit backslashes

### 4. Imports
- Standard library imports first
- Third-party imports second
- Local/application imports third
- Group imports with blank lines between groups
- Use absolute imports

### 5. Type Hints
- Use type hints for all function signatures
- Use `from __future__ import annotations` for forward references
- Use `TYPE_CHECKING` for imports only needed for type hints

### 6. Docstrings
- Use Google-style docstrings
- Include args, returns, raises sections where applicable
- Keep under 120 characters per line

### 7. Naming Conventions
- Classes: `PascalCase`
- Functions/methods/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private: `_leading_underscore`
- Type variables: `TypeVar` with descriptive names

## Pre-Commit Hooks

Install and configure pre-commit hooks to catch issues automatically:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        args: [--line-length=120]
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black-jupyter
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: [--profile=black, --line-length=120]
  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=120, --extend-ignore=E203,W503]
```

Run `pre-commit install` to enable.

## Editor Configuration

### VS Code (`.vscode/settings.json`)
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "ms-python.black-formatter",
  "editor.codeActionsOnSave": {
    "source.organizeImports": "explicit"
  },
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.pylintEnabled": false,
  "editor.insertSpaces": true,
  "editor.tabSize": 4,
  "editor.detectIndentation": false,
  "files.trimTrailingWhitespace": true,
  "files.insertFinalNewline": true,
  "files.trimFinalNewlines": true
}
```

### PyCharm
- Settings → Editor → Code Style → Python
  - Tab size: 4
  - Indent: 4
  - Continuation indent: 4
  - Keep line breaks: checked
  - Hard wrap at: 120
  - Visual guides: 120

## CI Configuration

```yaml
# .github/workflows/lint.yml
name: Lint
on: [push, pull_request]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install pre-commit
      - run: pre-commit run --all-files
```

## Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| W293 | Blank line has trailing spaces | Configure editor to trim trailing whitespace on save |
| E302 | Missing blank line before class/function | Add 2 blank lines before class/function |
| E501 | Line too long (>120 chars) | Break line using parentheses |
| W292 | No newline at end of file | Ensure file ends with newline |
| E303 | Too many blank lines | Remove extra blank lines |
| E128 | Continuation line under-indented | Indent 4 spaces from opening delimiter |

## Pre-commit Workflow

```bash
# One-time setup
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

## Code Review Checklist

- [ ] No trailing whitespace
- [ ] No lines > 120 chars
- [ ] Proper blank lines between classes/functions
- [ ] Type hints on all public functions
- [ ] Docstrings for public APIs
- [ ] No trailing commas in single-line structures
- [ ] Imports sorted and grouped
- [ ] No unused imports/variables

## Quick Fix Commands

```bash
# Auto-format with black
black --line-length=120 src/

# Sort imports
isort --profile=black --line-length=120 src/

# Check style
flake8 --max-line-length=120 src/

# Run all checks
pre-commit run --all-files
```

## File-Specific Notes

### simulation.py
- Watch for blank line whitespace (W293) in `__init__` and method bodies
- Ensure 2 blank lines before class/method definitions
- Keep methods under 50 lines when possible

### render.py
- Maximum line length 120 chars
- Ensure final newline at EOF
- Keep rendering methods under 60 lines

### simulation.py
- Pre-allocate numpy arrays in `__init__`
- Use `_timer` context manager for profiling
- Delegate to system classes (MovementSystem, ForagingSystem, etc.)

### movement.py / foraging.py / predator_behavior.py
- Keep methods focused and under 50 lines
- Use type hints for all parameters
- Access sim attributes via `self.sim` consistently

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                    QUICK STYLE REFERENCE                    │
├─────────────────────────────────────────────────────────────┤
│ Indentation          │ 4 spaces (no tabs)                   │
│ Blank lines          │ No trailing whitespace (W293)        │
│ Class separation     │ 2 blank lines (E302)                 │
│ Method separation    │ 1 blank line                         │
│ Line length          │ ≤ 120 chars (E501)                   │
│ End of file          │ Exactly 1 newline (W292)             │
│ Imports              │ stdlib → third-party → local         │
│ Type hints           │ All public functions/methods         │
| Docstrings           | Google style, <120 chars/line        │
└─────────────────────────────────────────────────────────────┘
```

## Automated Enforcement

Run these before committing:
```bash
# One-liner to fix most issues
black --line-length=120 src/ && isort --profile=black --line-length=120 src/ && flake8 --max-line-length=120 src/
```

## IDE Integration Tips

1. **Enable format on save** with Black formatter
2. **Enable organize imports on save**
3. **Show whitespace characters** to catch trailing spaces
4. **Set ruler at 120** for visual line length guide
5. **Highlight trailing whitespace** in editor theme

---

*Last updated: 2026-10-11*
*Project: Resource Simulator*
*Python version: 3.11+*