# Contributing to BEC Memory Model

Thank you for your interest in contributing! This document describes how to set up your development environment, run tests, and submit changes.

---

## Setting Up

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/Bose-Einstein-Condensate-Memory-Model-Neural-Analogy.git
cd Bose-Einstein-Condensate-Memory-Model-Neural-Analogy

# 2. Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=bec_memory --cov-report=term-missing

# Single test file
pytest tests/test_model.py -v
```

All tests must pass before a PR is merged.

---

## Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting:

```bash
ruff check bec_memory/ experiments/ tests/
ruff format bec_memory/ experiments/ tests/
```

Rules: PEP 8, import sorting (isort-compatible), and pyupgrade.

---

## Docstring Format

Use [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html):

```python
def condensate_fraction(self, T: float) -> float:
    """Fraction of neurons in the coherent memory trace.

    Uses the harmonic-trap ideal-Bose-gas result (Bagnato 1987):

        n₀/N = max(0, 1 − (T/T_c)^3)

    Parameters
    ----------
    T : float
        Effective noise temperature (same units as T_c).

    Returns
    -------
    float
        Fraction ∈ [0, 1].
    """
```

---

## Pull Request Guidelines

1. **Branch name:** `feature/your-feature` or `fix/your-bugfix`
2. **Commit messages:** Use [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `test:`, `refactor:`
3. **Tests:** Add tests for any new functionality
4. **Documentation:** Update `docs/THEORY.md` for theoretical additions
5. **Changelog:** Add an entry to `CHANGELOG.md` under `[Unreleased]`

---

## Project Structure

```
bec_memory/           Core Python package — all scientific logic
experiments/          Standalone analysis scripts (use bec_memory package)
tests/                pytest unit tests
docs/                 Theoretical documentation
figures/              Output figures and JSON results
```

---

## Reporting Issues

Please use the GitHub issue templates:
- **Bug report** — for incorrect physics, broken code, or test failures
- **Feature request** — for new physical models, analyses, or visualisations

---

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/) Code of Conduct. Please be respectful and constructive in all interactions.
