# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2025-06-25

### Added

**Python package `bec_memory/`**
- `BECMemoryModel` class with condensate fraction, critical temperature, forgetting curve, and retrieval probability methods
- `GPESolver` — full Gross-Pitaevskii equation solver using split-step Fourier methods and imaginary-time relaxation
- `ForgettingCurve` — BEC, Ebbinghaus, power-law, and exponential forgetting models with least-squares fitting
- `BogoliubovSpectrum` — quasi-particle dispersion, sound speed, healing length, and false-memory probability
- `utils.py` — dark-mode plotting helpers and 4-panel summary figure factory
- Comprehensive NumPy-style docstrings and type annotations throughout

**Experiments**
- `experiments/p52_real_data.py` — fully refactored validation script
  - Uses `argparse`, `logging`, and proper output path management
  - Imports from `bec_memory` package (no code duplication)
  - Full provenance metadata in JSON output
  - `--no-plots`, `--verbose`, `--output-dir` CLI flags

**Tests**
- `tests/test_model.py` — 20 unit tests covering BECMemoryModel
- `tests/test_forgetting.py` — 18 unit tests covering ForgettingCurve
- `tests/test_gpe.py` — 12 unit tests covering GPESolver

**Documentation**
- `README.md` — complete professional rewrite with badges, equations, quick start, results
- `docs/THEORY.md` — full theoretical framework with LaTeX equations and experimental validation
- `CONTRIBUTING.md` — development guide
- `CHANGELOG.md` — this file

**Packaging and CI**
- `pyproject.toml` — PEP 517/621 compliant package configuration
- `requirements.txt`
- `LICENSE` — MIT
- `.github/workflows/ci.yml` — GitHub Actions CI (pytest on Python 3.10–3.11)
- `.github/ISSUE_TEMPLATE/bug_report.md`
- `.github/ISSUE_TEMPLATE/feature_request.md`

### Changed
- Moved `figures_p52/` → `figures/` for cleaner project layout
- Original `test_p52_real_data.py` preserved as `test_p52_real_data.py` (legacy) and superseded by `experiments/p52_real_data.py`
- `BT38_BEC_Memory_Model.md` raw brainstorming notes preserved; superseded by `docs/THEORY.md`

---

## [0.0.1] — 2025 (initial commit)

### Added
- Initial `README.md` (stub)
- `BT38_BEC_Memory_Model.md` (brainstorming document)
- `test_p52_real_data.py` (flat analysis script)
- `figures_p52/p52_bec_memory_figure.png`
- `figures_p52/p52_bec_memory_results.json`
