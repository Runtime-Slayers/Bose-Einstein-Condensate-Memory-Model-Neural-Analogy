"""
conftest.py — root-level pytest configuration.

Registers all custom markers and sets MPLBACKEND so matplotlib
never tries to open a display (required on headless CI runners).
"""

import os

import pytest

# Force Agg backend before any matplotlib import occurs in tests
os.environ.setdefault("MPLBACKEND", "Agg")


def pytest_configure(config):
    """Register project-specific markers to satisfy --strict-markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with -m 'not slow')",
    )
    config.addinivalue_line(
        "markers",
        "network: marks tests that require network access",
    )
    config.addinivalue_line(
        "markers",
        "physics: marks tests that verify published physical constants",
    )
