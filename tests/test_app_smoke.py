"""Smoke test: the dashboard module must at least import cleanly.

Streamlit apps can't be meaningfully unit-tested without a running server,
but an import-time crash (bad syntax, missing dependency, wrong function
signature) is exactly the kind of thing worth catching in CI.
"""

import importlib


def test_app_module_imports():
    importlib.import_module("app")
