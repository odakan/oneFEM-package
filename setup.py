# Backward-compatibility shim — all config lives in pyproject.toml.
# Kept so that `pip install -e .` works on older pip versions (< 21.3).
from setuptools import setup
setup()
