# oneFEM/analysis/constraints/__init__.py

from .main import ConstraintHandler
from .plain import Plain
from .penalty import Penalty

# delete modules imported from .py directories
del main
del plain
del penalty

__all__ = [
    "ConstraintHandler",
    "Penalty",
    "Plain"
]