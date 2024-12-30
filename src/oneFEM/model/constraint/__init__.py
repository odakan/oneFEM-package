# oneFEM/model/constraint/__init__.py

from .main import Constraint

# MP constraints
from .multipoint.equalDOF import EqualDOF

# SP constraints
from .singlepoint.fix import Fix


__all__ = [
    "Constraint",
    "EqualDOF",
    "Fix"
]