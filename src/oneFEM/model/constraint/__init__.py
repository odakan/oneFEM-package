# oneFEM/model/constraint/__init__.py

from .main import Constraint

# MP constraints
from .multipoint.equalDOF import EqualDOF

# SP constraints
from .singlepoint.fix import Fix


# delete modules imported from .py directories
del main
del singlepoint
del multipoint


__all__ = [
    "Constraint",
    "EqualDOF",
    "Fix"
]