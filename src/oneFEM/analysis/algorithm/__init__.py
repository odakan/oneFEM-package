# oneFEM/analysis/algorithm/__init__.py

from .main import Algorithm
from .linear import Linear
from .krylov_newton import Krylov
from .newton_raphson import Newton

# delete modules imported from .py directories
del main
del linear
del krylov_newton
del newton_raphson

__all__ = [
    "Algorithm",
    "Linear",
    "Krylov",
    "Newton"
]