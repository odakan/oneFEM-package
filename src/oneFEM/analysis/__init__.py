# oneFEM/analysis/__init__.py

from .main import Analysis
from . import algorithm
from . import constraints
from . import eigen
from . import integrator
from . import numberer
from . import system
from . import test

# delete modules imported from .py directories
del main

__all__ = [
    "Analysis",
    "algorithm",
    "constraints",
    "eigen",
    "integrator",
    "numberer",
    "system",
    "test"
]