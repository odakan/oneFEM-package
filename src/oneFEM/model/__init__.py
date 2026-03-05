# oneFEM/model/__init__.py

from .main import Domain
from . import constraint
from . import element
from . import material
from . import node
from . import pattern
from . import tseries

# delete modules imported from .py directories
del main

__all__ = [
    "Domain",
    "constraint",
    "element",
    "material",
    "node",
    "pattern",
    "tseries"
]