# oneFEM/model/element/continuum/__init__.py

from .base import ContinuumElement
from .quad4 import Quad4
from .hex8 import Hex8

# delete modules imported from .py directories
del base
del quad4
del hex8

__all__ = [
    "ContinuumElement",
    "Quad4",
    "Hex8",
]
