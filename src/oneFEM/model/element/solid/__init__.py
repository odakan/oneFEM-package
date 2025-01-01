# oneFEM/model/element/solid__init__.py

# Solid elements
from .triangular import Tri3
from .quadrilateral import Quad4

# delete modules imported from .py directories
del triangular
del quadrilateral


__all__ = [
    "Tri3",
    "Quad4"
]