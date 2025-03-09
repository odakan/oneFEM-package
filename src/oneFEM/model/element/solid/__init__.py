# oneFEM/model/element/solid/__init__.py

# Solid elements
from .brick import Brick8
from .triangular import Tri3
from .quadrilateral import Quad4

# delete modules imported from .py directories
del brick
del triangular
del quadrilateral


__all__ = [
    "Tri3",
    "Quad4",
    "Brick8"
]