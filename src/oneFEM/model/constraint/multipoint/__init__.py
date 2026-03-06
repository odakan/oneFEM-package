# oneFEM/model/constraint/multipoint/__init__.py

from .equalDOF import EqualDOF

# delete modules imported from .py directories
del equalDOF


__all__ = [
    "EqualDOF",
]
