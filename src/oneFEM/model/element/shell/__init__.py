# oneFEM/model/element/shell__init__.py

# Shell elements
from .shellQ4 import ShellQ4

# delete modules imported from .py directories
del shellQ4


__all__ = [
    "ShellQ4"
]