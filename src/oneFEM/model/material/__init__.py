# oneFEM/model/material/__init__.py

from .main import Material
from . import nD
from . import uniaxial

# delete modules imported from .py directories
del main

__all__ = [
    "Material",
    "nD",
    "uniaxial"
]