# oneFEM/model/material/uniaxial/__init__.py

# uniaxial (1D) materials
from .main import uniaxialMaterial
from .elastic import Elastic

# delete modules imported from .py directories
del main
del elastic


__all__ = [
    "uniaxialMaterial",
    "Elastic"
]