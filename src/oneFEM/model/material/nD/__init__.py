# oneFEM/model/material/nD/__init__.py

# nD materials
from .main import nDMaterial
from .elastic_isotropic import ElasticIsotropic

# delete modules imported from .py directories
del main
del elastic_isotropic

__all__ = [
    "nDMaterial",
    "ElasticIsotropic"
]