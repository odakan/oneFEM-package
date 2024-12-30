# oneFEM/model/material/__init__.py

from .main import Material

# nD materials
from .nD.elastic_isotropic import ElasticIsotropic

# uniaxial (1D) materials
from .uniaxial.elastic import Elastic1D


__all__ = [
    "Material",
    "ElasticIsotropic",
    "Elastic1D"
]