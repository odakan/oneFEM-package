# oneFEM/analysis/system/__init__.py

from .main import System
from .umfpack import UmfPack

# delete modules imported from .py directories
del main
del umfpack

__all__ = [
    "System",
    "UmfPack"
]