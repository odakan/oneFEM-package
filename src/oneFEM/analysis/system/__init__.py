# oneFEM/analysis/system/__init__.py

from .main import System
from .umfpack import UmfPack
from .fullgeneral import FullGeneral

# delete modules imported from .py directories
del main
del umfpack
del fullgeneral

__all__ = [
    "System",
    "FullGeneral",
    "UmfPack"
]