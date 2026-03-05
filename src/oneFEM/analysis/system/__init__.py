# oneFEM/analysis/system/__init__.py

from .main import System
from .umfpack import UmfPack
from .fullgeneral import FullGeneral
from .profilespd import ProfileSPD

# delete modules imported from .py directories
del main
del umfpack
del fullgeneral
del profilespd

__all__ = [
    "System",
    "FullGeneral",
    "ProfileSPD",
    "UmfPack"
]
