# oneFEM/analysis/system/__init__.py

from .linear_soe import System, LinearSOE, LinearSOESolver, SpSolve
from .sparse_assembler import SparseAssembler
from .sparse_general import SparseGeneral
from .umfpack import UmfPackSolver, UmfPackSOE
from .full_general import FullGeneral
from .profilespd import ProfileSPD

# delete modules imported from .py directories
del linear_soe
del sparse_assembler
del sparse_general
del umfpack
del full_general
del profilespd

__all__ = [
    "System",
    "LinearSOE",
    "LinearSOESolver",
    "SpSolve",
    "SparseAssembler",
    "SparseGeneral",
    "UmfPackSolver",
    "UmfPackSOE",
    "FullGeneral",
    "ProfileSPD",
]
