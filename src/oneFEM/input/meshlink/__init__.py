# oneFEM/input/meshlink/__init__.py

from .main import MeshLink
from .cubit_mesh import CubitMesh
from .gid_mesh import GidMesh

# delete modules imported from .py directories
del main
del cubit_mesh
del gid_mesh

__all__ = [
    "MeshLink",
    "CubitMesh",
    "GidMesh"
]