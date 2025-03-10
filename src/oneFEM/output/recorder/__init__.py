# oneFEM/output/recorder/__init__.py

from .main import Recorder
from .element_recorder import ElementRecorder
from .node_recorder import NodeRecorder

# delete modules imported from .py directories
del main
del element_recorder
del node_recorder

__all__ = [
    "Recorder",
    "ElementRecorder",
    "NodeRecorder"
]