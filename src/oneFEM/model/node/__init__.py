# oneFEM/model/node/__init__.py

from .main import Node
from .node_2_2 import Node22
from .node_2_3 import Node23
from .node_2_4 import Node24
from .node_3_3 import Node33
from .node_3_4 import Node34
from .node_3_6 import Node36
from .node_3_7 import Node37

# delete modules imported from .py directories
del main
del node_2_2
del node_2_3
del node_2_4
del node_3_3
del node_3_4
del node_3_6
del node_3_7

__all__ = [
    "Node",
    "Node22",
    "Node23",
    "Node24",
    "Node33",
    "Node34",
    "Node36",
    "Node37"
]