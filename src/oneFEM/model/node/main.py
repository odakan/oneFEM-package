##-----------------------------------------------------------------------##
#                                                                         #
#        #--oneFEM--#: One FEM software in a galaxy far far away          #
#                                                                         #
#                   Computational Mechanics 2022                          #
#                       University of Pavia                               #
#               Written by: Onur Deniz AKAN, IUSS Pavia                   #
#                         15 January 2022                                 #
#                                                                         #
##-----------------------------------------------------------------------##
#NODE main object definition
#   definition of a FE node object and functions operating over a node
#   default 3D 3-dof mechanical solid node

from numpy import array

class Node(object):
    def __init__(self, node_id):
        """
        Node Constructor
        :param node_id: Node ID (integer)
        """
        if not isinstance(node_id, int):
            raise ValueError("Node ID must be an integer!")
        
        self.ID = int(node_id)

    # Node API (for internal use)
    def _setDOF(self, *args, **kwargs):
        raise RuntimeError("oneFEM.Node._setDOF() - This function is under the responsibility of a child node object.")

    def _update(self, *args, **kwargs):
        raise RuntimeError("oneFEM.Node._update() - This function is under the responsibility of a child node object.")
    
    def _commit(self, *args, **kwargs):
        raise RuntimeError("oneFEM.Node._commit() - This function is under the responsibility of a child node object.")
    
    # Node API (public)   
    def getResult(self, *args, **kwargs):
        raise RuntimeError("oneFEM.Node.getResult() - This function is under the responsibility of a child node object.")