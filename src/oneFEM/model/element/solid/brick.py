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
# 
# Author: Onur Deniz Akan
# Date: 08/02/2025
# Version: 0.1
#
#BRICK main object definition
#   definition of a FE node object and functions operating over a node
#   3D 3-dof mechanical solid node

from ..main import Element

class Brick(Element):
    def __init__(self, id=-1):
        super().__init__(id)


class Brick8(Brick):
    def __init__(self, id=-1):
        super().__init__(id)
        self.nodes = [None]*8