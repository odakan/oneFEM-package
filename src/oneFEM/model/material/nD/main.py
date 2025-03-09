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
# Date: 01/03/2025
# Version: 0.1
#
#NDMATERIAL main object definition
#   definition of a FE NDMATERIAL object and functions operating 
#   in a section object.

from ..main import Material

class nDMaterial(Material):
    def __init__(self, mat_id=-1):
        self.__ID = mat_id