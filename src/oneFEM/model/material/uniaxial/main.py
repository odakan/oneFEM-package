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
#uniaxialMaterial main object definition
#   definition of a FE uniaxialMaterial object and functions operating 
#   in a section object.

from ..main import Material

class uniaxialMaterial(Material):
    def __init__(self, mat_id=-1, E0=0):
        self.__ID = mat_id
        self.__C = E0
        self.__C0 = E0
        self.__f = 0.0
        self.__f_commit = 0.0

    def _getClassType(self):
        return "uniaxial material"

    def _setTrialStrain(self, strain):
        self.__f = self.__C * strain

    # get state
    def getStrain(self):
        return self.__eps

    def getStress(self):
        return self.__f

    def getTangent(self):
        return self.__C

    def getInitialTangent(self):
        return self.__C0

    # handle state
    def commitState(self):
        self.__f_commit = self.__f
        self.__C_commit = self.__C
        return 0

    def revertToLastCommit(self):
        self.__f = self.__f_commit
        self.__C=self.__C_commit
        return 0

    def revertToStart(self):
        self.__C = self.__C0
        self.__f = 0.0
        self.__f_commit = 0.0
        self.__eps = 0.0
        self.__eps_commit = 0.0
        return 0