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
#ELASTIC main object definition
#   definition of a FE ELASTIC object and functions operating 
#   in a section object.

from .main import uniaxialMaterial

class Elastic(uniaxialMaterial):
    def __init__(self, matID, E=0.0):
        super().__init__(matID, E)
        self.__eps = 0.0
        self.__eps_commit = 0.0

    def _getClassType(self):
        return "Elastic uniaxial material"

    def _setTrialStrain(self, strain):
        self.__eps = strain
        self.__f = self.__C * self.__eps

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
        self.__eps_commit = self.__eps
        return 0

    def revertToLastCommit(self):
        self.__f = self.__f_commit
        self.__C = self.__C_commit
        self.__eps=self.__eps_commit
        return 0

    def revertToStart(self):
        self.__C = self.__C0
        self.__f = 0.0
        self.__f_commit = 0.0
        self.__eps = 0.0
        self.__eps_commit = 0.0
        return 0