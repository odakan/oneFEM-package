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
        self.__f_commit = 0.0
        self.__eps_commit = 0.0
        self.__C_commit = 0.0

    def __repr__(self):
        return f"Elastic(ID={self._ID}, E={self._C})"

    def _getClassType(self):
        return "Elastic uniaxial material"

    def _setTrialStrain(self, strain):
        self._eps = float(strain)
        self._f = self._C * self._eps

    # get state
    def _getStrain(self):
        return self._eps

    def _getStress(self):
        return self._f

    def _getTangent(self):
        return self._C

    def _getInitialTangent(self):
        return self._C0

    # handle state
    def _commitState(self):
        self.__f_commit = self._f
        self.__C_commit = self._C
        self.__eps_commit = self._eps
        return 0

    def _revertToLastCommit(self):
        self._f = self.__f_commit
        self._C = self.__C_commit
        self._eps = self.__eps_commit
        return 0

    def _revertToStart(self):
        self._C = self._C0
        self._f = 0.0
        self.__f_commit = 0.0
        self._eps = 0.0
        self.__eps_commit = 0.0
        return 0
    
    def _getResult(self, query):
        """
        Retrieve results for specific DOFs
        :param query: String specifying the type of result ('stress', 'strain', or 'tangent')
        :return: List of results for the specified DOFs
        """
        if query in {"stress", "stresses", "sigma", "sig"}:
            return self.__f_commit
        elif query in {"strain", "strains", "epsilon", "eps"}:
            return self.__eps_commit
        elif query in {"tangent", "tang", "Ct"}:
            return self.__C_commit
        else:
            raise ValueError("oneFEM.Element.Elastic.getResult() - Unknown result type!")