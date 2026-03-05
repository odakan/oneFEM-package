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
from ...._systools.data import Vector

class uniaxialMaterial(Material):
    def __init__(self, mat_id=-1, E0=1.0):
        self._ID = mat_id
        self._f = 0.0
        self._eps = 0.0
        self._C = E0
        self._C0 = E0

    def __repr__(self):
        return f"uniaxialMaterial(ID={self._ID}, E={self._C})"

    def _getClassType(self):
        return "uniaxial material"

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
        return 0

    def _revertToLastCommit(self):
        return 0

    def _revertToStart(self):
        return 0
    
    def _getResult(self, query):
        """
        Retrieve results for specific DOFs
        :param query: String specifying the type of result ('stress', 'strain', or 'tangent')
        :return: List of results for the specified DOFs
        """
        if query in {"stress", "stresses", "sigma", "sig"}:
            return self._f
        elif query in {"strain", "strains", "epsilon", "eps"}:
            return self._eps
        elif query in {"tangent", "tang", "Ct"}:
            return self._C
        else:
            raise ValueError("oneFEM.Element.Elastic.getResult() - Unknown result type!")