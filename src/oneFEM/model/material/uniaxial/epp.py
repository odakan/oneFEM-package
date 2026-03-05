##-----------------------------------------------------------------------##
#                                                                         #
#        #--oneFEM--#: One FEM software in a galaxy far far away          #
#                                                                         #
#                   Computational Mechanics 2022                          #
#                       University of Pavia                               #
#               Written by: Onur Deniz AKAN, IUSS Pavia                   #
#                         04 March 2026                                   #
#                                                                         #
##-----------------------------------------------------------------------##
#
# ELASTIC-PERFECTLY-PLASTIC (EPP) uniaxial material
#
#   Bilinear stress-strain law with zero post-yield hardening:
#     sigma = E * (eps - eps_p)       if |sigma_trial| <= fy
#     sigma = sign(sigma_trial) * fy  if |sigma_trial| >  fy
#
#   Tangent:
#     E   in elastic range
#     0   when yielded
#

from .main import uniaxialMaterial


class ElasticPerfectlyPlastic(uniaxialMaterial):
    def __init__(self, matID, E=0.0, fy=0.0):
        super().__init__(matID, E)
        self.__fy = abs(fy)
        self.__eps_p = 0.0           # trial plastic strain
        self.__eps_p_commit = 0.0    # committed plastic strain
        self.__f_commit = 0.0
        self.__eps_commit = 0.0
        self.__C_commit = E

    def __repr__(self):
        return f"ElasticPerfectlyPlastic(ID={self._ID}, E={self._C0}, fy={self.__fy})"

    def _getClassType(self):
        return "ElasticPerfectlyPlastic uniaxial material"

    def _setTrialStrain(self, strain):
        self._eps = float(strain)
        E = self._C0
        # Trial stress from elastic predictor (using committed plastic strain)
        sigma_trial = E * (self._eps - self.__eps_p_commit)

        if abs(sigma_trial) <= self.__fy:
            # Elastic
            self._f = sigma_trial
            self._C = E
            self.__eps_p = self.__eps_p_commit
        else:
            # Yielded — return stress to yield surface
            if sigma_trial > 0.0:
                self._f = self.__fy
            else:
                self._f = -self.__fy
            self._C = 0.0
            self.__eps_p = self._eps - self._f / E

    def _getStrain(self):
        return self._eps

    def _getStress(self):
        return self._f

    def _getTangent(self):
        return self._C

    def _getInitialTangent(self):
        return self._C0

    def _commitState(self):
        self.__f_commit = self._f
        self.__eps_commit = self._eps
        self.__C_commit = self._C
        self.__eps_p_commit = self.__eps_p
        return 0

    def _revertToLastCommit(self):
        self._f = self.__f_commit
        self._eps = self.__eps_commit
        self._C = self.__C_commit
        self.__eps_p = self.__eps_p_commit
        return 0

    def _revertToStart(self):
        self._C = self._C0
        self._f = 0.0
        self._eps = 0.0
        self.__f_commit = 0.0
        self.__eps_commit = 0.0
        self.__C_commit = self._C0
        self.__eps_p = 0.0
        self.__eps_p_commit = 0.0
        return 0

    def _getResult(self, query):
        if query in {"stress", "stresses", "sigma", "sig"}:
            return self.__f_commit
        elif query in {"strain", "strains", "epsilon", "eps"}:
            return self.__eps_commit
        elif query in {"tangent", "tang", "Ct"}:
            return self.__C_commit
        elif query in {"plasticStrain", "eps_p"}:
            return self.__eps_p_commit
        else:
            raise ValueError(
                "oneFEM.ElasticPerfectlyPlastic._getResult() - Unknown result type!")
