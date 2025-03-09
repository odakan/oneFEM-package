from .main import Section
from ...material import Material
from ...material.uniaxial import uniaxialMaterial
from ...material.nD import nDMaterial
from ...._systools.data import Matrix

class Rectangular(Section):
    def __init__(self, secID, h=0.0, w=0.0, mat=Material()):
        super().__init__(secID, mat)
        self.__h0 = h
        self.__w0 = w
        self.__EIx0 = (w*h**3)/12
        self.__EIy0 = (h*w**3)/12
        self.__EIx = 0.0
        self.__EIx_commit = 0.0
        self.__EIy = 0.0
        self.__EIy_commit = 0.0
        self.__EA = 0.0
        self.__EA_commit = 0.0
        self.__eps = 0.0
        self.__eps_commit = 0.0

        self._C = self.__EA
        self._C0 = self._C

    def copy(self):
        return Rectangular(self._ID, h=self.__h0, w=self.__w0, mat=self._material)

    def _getClassType(self):
        if isinstance(self._material, nDMaterial):
            return "Rectangular nD section"
        if isinstance(self._material, uniaxialMaterial):
            return "Rectangular uniaxial section"
        else:
            return "Faulty section!"

    def _setTrialStrain(self, strain):
        self._f = self._C * strain

    # get state
    def getStrain(self):
        return self.__eps

    def getStress(self):
        return self._f

    def getTangent(self):
        return self._C

    def getInitialTangent(self):
        return self._C0

    # handle state
    def commitState(self):
        self._f_commit = self._f
        self._C_commit = self._C
        return 0

    def revertToLastCommit(self):
        self._f = self._f_commit
        self._C = self._C_commit
        return 0

    def revertToStart(self):
        self._C = self._C0
        self._f = 0.0
        self._f_commit = 0.0
        self.__eps = 0.0
        self.__eps_commit = 0.0
        return 0
        