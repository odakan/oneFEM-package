from .main import Section
from ...material import Material
from ...material.uniaxial import uniaxialMaterial
from ...material.nD import nDMaterial
from ...._systools.data import Matrix

class Rectangular(Section):
    def __init__(self, secID, h=0.0, w=0.0, mat=None):
        if mat is None:
            mat = Material()
        super().__init__(secID, mat)
        self.__h0 = h
        self.__w0 = w
        self.__Ix0 = (w*h**3)/12
        self.__Iy0 = (h*w**3)/12
        self.__Ix = 0.0
        self.__Ix_commit = 0.0
        self.__Iy = 0.0
        self.__Iy_commit = 0.0
        self.__eps = 0.0
        self.__eps_commit = 0.0

        # Compute EA from material tangent and cross-section area
        E = self._material._getTangent()
        A = self.__h0 * self.__w0
        self.__EA = E * A
        self.__EA_commit = self.__EA

        self._C = self.__EA
        self._C0 = self.__EA

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
        self.__eps = strain
        self._material._setTrialStrain(strain)
        self._f = self._material._getStress() * self.__h0 * self.__w0
        self._C = self._material._getTangent() * self.__h0 * self.__w0

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
        self.__eps_commit = self.__eps
        self._material._commitState()
        return 0

    def revertToLastCommit(self):
        self._f = self._f_commit
        self._C = self._C_commit
        self.__eps = self.__eps_commit
        self._material._revertToLastCommit()
        return 0

    def revertToStart(self):
        self._C = self._C0
        self._f = 0.0
        self._f_commit = 0.0
        self.__eps = 0.0
        self.__eps_commit = 0.0
        self._material._revertToStart()
        return 0
