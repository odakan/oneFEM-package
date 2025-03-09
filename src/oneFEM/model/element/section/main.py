from ...material import Material

class Section(object):
    def __init__(self, secID=-1, mat=Material()):
        self._ID = secID
        self._material = mat
        self._f = 0.0
        self._f_commit = 0.0
        self._C0 = 0.0
        self._C = 0.0
        self._C_commit = 0.0

    def copy(self):
        return Section()

    def _getClassType(self):
        return "section"

    def _setTrialStrain(self, strain):
        self._f = self._C * strain


    # get state
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
        self._C = self.__C0
        self._f = 0.0
        self._f_commit = 0.0
        self._eps = 0.0
        self._eps_commit = 0.0
        return 0