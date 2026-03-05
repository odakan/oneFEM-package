from .main import Test

class NormDispIncr(Test):
    def __init__(self, tID=-1, tol=1e-6, maxIter=10, printFlag=0):
        super().__init__(tID, tol, maxIter, printFlag)

    def start(self):
        self._currentIter = 0

    def test(self, norm_value):
        self._currentIter += 1
        if self._printFlag > 0:
            print(f"NormDispIncr: iter {self._currentIter}, norm = {norm_value:.6e}")
        if norm_value <= self._tol:
            return 0   # converged
        if self._currentIter >= self._maxIter:
            return -2  # max iterations exceeded
        return -1      # not yet converged
