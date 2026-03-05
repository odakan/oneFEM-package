class Test(object):
    def __init__(self, ID=-1, tol=1e-6, maxIter=10, printFlag=0):
        self._ID = ID
        self._tol = tol
        self._maxIter = maxIter
        self._printFlag = printFlag
        self._currentIter = 0

    @property
    def maxIter(self):
        return self._maxIter

    def start(self):
        self._currentIter = 0

    def test(self, norm_value):
        raise NotImplementedError("Test.test(): method must be implemented by subclasses.")
