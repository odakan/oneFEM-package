from .main import TSeries

class Constant(TSeries):
    def __init__(self, tID, factor=1.0):
        super().__init__(tID)
        self._factor = factor

    def getFactor(self, time=0.0):
        return self._factor
