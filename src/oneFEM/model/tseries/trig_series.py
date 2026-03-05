import numpy as np
from .main import TSeries

class Trig(TSeries):
    def __init__(self, tID, tStart=0.0, tEnd=1.0, period=1.0,
                 factor=1.0, shift=0.0):
        super().__init__(tID)
        self._tStart = tStart
        self._tEnd = tEnd
        self._period = period
        self._factor = factor
        self._shift = shift

    def getFactor(self, time=0.0):
        if time < self._tStart or time > self._tEnd:
            return 0.0
        return self._factor * np.sin(
            2.0 * np.pi * (time - self._tStart) / self._period + self._shift
        )
