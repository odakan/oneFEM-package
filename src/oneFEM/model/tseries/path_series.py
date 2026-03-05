import numpy as np
from .main import TSeries

class Path(TSeries):
    def __init__(self, tID, time_values=None, data_values=None,
                 dt=0.0, factor=1.0):
        super().__init__(tID)
        self._factor = factor

        if data_values is None:
            data_values = []

        if time_values is not None and len(time_values) > 0:
            # Explicit time-value pairs
            self._time = np.array(time_values, dtype=float)
            self._data = np.array(data_values, dtype=float)
        elif dt > 0.0 and len(data_values) > 0:
            # Uniform dt mode: infer time from dt
            n = len(data_values)
            self._time = np.arange(n, dtype=float) * dt
            self._data = np.array(data_values, dtype=float)
        else:
            self._time = np.array([], dtype=float)
            self._data = np.array([], dtype=float)

    def getFactor(self, time=0.0):
        if len(self._time) == 0:
            return 0.0
        return self._factor * float(np.interp(time, self._time, self._data))
