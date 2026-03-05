from ..pattern.main import Pattern
from ..tseries.main import TSeries

class UniformExcitation(Pattern):
    def __init__(self, pID, direction, tseries=None):
        """
        Uniform ground acceleration excitation (OpenSees-style).

        Applies F_eq = -M * r * a_g(t) where r is the influence vector
        (1.0 at the specified DOF direction for each node).

        :param pID: Pattern ID
        :param direction: DOF index for excitation direction (1-based, e.g. 1=x, 2=y, 3=z)
        :param tseries: TimeSeries providing ground acceleration a_g(t)
        """
        super().__init__()
        self._ID = pID
        self._direction = int(direction)  # 1-based DOF index
        self._tseries = tseries

    def getNodalLoads(self, time=0.0):
        """Not used for UniformExcitation — loads depend on mass matrix."""
        return {}

    def getAcceleration(self, time=0.0):
        """Return ground acceleration at given time."""
        if self._tseries is None:
            return 0.0
        return self._tseries.getFactor(time)

    @property
    def direction(self):
        """Return 1-based DOF direction index."""
        return self._direction
