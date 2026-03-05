from ..pattern.main import Pattern
from ..tseries.main import TSeries

class Plain(Pattern):
    def __init__(self, pID, tseries=TSeries(), load=[[]]):
        super().__init__()
        self._ID = pID
        self._tseries = tseries
        self._load = load  # list of [nodeID, fx, fy, fz, ...]

    def getNodalLoads(self, time=0.0):
        """
        Return a dict {nodeID: [fx, fy, fz, ...]} scaled by time series factor.
        """
        factor = self._tseries.getFactor(time)
        loads = {}
        for entry in self._load:
            if len(entry) > 1:
                nodeID = int(entry[0])
                forces = [f * factor for f in entry[1:]]
                loads[nodeID] = forces
        return loads

    def getNodeIDs(self):
        return [int(entry[0]) for entry in self._load if len(entry) > 1]
