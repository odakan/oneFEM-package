class TSeries(object):
    def __init__(self, tID=-1):
        self._ID = tID

    def getFactor(self, time=0.0):
        raise NotImplementedError("TSeries.getFactor(): method must be implemented by subclasses.")
