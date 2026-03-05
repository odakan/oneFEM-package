class System(object):
    def __init__(self, ID=-1):
        self._ID = ID

    def solve(self, A, b):
        raise NotImplementedError("System.solve(): method must be implemented by subclasses.")