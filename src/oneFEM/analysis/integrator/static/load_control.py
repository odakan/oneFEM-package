from ..main import Integrator

class LoadControl(Integrator):
    def __init__(self, iID=-1, dLambda=1.0):
        super().__init__(iID)
        self._dLambda = dLambda
