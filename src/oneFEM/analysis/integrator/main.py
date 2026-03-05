import numpy as np
from ..._systools.data import Vector, Matrix

class Integrator(object):
    def __init__(self, ID=-1):
        self._ID = ID
        self._K_eff = None   # effective tangent (numpy array)
        self._R = None       # residual (numpy array)
        self._current_time = 0.0
        self._dt = 0.0
        self._U = None       # current trial displacement (numpy array)

    def initialize(self, model):
        """Called once before the time loop for dynamic integrators."""
        pass

    def newStep(self, model, dt, current_time):
        """Called at the start of each step."""
        self._current_time = current_time
        self._dt = dt
        # Start from committed displacements
        self._U = model.getCommittedDisp()
        # Set on model and commit so elements see the starting state
        model.u = Vector(list(self._U))
        model._commit(model.F, model.u)
        for element in model.elements:
            element._update()

    def formTangent(self, model, tangent='current'):
        """Form the effective tangent matrix.
        Default: K_T from assembly (static linear/nonlinear).
        """
        if tangent == 'current':
            model._assemble(time=self._current_time + self._dt)
            # Restore u after assembly (assembly zeros it)
            model.u = Vector(list(self._U))
        self._K_eff = np.asarray(model.K).copy()

    def formUnbalance(self, model):
        """Form the residual vector.
        Default: R = F_ext - F_int (static).
        """
        F_ext = np.asarray(model.F)
        F_int = model.getInternalForce()
        self._R = F_ext - F_int

    def update(self, model, dU):
        """Apply displacement increment and update domain.
        Default: u += dU, commit to nodes, update elements.
        """
        self._U += dU
        model.u = Vector(list(self._U))
        model._commit(model.F, model.u)
        for element in model.elements:
            element._update()

    def commit(self, model):
        """Commit elements after convergence."""
        for element in model.elements:
            element._commit()

    def isExplicit(self):
        return False

    def getTangent(self):
        return self._K_eff

    def getUnbalance(self):
        return self._R
