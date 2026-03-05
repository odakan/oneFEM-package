from .algorithm.main import Algorithm           # algorithm objects
from .algorithm.linear import Linear            # linear algorithm
from .constraints.main import ConstraintHandler # constraint handler objects
from .eigen.main import Eigen                   # eigen value and vector solvers
from .integrator.main import Integrator         # integrator (time-stepping) objects
from .integrator.dynamic.newmark import Newmark # Newmark dynamic integrator
from .integrator.dynamic.central_difference import CDiff  # Central Difference
from .numberer.main import Numberer             # d.o.f. numberer objects
from .system.main import System                 # system of equation (matrix) handlers
from .test.main import Test                     # convergence test objects

class Analysis(object):
    def __init__(self, ID=-1, algorithm=None, constraints=None,
                 integrator=None, numberer=None, system=None, test=None):
        self._ID = int(ID)
        self._time = 0.0
        self._solution_algorithm = None
        self._convergence_test = None
        self._constraint_handler = None
        self._solution_integrator = None
        self._system_of_equations = None
        self._dof_numberer = None

        # Free and fixed DOF indices
        self.uu = []
        self.pp = []

        # Store analysis components (use defaults if None)
        if algorithm is None:
            algorithm = Algorithm()
        if constraints is None:
            constraints = ConstraintHandler()
        if integrator is None:
            integrator = Integrator()
        if numberer is None:
            numberer = Numberer()
        if system is None:
            system = System()
        if test is None:
            test = Test()
        self.__add_analysis(algorithm, constraints, integrator,
                            numberer, system, test)


    def __add_analysis(self, alg=None, const=None, integ=None,
                       numb=None, syst=None, test=None):
        if alg is not None:
            self._solution_algorithm = alg
        if const is not None:
            self._constraint_handler = const
        if integ is not None:
            self._solution_integrator = integ
        if numb is not None:
            self._dof_numberer = numb
        if syst is not None:
            self._system_of_equations = syst
        if test is not None:
            self._convergence_test = test


    def _organize(self, model):
        """Identify and separate free and fixed DOFs."""
        self.pp = []  # Fixed DOFs
        self.uu = []  # Free DOFs
        for node in model.nodes:
            nDOF = node.getNDOF()
            node_dofs = node.getDOFs()
            fix = node._fix
            if hasattr(node_dofs, 'tolist'):
                dof_list = node_dofs.tolist()
            elif hasattr(node_dofs, 'data'):
                dof_list = node_dofs.data.tolist()
            else:
                dof_list = list(node_dofs)

            for j in range(nDOF):
                if fix[j]:
                    self.pp.append(dof_list[j])
                else:
                    self.uu.append(dof_list[j])


    def _analyze(self, model, nSteps=1, dt=0.0):
        """Run the analysis pipeline.

        The Integrator forms the tangent and residual.
        The Algorithm manages the iteration loop (or single pass).
        """
        # 1. Number DOFs and initialize elements
        model._domain()

        # 2. Separate free/fixed DOFs
        self._organize(model)

        integrator = self._solution_integrator
        algorithm = self._solution_algorithm
        is_dynamic = isinstance(integrator, (Newmark, CDiff))

        # For dynamic: initial assembly and compute a_0
        if is_dynamic:
            model._assemble(time=self._time)
            integrator.initialize(model)

        # 3. Time stepping loop
        for step in range(nSteps):
            # Assembly time: explicit uses current time, implicit uses t+dt
            if integrator.isExplicit():
                assembly_time = self._time
            else:
                assembly_time = self._time + dt
            model._assemble(time=assembly_time)

            # New step (integrator sets up predictor, stores M/C, etc.)
            integrator.newStep(model, dt, self._time)

            # Solve (algorithm calls formTangent/formUnbalance/solve/update)
            algorithm.solve(model, self.uu, self.pp,
                            integrator, self._system_of_equations,
                            self._convergence_test)

            # Commit (advance internal state, commit elements)
            integrator.commit(model)

            # Record
            model._record(time=self._time + dt)

            # Advance time
            self._time += dt
