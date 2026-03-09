import numpy as np
from .algorithm.main import Algorithm           # algorithm objects
from .algorithm.linear import Linear            # linear algorithm
from .constraints.main import ConstraintHandler # constraint handler objects
from .eigen.main import Eigen                   # eigen value and vector solvers
from .integrator.main import Integrator         # integrator (time-stepping) objects
from .integrator.dynamic.newmark import Newmark # Newmark dynamic integrator
from .integrator.dynamic.central_difference import CDiff  # Central Difference
from .numberer.main import Numberer             # d.o.f. numberer objects
from .numberer.rcm import RCM                   # RCM numberer
from .system.linear_soe import System, LinearSOE  # system of equation (matrix) handlers
from .system.sparse_general import SparseGeneral  # sparse system
from .test.main import Test                     # convergence test objects
from ..model.pattern.uniform_excitation import UniformExcitation
from .._systools.data import Vector

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

        # DOF maps (built during _analyze)
        self._node_dof_map = None
        self._elem_dof_map = None

        # Assembly system (SparseGeneral, always created internally)
        self._assembly_system = None

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


    def _buildDOFMaps(self, model):
        """Build node and element DOF maps from current DOF assignments."""
        node_dof_map = {}
        for nd in model.nodes:
            dof_data = nd.getDOFs()
            if hasattr(dof_data, 'tolist'):
                node_dof_map[nd._ID] = np.array(dof_data.tolist(),
                                                  dtype=np.int32)
            else:
                node_dof_map[nd._ID] = np.array(list(dof_data),
                                                  dtype=np.int32)

        elem_dof_map = {}
        for elem in model.elements:
            edofs = []
            for nd in elem.getNodes():
                edofs.extend(node_dof_map[nd._ID].tolist())
            elem_dof_map[elem._ID] = np.array(edofs, dtype=np.int32)

        self._node_dof_map = node_dof_map
        self._elem_dof_map = elem_dof_map


    def _assembleF(self, model, time=0.0):
        """Assemble external force vector from load patterns.

        Sets model.F as a Vector of size nDOF.
        """
        n = model.nDOF
        F_data = np.zeros(n)

        for pattern in model.patterns:
            if isinstance(pattern, UniformExcitation):
                continue  # handled below
            nodal_loads = pattern.getNodalLoads(time)
            for nodeID, forces in nodal_loads.items():
                dof_arr = self._node_dof_map.get(nodeID)
                if dof_arr is None:
                    continue
                for k, dof_idx in enumerate(dof_arr):
                    if k < len(forces):
                        F_data[dof_idx] += forces[k]

        # Apply UniformExcitation: F_eq = -M * r * a_g(t)
        M_sp = self._assembly_system.getM()
        if M_sp is not None:
            for pattern in model.patterns:
                if isinstance(pattern, UniformExcitation):
                    a_g = pattern.getAcceleration(time)
                    if a_g != 0.0:
                        direction = pattern.direction  # 1-based
                        M_data = M_sp.toarray()
                        r = np.zeros(n)
                        for node in model.nodes:
                            dof_arr = self._node_dof_map.get(node._ID)
                            if dof_arr is not None and direction <= len(dof_arr):
                                r[dof_arr[direction - 1]] = 1.0
                        F_data -= a_g * M_data.dot(r)

        model.F = Vector(list(F_data))


    def _analyze(self, model, nSteps=1, dt=0.0):
        """Run the analysis pipeline.

        Analysis is the single entry point. It owns:
        - DOF numbering (via Numberer)
        - System setup (SparseGeneral for assembly)
        - Force assembly (from patterns)
        - Integrator/Algorithm orchestration
        """
        numberer = self._dof_numberer
        use_rcm = isinstance(numberer, RCM)

        # 1. Number DOFs and initialize elements
        if use_rcm:
            model._domain(numberer=numberer)
        else:
            model._domain()

        # 2. Separate free/fixed DOFs
        if use_rcm:
            self.uu = numberer._uu
            self.pp = numberer._pp
        else:
            self._organize(model)

        # 3. Build DOF maps (after _domain resolved node tags to objects)
        if use_rcm:
            self._node_dof_map = numberer._node_dof_map
            self._elem_dof_map = numberer._elem_dof_map
        else:
            self._buildDOFMaps(model)

        # 4. Set up assembly system
        # If user passed a LinearSOE (SparseGeneral, UmfPackSOE, etc.), use it as
        # both assembly and solver system — single object, no duplication.
        solver_system = self._system_of_equations
        if isinstance(solver_system, LinearSOE):
            self._assembly_system = solver_system
            self._assembly_system.setSize(model.nDOF, self._elem_dof_map,
                                           model.elements)
        else:
            self._assembly_system = SparseGeneral()
            self._assembly_system.setSize(model.nDOF, self._elem_dof_map,
                                           model.elements)
            # Non-SparseGeneral solver (FullGeneral etc.) or default
            if isinstance(solver_system, System) and type(solver_system) is not System:
                pass  # use the user's solver
            else:
                solver_system = self._assembly_system

        # Store on model for lazy K/M access (eigen, etc.)
        model._system = self._assembly_system

        # 5. Wire integrator to assembly system
        integrator = self._solution_integrator
        algorithm = self._solution_algorithm
        integrator.setLinks(self._assembly_system)

        is_dynamic = isinstance(integrator, (Newmark, CDiff))

        # Initialize state vectors on model
        n = model.nDOF
        if model.F is None or len(model.F) != n:
            model.F = Vector([0.0] * n)
        if model.u is None or len(model.u) != n:
            model.u = Vector([0.0] * n)
        if model.v is None or len(model.v) != n:
            model.v = Vector([0.0] * n)
        if model.a is None or len(model.a) != n:
            model.a = Vector([0.0] * n)

        # 6. For dynamic: initial assembly and compute a_0
        if is_dynamic:
            integrator._assembleK(model)
            self._assembleF(model, time=self._time)
            integrator.initialize(model)

        # 7. Time stepping loop
        for step in range(nSteps):
            # Assembly time: explicit uses current time, implicit uses t+dt
            if integrator.isExplicit():
                assembly_time = self._time
            else:
                assembly_time = self._time + dt

            # Assemble F from patterns at the correct time
            # For first static step, also do initial K/M assembly
            if not is_dynamic and step == 0:
                integrator._assembleK(model)
            self._assembleF(model, time=assembly_time)

            # New step (integrator sets up predictor, stores M/C, etc.)
            integrator.newStep(model, dt, self._time)

            # Solve (algorithm calls formTangent/formUnbalance/solve/update)
            algorithm.solve(model, self.uu, self.pp,
                            integrator, solver_system,
                            self._convergence_test)

            # Commit (advance internal state, commit elements)
            integrator.commit(model)

            # Record
            model._record(time=self._time + dt)

            # Advance time
            self._time += dt
