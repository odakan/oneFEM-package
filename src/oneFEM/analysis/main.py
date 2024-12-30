##-----------------------------------------------------------------------##
#                                                                         #
#        #--oneFEM--#: One FEM software in a galaxy far far away          #
#                                                                         #
#                   Computational Mechanics 2022                          #
#                       University of Pavia                               #
#               Written by: Onur Deniz AKAN, IUSS Pavia                   #
#                         24 January 2022                                 #
#                                                                         #
##-----------------------------------------------------------------------##
#ANALYSIS main object definition
#   keeps universal vars, functions and list of analyses

from .algorithm.main import Algorithm           # algorithm objects
from .constraints.main import ConstraintHandler # constraint handler objects
from .eigen.main import Eigen                   # eigen value and vector solvers
from .integrator.main import Integrator         # integrator (time-stepping) objects
from .numberer.main import Numberer             # d.o.f. numberer objects
from .system.main import System                 # system of equation (matrix) handlers
from .test.main import Test                     # convergence test objects

class Analysis(object):
    def __init__(self, ID, dynamic=None, algorithm=None, constraints=None, integrator=None,
                 numberer=None, system=None, test=None, tolerance=None):
        # Initialize the properties
        self.__ID = int(ID)                     # integer
        self.__nSteps = int(0)                  # integer
        self.__dt = float(0.0)                  # float
        self.__solution_type = None             # boolean [False:static - True:dynamic]
        self.__solution_algorithm = None        # Algorithm object
        self.__convergence_test = None          # Test object
        self.__convergence_tolerance = None     # float
        self.__constraint_handler = None        # Constratints object
        self.__solution_integrator = None       # Integrator object
        self.__system_of_equations = None       # System object
        self.__dof_numberer = None              # Numberer object
        self.__is_ready_for_analysis = False    # boolean [True:analyze becomes available]

        # Initialize convergence-related properties
        self.__norm = None
        self.__nIter = None
        self.__verbosity = 0
        self.__is_converged = False

        # add analysis
        self.__add_analysis(dynamic, algorithm, constraints, integrator,
                 numberer, system, test, tolerance)

        # Initialize vectors for free and fixed degrees of freedom
        self.uu = []  # Free DOF vector
        self.pp = []  # Fixed DOF vector


    def __add_analysis(self, alg, const, int, numb, sys, test, tol):
        pass


    def set(self, cmd):
        # Parse the add analysis command
        if len(cmd) < 7:
            raise ValueError("DOMAIN: Improper analysis definition!")

        atype = cmd[2]  # Analysis type
        ahandle = cmd[3]  # Analysis handle
        data = cmd[1:]  # Analysis-specific data

        # Process based on analysis type and handle
        if "static" in ahandle:
            if "linear" in ahandle:
                # Linear solver
                analysis = linstat().add_analysis(data)
            elif "newton" in ahandle or "newton-raphson" in ahandle:
                # Newton-Raphson algorithm
                analysis = nr_solve().add_analysis(data)
            elif "krylov" in ahandle or "krylov-newton" in ahandle:
                # Newton-Raphson with line search
                analysis = kn_solve().add_analysis(data)
            else:
                raise ValueError(f"ANALYSIS: {ahandle} static analysis does not exist!")
        elif "transient" in atype or "dynamic" in atype:
            if "linear" in ahandle:
                # Newmark implicit solver
                analysis = lindyn().add_analysis(data)
            elif ahandle in {"newmark", "nmk"}:
                # Newmark implicit solver
                analysis = nmk_solve().add_analysis(data)
            elif ahandle in {"central_difference", "cd"}:
                # Central difference explicit solver
                analysis = cdiff().add_analysis(data)
            else:
                raise ValueError(f"ANALYSIS: {ahandle} transient analysis does not exist!")
        else:
            raise ValueError(f"ANALYSIS: {atype} type analysis does not exist!")

        # If all settings are read, set the analysis 
        self._set_analysis(algorithm, constraints, integrator,
                 numberer, system, test, tolerance)

    def _analyze(self, model, nSteps=None, dt=None):
        # Trigger a step of analysis
        pass

    def __organize(self, model):
        # Identify and separate free and fixed DOFs
        self.pp = []  # Fixed DOFs
        self.uu = []  # Free DOFs
        for node in model.nodes:
            for j in range(node.nDOF):
                if node.fix[j] == 1:  # Fixed node
                    self.pp.append(node.dofs[j])
                else:  # Free node
                    self.uu.append(node.dofs[j])

    def convergence(self, verbose=-1):
        # print convergence statistics
        # verbose:  0 print now and nothing later
        #           1 print now and keep printing after each converge
        #           2 print now and keep printing after each test
        __current_convergence_info = "{:8.6f}  {:d}".format(self.__norm, self.__nIter)
        __step_convergence_info = "{:8.6f}  {:d}".format(self.__norm, self.__nIter)
        if verbose == -1:
            if self.__verbosity == 1 and self.__is_converged:
                print(__step_convergence_info)

            elif self.__verbosity == 2:
                print(__current_convergence_info)


    def __test(self):
        # update the norm and increment the number of iterations
        self.__is_converged = False

        # compute the norm
        self__norm = self.__convergence_test.norm()

        # check for convergence
        if self.__norm < self.__convergence_tolerance:
            self.__is_converged = True

        # print some stats if requested
        self.convergence()  
