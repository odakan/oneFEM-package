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
    def __init__(self, ID, algorithm=Algorithm(), constraints=ConstraintHandler(), integrator=Integrator(),
                 numberer=Numberer(), system=System(), test=Test()):
        # Initialize the properties
        self.__ID = int(ID)                     # integer
        self.__hist_steps = []                  # list of lists step history [[dt, nSteps], []]
        self.__time = 0.0                       # float analysis clock (current)
        self.__dt = float(0.0)                  # float last dt
        self.__solution_type = None             # boolean [False:static - True:dynamic]
        self.__solution_algorithm = None        # Algorithm object
        self.__convergence_test = None          # Test object
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

        # Initialize vectors for free and fixed degrees of freedom
        self.uu = []  # Free DOF vector
        self.pp = []  # Fixed DOF vector

        # check analysis
        self.__add_analysis(algorithm, constraints, integrator,
                            numberer, system, test)


    def __add_analysis(self, alg=None, const=None, integ=None 
                       , numb=None, syst=None, test=None):
        # do some chekcs
        if alg != None:
            pass
        
        elif const != None:
            pass

        elif integ != None:
            pass

        elif numb != None:
            pass

        elif syst != None:
            pass

        elif test != None:
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


    # ANALYSIS API (called upon by the SimManager)
    def _convergence(self, verbose=-1):
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


    def _analyze(self, model, nSteps=None, dt=None):
        # Trigger a step of analysis
        pass
