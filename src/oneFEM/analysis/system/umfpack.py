from scipy.sparse.linalg import use_solver
from .main import System

class UmfPack(System):
    def __init__(self):
        # Check if UMFPACK is enabled
        use_solver(useUmfpack=True)





