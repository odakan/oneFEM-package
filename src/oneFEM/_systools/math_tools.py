##-----------------------------------------------------------------------##
#                                                                         #
#      #--oneFEM--#: One FEM software in a galaxy far far away            #
#                                                                         #
#                   Computational Mechanics 2025                          #
#                   University of Chieti-Pescara                          #
#                 Written by: Onur Deniz AKAN, Ud'A                       #
#                         9 February 2025                                 #
#                                                                         #
##-----------------------------------------------------------------------##
# 
# Author: Onur Deniz Akan
# Date: 14/02/2025
# Version: 0.1
#
#Math tools required in computations
#   definition of frequently-needed math functions

# import data structures
#from .data import Vector, Matrix, Tensor, CTensor

# import precision math tools
from numpy import isclose

# define globals
RTOL = 1e-6
ATOL = 1e-12

def _is_close(val1, val2, atol=None, rtol=RTOL):
    # Check if two values are close
    # val1 and val2 are floating point numbers
    # atol is the absolute tolerance (optional)
    # rtol is the relative tolerance (optional)
    # Return True if they are close, False otherwise
    if atol is None:
        return isclose(val1, val2, rtol=rtol, atol=ATOL)
    else:
        return isclose(val1, val2, atol=atol)

def _is_zero(val):
    # Check if the value is close to zero
    # val is a floating point number
    # Return True if it is close, False otherwise
    return isclose(val, 0.0, rtol=RTOL, atol=ATOL)