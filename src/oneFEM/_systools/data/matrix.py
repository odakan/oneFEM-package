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
#Matrix main object definition
#   definition of the Matrix data object

from ..math_tools import _is_close
from numpy import array, copy

class Matrix(object):
    def __init__(self, init=[], shape=[0, 0], dtype=float):
        # initialize variables
        self.__dtype = dtype

        if isinstance(shape, int):
            self.__size = shape
        else:
            raise ValueError("Matrix.__init__(): shape must be an integer")

        # initialize data
        if isinstance(init, list):
            if isinstance(init[0], float) or isinstance(init[0], int):
                self.__data = array(init.copy(), dtype=self.__dtype)
                self.__size = len(init)
            else:
                raise ValueError("Matrix.__init__(): data must be a list of floats or integers")
    
            
        elif isinstance(init, Matrix):
            self.__copy(init)

        elif isinstance(init, array):
            if init.shape[0] > 0 and init.shape[1] == 1:
                self.__data = copy(init, dtype=self.__dtype)
            elif init.shape[1] > 0 and init.shape[0] == 1:
                self.__data = copy(init, dtype=self.__dtype)
            else:
                raise ValueError("Matrix.__init__(): data must be a 1D array")

        elif init==[]:
            if shape > 0:
                self.__data = array([0.0]*shape, dtype=self.__dtype)
                if not self.__is_row_vector:
                    self.__data = self.__data.transpose()
            else:
                self.__data = array([], dtype=self.__dtype)

        else:
            raise ValueError("Matrix.__init__(): data type not supported")
        

    # private methods
    def __copy(self, vector_other):
        # copy the data of another vector
        # vector_other is a Vector object instance
        self.data = copy(vector_other.data)
        self.length = vector_other.length
        self.dtype = vector_other.dtype


    # public methods
    # implement division
    def __truediv__(self, scalar):
        # divide the vector by a scalar
        # scalar is a floating point number
        return Matrix(self.data/scalar, dtype=self.dtype)
    

    #implement multiplication
    def __mul__(self, scalar):
        # multiply the vector by a scalar
        # scalar is a floating point number
        return Matrix(self.data*scalar, dtype=self.dtype)
    

    #implement addition
    def __add__(self, vector_other):
        # add two vectors
        # vector_other is a Vector object instance
        return Matrix(self.data + vector_other.data, dtype=self.dtype)
    

    #implement subtraction
    def __sub__(self, vector_other):
        # subtract two vectors
        # vector_other is a Vector object instance
        return Matrix(self.data - vector_other.data, dtype=self.dtype)
    

    def __len__(self):
        # return the length of the vector
        return self.__size


    def __str__(self):
        # return the string representation of the vector
        return self.data.__str__()

