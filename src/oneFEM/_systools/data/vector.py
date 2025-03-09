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
#Vector main object definition
#   definition of the Vector data object

from ..math_tools import _is_close
from numpy import dot, cross, sqrt
from numpy import array, copy
from .matrix import Matrix

class Vector(object):
    def __init__(self, init=[], shape=0, row_vector=True, dtype=float):
        # initialize variables
        self.__dtype = dtype
        self.__is_row_vector = row_vector

        if isinstance(shape, int):
            self.__length = shape
        else:
            raise ValueError("Vector.__init__(): shape must be an integer")

        # initialize data
        if isinstance(init, list):
            if len(init) > 0:
                self.__length = len(init)
                self.__data = array(init.copy(), dtype=self.__dtype)
                if not self.__is_row_vector:
                    self.__data = self.__data.transpose()
            
        elif isinstance(init, Vector):
            self.__copy(init)

        elif isinstance(init, array):
            if init.shape[0] > 0 and init.shape[1] == 1:
                self.__data = copy(init, dtype=self.__dtype)
                if not self.__is_row_vector:
                    self.__data = self.__data.transpose()
            elif init.shape[1] > 0 and init.shape[0] == 1:
                self.__data = copy(init, dtype=self.__dtype)
            else:
                raise ValueError("Vector.__init__(): data must be a 1D array")

        elif init==[]:
            if shape > 0:
                self.__data = array([0.0]*shape, dtype=self.__dtype)
                if not self.__is_row_vector:
                    self.__data = self.__data.transpose()
            else:
                self.__data = array([], dtype=self.__dtype)

        else:
            raise ValueError("Vector.__init__(): data type not supported")
        

    # private methods
    def __copy(self, vector_other):
        # copy the data of another vector
        # vector_other is a Vector object instance
        self.data = copy(vector_other.data)
        self.length = vector_other.length
        self.is_row_vector = vector_other.is_row_vector
        self.dtype = vector_other.dtype


    # public methods
    # implement matrix multiplication
    def __matmul__(self, vector_other):
        # multiply two vectors to obtain a matrix
        # vector_other is a vector instance
        matrix_result = Matrix()
        return matrix_result

    # implement division
    def __truediv__(self, scalar):
        # divide the vector by a scalar
        # scalar is a floating point number
        return Vector(self.data/scalar, dtype=self.dtype)
    

    #implement multiplication
    def __mul__(self, scalar):
        # multiply the vector by a scalar
        # scalar is a floating point number
        return Vector(self.data*scalar, dtype=self.dtype)
    

    #implement addition
    def __add__(self, vector_other):
        # add two vectors
        # vector_other is a Vector object instance
        return Vector(self.data + vector_other.data, dtype=self.dtype)
    

    #implement subtraction
    def __sub__(self, vector_other):
        # subtract two vectors
        # vector_other is a Vector object instance
        return Vector(self.data - vector_other.data, dtype=self.dtype)
    

    def __len__(self):
        # return the length of the vector
        return self.__length


    def __str__(self):
        # return the string representation of the vector
        return self.data.__str__()


    def _is_parallel(self, vector_other):
    # Check if two vectors are parallel
    # vector_other is a Vector object instance
    # Return True if they are parallel, False otherwise
    # if the dot product is 0 or smaller than a tolerance, vectors are parallel
        return _is_close(dot(self.data, vector_other.data), 0.0)
    

    def _cross(self, vector_other):
        return Vector(cross(self.__data, vector_other.data))
    

    def _normalize(self):
        magnitude = sum(val * val for val in self.__data)
        if _is_close(magnitude, 0.0):
            RuntimeWarning("Vector._normalize(): divison by zero skipped!")
            return  # Avoid division by zero
        
        self.__data /= sqrt(magnitude)