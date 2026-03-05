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

import warnings
from ..math_tools import _is_close
from numpy import dot, cross, sqrt, abs as np_abs
from numpy import array, copy, zeros, ndarray, asarray

class Vector(object):
    def __init__(self, init=None, shape=0, row_vector=True, dtype=float):
        # initialize variables
        self.__dtype = dtype
        self.__is_row_vector = row_vector

        if isinstance(shape, int):
            self.__length = shape
        else:
            raise ValueError("Vector.__init__(): shape must be an integer")

        # handle None default
        if init is None:
            init = []

        # initialize data
        if isinstance(init, list):
            if len(init) > 0:
                self.__length = len(init)
                self.__data = array(init, dtype=self.__dtype)
            elif shape > 0:
                self.__data = zeros(shape, dtype=self.__dtype)
            else:
                self.__data = array([], dtype=self.__dtype)

        elif isinstance(init, Vector):
            self.__copy(init)

        elif isinstance(init, ndarray):
            self.__data = copy(init).astype(self.__dtype).flatten()
            self.__length = len(self.__data)

        elif shape > 0:
            self.__data = zeros(shape, dtype=self.__dtype)

        else:
            raise ValueError("Vector.__init__(): data type not supported")


    # property accessors
    @property
    def data(self):
        return self.__data

    @property
    def length(self):
        return self.__length

    @property
    def dtype(self):
        return self.__dtype

    @property
    def is_row_vector(self):
        return self.__is_row_vector

    # private methods
    def __copy(self, vector_other):
        self.__data = copy(vector_other.data)
        self.__length = vector_other.length
        self.__is_row_vector = vector_other.is_row_vector
        self.__dtype = vector_other.dtype

    # indexing support
    def __getitem__(self, key):
        result = self.__data[key]
        if isinstance(result, ndarray):
            return Vector(list(result), dtype=self.__dtype)
        return result

    def __setitem__(self, key, value):
        if isinstance(value, Vector):
            self.__data[key] = value.data
        else:
            self.__data[key] = value

    # numpy interop
    def __array__(self, dtype=None):
        if dtype is not None:
            return self.__data.astype(dtype)
        return self.__data

    # public methods
    # implement matrix multiplication (outer product for vectors)
    def __matmul__(self, vector_other):
        return self.outer(vector_other)

    # implement division
    def __truediv__(self, scalar):
        return Vector(list(self.__data / scalar), dtype=self.__dtype)


    #implement multiplication
    def __mul__(self, scalar):
        return Vector(list(self.__data * scalar), dtype=self.__dtype)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)


    #implement addition
    def __add__(self, vector_other):
        if isinstance(vector_other, Vector):
            return Vector(list(self.__data + vector_other.data), dtype=self.__dtype)
        return Vector(list(self.__data + vector_other), dtype=self.__dtype)


    #implement subtraction
    def __sub__(self, vector_other):
        if isinstance(vector_other, Vector):
            return Vector(list(self.__data - vector_other.data), dtype=self.__dtype)
        return Vector(list(self.__data - vector_other), dtype=self.__dtype)

    #implement negation
    def __neg__(self):
        return Vector(list(-self.__data), dtype=self.__dtype)


    def __len__(self):
        try:
            return self.__length
        except AttributeError:
            return 0


    def __str__(self):
        return self.__data.__str__()

    def __iter__(self):
        return iter(self.__data)

    # linear algebra
    def dot(self, other):
        if isinstance(other, Vector):
            return float(dot(self.__data, other.data))
        return float(dot(self.__data, asarray(other)))

    def outer(self, other):
        from .matrix import Matrix
        from numpy import outer as np_outer
        if isinstance(other, Vector):
            return Matrix(init=np_outer(self.__data, other.data))
        return Matrix(init=np_outer(self.__data, asarray(other)))

    def _is_parallel(self, vector_other):
        # Check if two vectors are parallel
        # Parallel means cross product magnitude is ~0
        c = cross(self.__data, vector_other.data)
        return _is_close(float(dot(c, c)), 0.0)


    def _cross(self, vector_other):
        return Vector(list(cross(self.__data, vector_other.data)))


    def _normalize(self):
        magnitude = float(dot(self.__data, self.__data))
        if _is_close(magnitude, 0.0):
            warnings.warn("Vector._normalize(): division by zero skipped!", RuntimeWarning)
            return

        self.__data = self.__data / sqrt(magnitude)

    def norm(self):
        return float(sqrt(dot(self.__data, self.__data)))

    def tolist(self):
        return self.__data.tolist()
