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
from ..backend import np

class Matrix(object):
    def __init__(self, init=[], shape=[0, 0], dtype=float):
        # initialize variables
        self.__dtype = dtype

        if isinstance(init, np.ndarray) and init.ndim == 2:
            self.__data = init.astype(dtype)
            self.__size = list(init.shape)
            return

        if isinstance(init, list):
            if len(init) > 0:
                # list-of-lists -> 2D array
                if isinstance(init[0], (list, np.ndarray)):
                    self.__data = np.array(init, dtype=self.__dtype)
                    self.__size = list(self.__data.shape)
                    return
                # flat list of numbers (legacy)
                elif isinstance(init[0], (float, int)):
                    self.__data = np.array(init, dtype=self.__dtype).reshape(1, -1)
                    self.__size = list(self.__data.shape)
                    return

            # empty list -> use shape
            if isinstance(shape, list) and len(shape) == 2:
                self.__size = shape
                if shape[0] > 0 and shape[1] > 0:
                    self.__data = np.zeros(shape, dtype=self.__dtype)
                else:
                    self.__data = np.array([]).reshape(0, 0).astype(self.__dtype)
            else:
                self.__size = [0, 0]
                self.__data = np.array([]).reshape(0, 0).astype(self.__dtype)

        elif isinstance(init, Matrix):
            self.__copy(init)

        else:
            raise ValueError("Matrix.__init__(): data type not supported")


    # property accessors
    @property
    def data(self):
        return self.__data

    @property
    def size(self):
        return self.__size

    @property
    def shape(self):
        return self.__size

    @property
    def dtype(self):
        return self.__dtype

    # private methods
    def __copy(self, matrix_other):
        self.__data = np.copy(matrix_other.data)
        self.__size = list(matrix_other.size)
        self.__dtype = matrix_other.dtype

    # indexing support
    def __getitem__(self, key):
        result = self.__data[key]
        if isinstance(result, np.ndarray) and result.ndim == 2:
            return Matrix(init=result, dtype=self.__dtype)
        elif isinstance(result, np.ndarray) and result.ndim == 1:
            from .vector import Vector
            return Vector(list(result), dtype=self.__dtype)
        return result

    def __setitem__(self, key, value):
        if isinstance(value, Matrix):
            self.__data[key] = value.data
        elif isinstance(value, np.ndarray):
            self.__data[key] = value
        else:
            from .vector import Vector
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
    # implement division
    def __truediv__(self, scalar):
        return Matrix(init=self.__data / scalar, dtype=self.__dtype)


    #implement multiplication
    def __mul__(self, scalar):
        return Matrix(init=self.__data * scalar, dtype=self.__dtype)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)


    #implement addition
    def __add__(self, other):
        if isinstance(other, Matrix):
            return Matrix(init=self.__data + other.data, dtype=self.__dtype)
        return Matrix(init=self.__data + np.asarray(other), dtype=self.__dtype)

    #implement in-place addition (for assembly scatter)
    def __iadd__(self, other):
        if isinstance(other, Matrix):
            self.__data += other.data
        else:
            self.__data += np.asarray(other)
        return self


    #implement subtraction
    def __sub__(self, other):
        if isinstance(other, Matrix):
            return Matrix(init=self.__data - other.data, dtype=self.__dtype)
        return Matrix(init=self.__data - np.asarray(other), dtype=self.__dtype)


    def __len__(self):
        return self.__size

    def __str__(self):
        return self.__data.__str__()

    @property
    def T(self):
        """Return transpose as a new Matrix."""
        return Matrix(init=self.__data.T.copy(), dtype=self.__dtype)

    def __matmul__(self, other):
        """Matrix @ Matrix -> Matrix, Matrix @ Vector -> Vector."""
        from .vector import Vector
        if isinstance(other, Matrix):
            return Matrix(init=self.__data @ other.data, dtype=self.__dtype)
        if isinstance(other, Vector):
            result = self.__data @ other.data
            return Vector(list(result), dtype=self.__dtype)
        return NotImplemented

    # linear algebra
    def dot(self, other):
        from .vector import Vector
        if isinstance(other, Vector):
            result = self.__data.dot(other.data)
            return Vector(list(result), dtype=self.__dtype)
        elif isinstance(other, Matrix):
            result = self.__data.dot(other.data)
            return Matrix(init=result, dtype=self.__dtype)
        result = self.__data.dot(np.asarray(other))
        return result

    def solve(self, rhs):
        from .vector import Vector
        if isinstance(rhs, Vector):
            result = np.linalg.solve(self.__data, rhs.data)
            return Vector(list(result), dtype=self.__dtype)
        result = np.linalg.solve(self.__data, np.asarray(rhs))
        return Vector(list(result), dtype=self.__dtype)
