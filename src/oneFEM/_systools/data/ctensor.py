"""
ctensor.py - Compressed Tensor class for symmetric 2nd and 4th order tensors.

Full Python translation of CTensor.h/.cpp (Akan, 2023).

All 2nd-order tensors use Voigt notation: [σ11, σ22, σ33, σ12, σ23, σ13].
4th-order tensors are 6x6 matrices (compressed symmetric representation).

CTensor supports 5 matrix representations (Helnwein, 2001):
    FULL = 0       Full (unsymmetric) tensor matrix
    COV = 1        Covariant (strain-like)
    CONTR = 2      Contravariant (stress/stiffness-like)
    COVCONTR = 3   Mixed covariant-contravariant (rows=Cov, cols=Contr)
    CONTRCOV = 4   Mixed contravariant-covariant (rows=Contr, cols=Cov)

Rule of thumb:
    Stress and Stiffness-like tensors -> contravariant (rep=2)
    Strain-like tensors -> covariant (rep=1)

Reference:
    Helnwein, P. (2001). Some remarks on the compressed matrix representation
    of symmetric second-order and fourth-order tensors. CMAME, 190(22), 2753-2770.

Written: Onur Deniz Akan (IUSS Pavia)
Based on: CTensor.h/.cpp (Akan, 2023)
"""

from math import sqrt as _sqrt
from ..backend import np

# =========================================================================
# Constants
# =========================================================================

DBL_EPSILON = 2.2204460492503131e-16
UP_LIMIT = 1.0e+30
LOW_LIMIT = 20.0 * DBL_EPSILON
_SMALL_VALUE = 1e-8

# Pre-computed factor tuples for 3D (avoid repeated allocation)
_FACTORS_ONES_6 = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
_FACTORS_COV_6 = (1.0, 1.0, 1.0, 0.5, 0.5, 0.5)
_FACTORS_CONTR_6 = (1.0, 1.0, 1.0, 2.0, 2.0, 2.0)
_FACTORS_ONES_3 = (1.0, 1.0, 1.0)
_FACTORS_COV_3 = (1.0, 1.0, 0.5)
_FACTORS_CONTR_3 = (1.0, 1.0, 2.0)


# =========================================================================
# CTensor class — full Python translation of C++ CTensor
# =========================================================================

class CTensor:
    """Compressed Tensor for 2nd and 4th order symmetric tensors.

    Stores data in a flat list of size MAX_SIZE (81), using Voigt notation.
    Supports full representation system (Full/Cov/Contr/CovContr/ContrCov),
    factor tables, fused operations, and cached constants.
    """

    __slots__ = ('_dim', '_order', '_repr', '_data', '_nRows', '_nCols')

    MAX_SIZE = 81  # 9x9 max for FULL-rep 3D symmetric 4th-order tensors

    # Representation enums
    FULL = 0
    COV = 1
    CONTR = 2
    COVCONTR = 3
    CONTRCOV = 4

    # Representation name map for __repr__
    _REP_NAMES = {
        -1: 'unset', 0: 'full', 1: 'covariant', 2: 'contravariant',
        3: 'mixed (covcontr)', 4: 'mixed (contrcov)'
    }

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, *args, **kwargs):
        """Construct a CTensor.

        Dispatch based on argument types:
            CTensor()                           — empty default
            CTensor(other: CTensor)             — copy
            CTensor(nRows, rep)                 — 2nd-order zeroed
            CTensor(data, nRows, rep)           — 2nd-order from array
            CTensor(nRows, nCols, rep)          — 4th-order zeroed
            CTensor(data, nRows, nCols, rep)    — 4th-order from array

        For dev+vol constructors, use the factory classmethods:
            CTensor.from_dev_vol(dev, vol, linearized)
            CTensor.from_dev_vol_tensor(dev, vol_tensor, linearized)
        """
        self._dim = 0
        self._order = 0
        self._repr = -1
        self._data = [0.0] * self.MAX_SIZE
        self._nRows = 0
        self._nCols = 0

        nargs = len(args)
        if nargs == 0:
            # default: empty tensor
            return
        elif nargs == 1 and isinstance(args[0], CTensor):
            # copy constructor
            other = args[0]
            self._dim = other._dim
            self._order = other._order
            self._repr = other._repr
            self._nRows = other._nRows
            self._nCols = other._nCols
            n = self._nRows * self._nCols
            for i in range(n):
                self._data[i] = other._data[i]
        elif nargs == 2:
            # CTensor(nRows, rep) — 2nd-order zeroed
            nRows, rep = int(args[0]), int(args[1])
            if rep < 0 or rep > 2:
                raise ValueError("CTensor: 2nd-order tensor does not have a mixed representation!")
            self._repr = rep
            self._nRows = nRows
            self._nCols = 1
            self._set_order(2)
            self._matrix_dim(nRows)
        elif nargs == 3:
            if hasattr(args[0], '__len__') or hasattr(args[0], '__iter__'):
                # CTensor(data, nRows, rep) — 2nd-order from array
                data_in = self._flatten_input(args[0])
                nRows, rep = int(args[1]), int(args[2])
                if rep < 0 or rep > 2:
                    raise ValueError("CTensor: 2nd-order tensor does not have a mixed representation!")
                self._repr = rep
                self._nRows = nRows
                self._nCols = 1
                self._set_order(2)
                self._matrix_dim(nRows)
                for i in range(nRows):
                    self._data[i] = data_in[i]
            else:
                # CTensor(nRows, nCols, rep) — 4th-order zeroed
                nRows, nCols, rep = int(args[0]), int(args[1]), int(args[2])
                if rep < 0 or rep > 4:
                    raise ValueError("CTensor: unsupported matrix representation!")
                self._repr = rep
                self._nRows = nRows
                self._nCols = nCols
                self._set_order(4)
                self._matrix_dim(nRows)
        elif nargs == 4:
            # CTensor(data, nRows, nCols, rep) — 4th-order from array
            data_in = self._flatten_input(args[0])
            nRows, nCols, rep = int(args[1]), int(args[2]), int(args[3])
            if rep < 0 or rep > 4:
                raise ValueError("CTensor: unsupported matrix representation!")
            self._repr = rep
            self._nRows = nRows
            self._nCols = nCols
            self._set_order(4)
            self._matrix_dim(nRows)
            n = nRows * nCols
            for i in range(n):
                self._data[i] = data_in[i]
        else:
            raise TypeError(f"CTensor: unsupported constructor with {nargs} arguments")

    @staticmethod
    def _flatten_input(data):
        """Flatten any array-like input to a list of floats (duck-typed, no numpy)."""
        if hasattr(data, 'ravel'):  # numpy array (duck-typed, no import needed)
            return [float(x) for x in data.ravel()]
        if hasattr(data, '__len__'):
            # Check if 2D (list of lists / matrix)
            if len(data) > 0 and hasattr(data[0], '__len__'):
                return [float(x) for row in data for x in row]
            return [float(x) for x in data]
        return [float(data)]

    @classmethod
    def from_dev_vol(cls, deviatoric, volumetric, linearized):
        """Construct from deviatoric CTensor + volumetric scalar."""
        result = cls()
        result.set_data_dev_vol(deviatoric, volumetric, linearized)
        return result

    @classmethod
    def from_dev_vol_tensor(cls, deviatoric, volumetric, linearized):
        """Construct from deviatoric CTensor + volumetric CTensor."""
        result = cls()
        result.set_data_dev_vol_tensor(deviatoric, volumetric, linearized)
        return result

    def copy(self):
        """Return a deep copy (fast path, bypasses __init__ dispatch)."""
        obj = object.__new__(CTensor)
        obj._dim = self._dim
        obj._order = self._order
        obj._repr = self._repr
        obj._nRows = self._nRows
        obj._nCols = self._nCols
        obj._data = self._data[:]  # list slice copy
        return obj

    # ------------------------------------------------------------------
    # Trivial inline methods
    # ------------------------------------------------------------------

    def zero(self):
        """Zero all data."""
        n = self._nRows * self._nCols
        for i in range(n):
            self._data[i] = 0.0

    def get_rep(self):
        """Return the representation enum."""
        return self._repr

    def get_order(self):
        """Return the tensor order (2 or 4)."""
        return self._order

    def length(self):
        """Return total number of stored components."""
        return self._nRows * self._nCols

    def no_rows(self):
        """Return number of rows."""
        return self._nRows

    def no_cols(self):
        """Return number of columns."""
        return self._nCols

    def trace(self):
        """Trace: sum of first dim normal components."""
        s = self._data[0] + self._data[1]
        if self._dim == 3:
            s += self._data[2]
        return s

    # ------------------------------------------------------------------
    # Element access
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        """Access elements: t[i] for 2nd-order, t[i,j] for 4th-order."""
        if isinstance(key, tuple):
            row, col = key
            return self._data[row * self._nCols + col]
        return self._data[key]

    def __setitem__(self, key, value):
        """Set elements: t[i] = v for 2nd-order, t[i,j] = v for 4th-order."""
        if isinstance(key, tuple):
            row, col = key
            self._data[row * self._nCols + col] = value
        else:
            self._data[key] = value

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def _set_order(self, ord_val):
        """Set tensor order (2 or 4)."""
        if ord_val not in (2, 4):
            raise ValueError("CTensor: unsupported tensor order!")
        self._order = ord_val

    def _matrix_dim(self, nRows):
        """Set spatial dimension from matrix size."""
        self._dim = 2 if nRows < 6 else 3

    def set_order(self, ord_val):
        """Public set_order (matches C++ API)."""
        self._set_order(ord_val)
        return 0

    def make_rep(self, rep):
        """Convert to the given representation."""
        if self._order == 2 and rep > 2:
            raise ValueError("CTensor: 2nd-order tensor does not have a mixed representation!")
        if rep < 0 or rep > 4:
            raise ValueError("CTensor: unsupported matrix representation!")
        if self._repr == rep:
            return 0
        elif self._repr == -1:
            self._repr = rep
        else:
            if rep == 0:
                self._to_full()
            elif rep == 1:
                self._to_cov()
            elif rep == 2:
                self._to_contr()
            elif rep == 3:
                self._to_cov_contr()
            else:
                self._to_contr_cov()
        return 0

    def volumetric(self, linearized):
        """Return the volumetric part of this tensor."""
        if linearized:
            if self._order == 2:
                if self._repr == 1:
                    rep = 3  # CovContr
                elif self._repr == 2:
                    rep = 4  # ContrCov
                else:
                    rep = 0
                return CTensor.Constants.IIvol(self._dim, rep).__xor__(self)
            else:
                raise NotImplementedError("CTensor: 4th-order volumetric return not implemented!")
        else:
            raise NotImplementedError("CTensor: multiplicative volumetric decomposition not implemented!")

    def deviator(self, linearized):
        """Return the deviatoric part of this tensor."""
        if linearized:
            if self._order == 2:
                # Fast path for 3D CONTR and COV (most common cases)
                if self._dim == 3 and self._repr in (1, 2):
                    result = self.copy()
                    vol = (result._data[0] + result._data[1] + result._data[2]) / 3.0
                    result._data[0] -= vol
                    result._data[1] -= vol
                    result._data[2] -= vol
                    # shear components unchanged: IIdev shear diag = 1.0
                    # for CONTRCOV (used by CONTR self) and COVCONTR (used by COV self)
                    return result
                # General path
                if self._repr == 1:
                    rep = 3
                elif self._repr == 2:
                    rep = 4
                else:
                    rep = 0
                return CTensor.Constants.IIdev(self._dim, rep).__xor__(self)
            else:
                raise NotImplementedError("CTensor: 4th-order deviator return not implemented!")
        else:
            raise NotImplementedError("CTensor: multiplicative deviatoric decomposition not implemented!")

    def make_vector(self):
        """Return data as a numpy (n,) array. Only for 2nd-order tensors."""
        if self._order != 2:
            raise ValueError("CTensor: cannot make vector from a 4th-order tensor!")
        return np.array(self._data[:self._nRows])

    def make_matrix(self):
        """Return data as a numpy (n, m) array. Only for 4th-order tensors."""
        if self._order != 4:
            raise ValueError("CTensor: cannot make matrix from a 2nd-order tensor!")
        n = self._nRows * self._nCols
        return np.array(self._data[:n]).reshape(self._nRows, self._nCols)

    def to_vector(self):
        """Return data as a Vector object. Only for 2nd-order tensors."""
        from .vector import Vector
        return Vector(list(self.make_vector()), dtype=float)

    def to_matrix(self):
        """Return data as a Matrix object. Only for 4th-order tensors."""
        from .matrix import Matrix
        return Matrix(init=self.make_matrix(), dtype=float)

    # ------------------------------------------------------------------
    # setData / resize
    # ------------------------------------------------------------------

    def set_data(self, data_in, size, rep):
        """Set 2nd-order tensor data from array."""
        if rep < 0 or rep > 2:
            raise ValueError("CTensor: 2nd-order tensor does not have a mixed representation!")
        data_in = self._flatten_input(data_in)
        self._nRows = size
        self._nCols = 1
        for i in range(size):
            self._data[i] = data_in[i]
        self._repr = rep
        self._set_order(2)
        self._matrix_dim(size)
        return 0

    def set_data_4(self, data_in, nRows, nCols, rep):
        """Set 4th-order tensor data from array."""
        if rep < 0 or rep > 4:
            raise ValueError("CTensor: unsupported matrix representation!")
        data_in = self._flatten_input(data_in)
        self._nRows = nRows
        self._nCols = nCols
        n = nRows * nCols
        for i in range(n):
            self._data[i] = data_in[i]
        self._repr = rep
        self._set_order(4)
        self._matrix_dim(nRows)
        return 0

    def set_data_dev_vol(self, deviatoric, volumetric, linearized):
        """Set from deviatoric CTensor + volumetric scalar (linearized additive)."""
        # Copy deviatoric
        self._dim = deviatoric._dim
        self._order = deviatoric._order
        self._repr = deviatoric._repr
        self._nRows = deviatoric._nRows
        self._nCols = deviatoric._nCols
        n = self._nRows * self._nCols
        for i in range(n):
            self._data[i] = deviatoric._data[i]

        if linearized:
            # Fast path for 3D 2nd-order: I = [1,1,1,0,0,0], just add vol to normals
            if self._order == 2 and self._dim == 3:
                self._data[0] += volumetric
                self._data[1] += volumetric
                self._data[2] += volumetric
            elif self._order == 2:
                scaled_I = CTensor.Constants.I(self._dim, self._repr) * volumetric
                self.add_tensor(1.0, scaled_I, 1.0)
            else:
                scaled_IIvol = CTensor.Constants.IIvol(self._dim, self._repr) * volumetric
                self.add_tensor(1.0, scaled_IIvol, 1.0)
        else:
            # multiplicative composition
            if self._order == 2:
                scaled_I = CTensor.Constants.I(self._dim, self._repr) * volumetric
                result = self.dot(scaled_I, True)
                n2 = result._nRows * result._nCols
                for i in range(n2):
                    self._data[i] = result._data[i]
                self._nRows = result._nRows
                self._nCols = result._nCols
                self._order = result._order
                self._repr = result._repr
                self._dim = result._dim
            else:
                scaled_I = CTensor.Constants.I(self._dim, self._repr) * volumetric
                result = scaled_I.__xor__(self)
                n2 = result._nRows * result._nCols
                for i in range(n2):
                    self._data[i] = result._data[i]
                self._nRows = result._nRows
                self._nCols = result._nCols
                self._order = result._order
                self._repr = result._repr
                self._dim = result._dim

        # order sanity
        if self._nCols > 1 and self._order == 2:
            self._order = 4
        if self._nCols == 1 and self._order == 4:
            self._order = 2
        return 0

    def set_data_dev_vol_tensor(self, deviatoric, volumetric, linearized):
        """Set from deviatoric CTensor + volumetric CTensor (linearized additive)."""
        self._dim = deviatoric._dim
        self._order = deviatoric._order
        self._repr = deviatoric._repr
        self._nRows = deviatoric._nRows
        self._nCols = deviatoric._nCols
        n = self._nRows * self._nCols
        for i in range(n):
            self._data[i] = deviatoric._data[i]

        if linearized:
            self.add_tensor(1.0, volumetric, 1.0)
        else:
            if self._order == 2:
                self.add_dot_product(1.0, volumetric, 1.0, True)
            else:
                self.add_double_dot_product(1.0, volumetric, 1.0, True)

        if self._nCols > 1 and self._order == 2:
            self._order = 4
        if self._nCols == 1 and self._order == 4:
            self._order = 2
        return 0

    def resize(self, nRows, nCols=None):
        """Resize tensor (zeroes data)."""
        if nCols is None:
            # 2nd-order
            self._nRows = nRows
            self._nCols = 1
            for i in range(nRows):
                self._data[i] = 0.0
            self._matrix_dim(nRows)
        else:
            # 4th-order
            self._nRows = nRows
            self._nCols = nCols
            n = nRows * nCols
            for i in range(n):
                self._data[i] = 0.0
            self._matrix_dim(nRows)
        return 0

    # ------------------------------------------------------------------
    # Factor tables (static)
    # ------------------------------------------------------------------

    @staticmethod
    def _rep_factors_2(size, dim, rep):
        """Per-component scaling factors for 2nd-order representation."""
        # Fast path for common cases
        if size == 6:
            if rep == 0:
                return _FACTORS_ONES_6
            elif rep == 1:
                return _FACTORS_COV_6
            elif rep == 2:
                return _FACTORS_CONTR_6
            return _FACTORS_ONES_6
        if size == 3:
            if rep == 0:
                return _FACTORS_ONES_3
            elif rep == 1:
                return _FACTORS_COV_3
            elif rep == 2:
                return _FACTORS_CONTR_3
            return _FACTORS_ONES_3
        # General path
        if rep == 0:
            return (1.0,) * size
        elif rep == 1:
            return tuple(1.0 if i < dim else 0.5 for i in range(size))
        elif rep == 2:
            return tuple(1.0 if i < dim else 2.0 for i in range(size))
        return (1.0,) * size

    @staticmethod
    def _rep_factors_4(size, dim, rep):
        """Row and column factors for 4th-order representation."""
        rep_map = {0: (0, 0), 1: (1, 1), 2: (2, 2), 3: (1, 2), 4: (2, 1)}
        row_rep, col_rep = rep_map.get(rep, (0, 0))
        return (CTensor._rep_factors_2(size, dim, row_rep),
                CTensor._rep_factors_2(size, dim, col_rep))

    # ------------------------------------------------------------------
    # Norms and normalization
    # ------------------------------------------------------------------

    def normalize(self):
        """Normalize tensor by its representation-aware norm. Returns 0 on success."""
        self_norm = self.norm()
        if self_norm > _SMALL_VALUE:
            n = self._nRows * self._nCols
            inv_norm = 1.0 / self_norm
            for i in range(n):
                self._data[i] *= inv_norm
            return 0
        return -1

    def det(self):
        """Determinant (not implemented)."""
        raise NotImplementedError("CTensor: det() not implemented!")

    def norm(self):
        """Representation-aware norm."""
        result = 0.0
        if self._order == 2:
            f = self._rep_factors_2(self._nRows, self._dim, self._repr)
            if self._nRows == 6:
                d = self._data
                result = (f[0]*d[0]*d[0] + f[1]*d[1]*d[1] + f[2]*d[2]*d[2]
                        + f[3]*d[3]*d[3] + f[4]*d[4]*d[4] + f[5]*d[5]*d[5])
            else:
                for i in range(self._nRows):
                    result += f[i] * self._data[i] * self._data[i]
        elif self._order == 4:
            rowF, colF = self._rep_factors_4(self._nRows, self._dim, self._repr)
            for i in range(self._nRows):
                rf = rowF[i]
                for j in range(self._nCols):
                    val = self._data[i * self._nCols + j]
                    result += rf * colF[j] * val * val
        return _sqrt(result)

    def J2(self):
        """Second invariant of deviatoric tensor: 0.5 * (self % self)."""
        return 0.5 * self.__mod__(self)

    def octahedral(self):
        """Octahedral shear: sqrt(2/3 * J2)."""
        return _sqrt(2.0 / 3.0 * self.J2())

    # ------------------------------------------------------------------
    # Double-dot product: 2nd:2nd -> scalar  (operator %)
    # ------------------------------------------------------------------

    def __mod__(self, other):
        """Double dot product between two 2nd-order CTensors -> scalar."""
        if not isinstance(other, CTensor):
            return NotImplemented
        if self._dim != other._dim:
            raise ValueError("CTensor %: dimensions do not match!")
        if self._order != 2 or other._order != 2:
            raise ValueError("CTensor %: both must be 2nd-order! Use ^ for 4th-order.")

        d = self._data
        e = other._data

        # 3D fast paths (inline factors, no allocation)
        if self._nRows == 6:
            if self._repr == other._repr:
                if self._repr == 2:  # CONTR: factors [1,1,1,2,2,2]
                    return (d[0]*e[0] + d[1]*e[1] + d[2]*e[2]
                            + 2.0*(d[3]*e[3] + d[4]*e[4] + d[5]*e[5]))
                elif self._repr == 1:  # COV: factors [1,1,1,0.5,0.5,0.5]
                    return (d[0]*e[0] + d[1]*e[1] + d[2]*e[2]
                            + 0.5*(d[3]*e[3] + d[4]*e[4] + d[5]*e[5]))
                elif self._repr == 0:  # FULL: factors all 1.0
                    return (d[0]*e[0] + d[1]*e[1] + d[2]*e[2]
                            + d[3]*e[3] + d[4]*e[4] + d[5]*e[5])
            # Cov+Contr conjugate pair: factors all 1.0
            elif ((self._repr == 1 and other._repr == 2) or
                  (self._repr == 2 and other._repr == 1)):
                return (d[0]*e[0] + d[1]*e[1] + d[2]*e[2]
                        + d[3]*e[3] + d[4]*e[4] + d[5]*e[5])

        # General path
        if self._repr == other._repr:
            f = self._rep_factors_2(self._nRows, self._dim, self._repr)
        elif (self._repr == 1 and other._repr == 2) or (self._repr == 2 and other._repr == 1):
            f = (1.0,) * self._nRows
        else:
            raise ValueError("CTensor %: unsupported representation combination!")

        result = 0.0
        for i in range(self._nRows):
            result += f[i] * d[i] * e[i]
        return result

    # ------------------------------------------------------------------
    # Mixed-order double-dot: 2x4, 4x2, 4x4 -> result  (operator ^)
    # ------------------------------------------------------------------

    def __xor__(self, other):
        """Double dot between [2-4], [4-2], or [4-4] CTensors."""
        if not isinstance(other, CTensor):
            return NotImplemented
        if self._dim != other._dim:
            raise ValueError("CTensor ^: dimensions do not match!")

        result = CTensor()

        if self._order == 2 and other._order == 4:
            # 2nd ^ 4th -> 2nd
            if self._nRows != other._nRows:
                raise ValueError("CTensor ^: 2nd-4th size mismatch!")
            rep = self._resolve_rep_2x4(self._repr, other._repr)
            result._order = 2
            result._dim = self._dim
            result._repr = rep
            result._nRows = other._nCols
            result._nCols = 1
            for i in range(result._nRows):
                s = 0.0
                for j in range(self._nRows):
                    s += self._data[j] * other._data[j * other._nCols + i]
                result._data[i] = s

        elif self._order == 4 and other._order == 2:
            # 4th ^ 2nd -> 2nd
            if self._nCols != other._nRows:
                raise ValueError("CTensor ^: 4th-2nd size mismatch!")
            rep = self._resolve_rep_4x2(self._repr, other._repr)
            result._order = 2
            result._dim = self._dim
            result._repr = rep
            result._nRows = self._nRows
            result._nCols = 1

            sd = self._data
            od = other._data

            # 3D fast path
            if self._nRows == 6 and self._nCols == 6:
                result._data[0] = sd[0]*od[0] + sd[1]*od[1] + sd[2]*od[2] + sd[3]*od[3] + sd[4]*od[4] + sd[5]*od[5]
                result._data[1] = sd[6]*od[0] + sd[7]*od[1] + sd[8]*od[2] + sd[9]*od[3] + sd[10]*od[4] + sd[11]*od[5]
                result._data[2] = sd[12]*od[0] + sd[13]*od[1] + sd[14]*od[2] + sd[15]*od[3] + sd[16]*od[4] + sd[17]*od[5]
                result._data[3] = sd[18]*od[0] + sd[19]*od[1] + sd[20]*od[2] + sd[21]*od[3] + sd[22]*od[4] + sd[23]*od[5]
                result._data[4] = sd[24]*od[0] + sd[25]*od[1] + sd[26]*od[2] + sd[27]*od[3] + sd[28]*od[4] + sd[29]*od[5]
                result._data[5] = sd[30]*od[0] + sd[31]*od[1] + sd[32]*od[2] + sd[33]*od[3] + sd[34]*od[4] + sd[35]*od[5]
            else:
                for i in range(self._nRows):
                    s = 0.0
                    for j in range(self._nCols):
                        s += self._data[i * self._nCols + j] * other._data[j]
                    result._data[i] = s

        elif self._order == 4 and other._order == 4:
            # 4th ^ 4th -> 4th
            if self._nCols != other._nRows:
                raise ValueError("CTensor ^: 4th-4th size mismatch!")
            rep = self._resolve_rep_4x4(self._repr, other._repr)
            result._order = 4
            result._dim = self._dim
            result._repr = rep
            result._nRows = self._nRows
            result._nCols = other._nCols
            for i in range(self._nRows):
                for j in range(other._nCols):
                    s = 0.0
                    for k in range(self._nCols):
                        s += self._data[i * self._nCols + k] * other._data[k * other._nCols + j]
                    result._data[i * result._nCols + j] = s
        else:
            raise ValueError("CTensor ^: unsupported order combination!")

        return result

    @staticmethod
    def _resolve_rep_2x4(rep_a, rep_b):
        """Resolve output representation for 2nd^4th double dot."""
        pairs = {
            (0, 0): 0, (1, 2): 2, (2, 1): 1,
            (1, 3): 1, (2, 4): 2,
        }
        key = (rep_a, rep_b)
        if key not in pairs:
            raise ValueError("CTensor ^: unsupported 2x4 representation combination!")
        return pairs[key]

    @staticmethod
    def _resolve_rep_4x2(rep_a, rep_b):
        """Resolve output representation for 4th^2nd double dot."""
        pairs = {
            (0, 0): 0, (1, 2): 1, (2, 1): 2,
            (4, 2): 2, (3, 1): 1,
        }
        key = (rep_a, rep_b)
        if key not in pairs:
            raise ValueError("CTensor ^: unsupported 4x2 representation combination!")
        return pairs[key]

    @staticmethod
    def _resolve_rep_4x4(rep_a, rep_b):
        """Resolve output representation for 4th^4th double dot."""
        pairs = {
            (0, 0): 0, (1, 2): 3, (2, 1): 4,
            (4, 2): 2, (3, 1): 1,
            (4, 4): 4, (3, 3): 3,
        }
        key = (rep_a, rep_b)
        if key not in pairs:
            raise ValueError("CTensor ^: unsupported 4x4 representation combination!")
        return pairs[key]

    # ------------------------------------------------------------------
    # Dyadic product: 2nd * 2nd -> 4th  (operator * with CTensor)
    # ------------------------------------------------------------------

    def _dyadic(self, other):
        """Dyadic product between two 2nd-order CTensors -> 4th-order."""
        if self._dim != other._dim:
            raise ValueError("CTensor dyadic: dimensions do not match!")
        if self._order != 2 or other._order != 2:
            raise ValueError("CTensor dyadic: both must be 2nd-order!")

        # output size
        if self._repr == 0 and other._repr == 0:
            m = 9 if self._dim == 3 else 4
        else:
            m = 6 if self._dim == 3 else 3

        # output rep
        rep_pairs = {
            (0, 0): 0, (1, 1): 1, (2, 2): 2,
            (1, 2): 3, (2, 1): 4,
        }
        key = (self._repr, other._repr)
        if key not in rep_pairs:
            raise ValueError("CTensor dyadic: unsupported representation combination!")
        rep = rep_pairs[key]

        result = CTensor(m, m, rep)
        for i in range(self._nRows):
            for j in range(other._nRows):
                result._data[i * m + j] = self._data[i] * other._data[j]
        return result

    # ------------------------------------------------------------------
    # Single dot product (stub)
    # ------------------------------------------------------------------

    def dot(self, other, pre=False):
        """Single dot product between two CTensors (not fully implemented)."""
        raise NotImplementedError("CTensor: dot() not implemented!")

    # ------------------------------------------------------------------
    # Invert (stub)
    # ------------------------------------------------------------------

    def invert(self):
        """Invert the tensor (not implemented)."""
        raise NotImplementedError("CTensor: invert() not implemented!")

    # ------------------------------------------------------------------
    # Fused operations
    # ------------------------------------------------------------------

    def add_tensor(self, factThis, other, factOther):
        """self = factThis*self + factOther*other (element-wise)."""
        if self._nRows != other._nRows or self._nCols != other._nCols:
            raise ValueError("CTensor add_tensor: dimension mismatch!")
        n = self._nRows * self._nCols
        for i in range(n):
            self._data[i] = factThis * self._data[i] + factOther * other._data[i]
        return 0

    def add_tensor_transpose(self, factThis, other, factOther):
        """self = factThis*self + factOther*other^T (4th-order only)."""
        if self._order != 4 or other._order != 4:
            raise ValueError("CTensor add_tensor_transpose: order must be 4!")
        if self._nRows != other._nRows or self._nCols != other._nCols:
            raise ValueError("CTensor add_tensor_transpose: dimension mismatch!")
        for i in range(self._nRows):
            for j in range(self._nCols):
                idx = i * self._nCols + j
                self._data[idx] = (factThis * self._data[idx]
                                   + factOther * other._data[j * other._nCols + i])
        return 0

    def add_dot_product(self, factThis, other, factOther, premultiply):
        """Fused dot product (not implemented)."""
        raise NotImplementedError("CTensor: add_dot_product() not implemented!")

    def add_double_dot_product(self, factThis, other, factOther, premultiply):
        """Fused double dot: self = factThis*self + factOther*(other^self) or (self^other)."""
        if premultiply:
            # self = factThis*self + factOther*(other ^ self)
            if self._order == 2 and other._order == 4:
                # 4th ^ 2nd -> 2nd
                if other._nCols != self._nRows:
                    raise ValueError("CTensor add_double_dot_product: dimension mismatch!")
                temp = [0.0] * self._nRows
                if other._nRows == 6 and other._nCols == 6 and self._nRows == 6:
                    od = other._data
                    sd = self._data
                    temp[0] = od[0]*sd[0] + od[1]*sd[1] + od[2]*sd[2] + od[3]*sd[3] + od[4]*sd[4] + od[5]*sd[5]
                    temp[1] = od[6]*sd[0] + od[7]*sd[1] + od[8]*sd[2] + od[9]*sd[3] + od[10]*sd[4] + od[11]*sd[5]
                    temp[2] = od[12]*sd[0] + od[13]*sd[1] + od[14]*sd[2] + od[15]*sd[3] + od[16]*sd[4] + od[17]*sd[5]
                    temp[3] = od[18]*sd[0] + od[19]*sd[1] + od[20]*sd[2] + od[21]*sd[3] + od[22]*sd[4] + od[23]*sd[5]
                    temp[4] = od[24]*sd[0] + od[25]*sd[1] + od[26]*sd[2] + od[27]*sd[3] + od[28]*sd[4] + od[29]*sd[5]
                    temp[5] = od[30]*sd[0] + od[31]*sd[1] + od[32]*sd[2] + od[33]*sd[3] + od[34]*sd[4] + od[35]*sd[5]
                else:
                    for i in range(other._nRows):
                        s = 0.0
                        for j in range(other._nCols):
                            s += other._data[i * other._nCols + j] * self._data[j]
                        temp[i] = s
                for i in range(self._nRows):
                    self._data[i] = factThis * self._data[i] + factOther * temp[i]
                return 0

            elif self._order == 4 and other._order == 4:
                # 4th ^ 4th -> 4th
                if other._nCols != self._nRows or other._nRows != self._nRows:
                    raise ValueError("CTensor add_double_dot_product: dimension mismatch!")
                n = self._nRows * self._nCols
                temp = [0.0] * n
                for i in range(self._nRows):
                    for j in range(self._nCols):
                        s = 0.0
                        for k in range(other._nCols):
                            s += other._data[i * other._nCols + k] * self._data[k * self._nCols + j]
                        temp[i * self._nCols + j] = s
                for i in range(n):
                    self._data[i] = factThis * self._data[i] + factOther * temp[i]
                return 0
        else:
            # self = factThis*self + factOther*(self ^ other)
            if self._order == 2 and other._order == 4:
                # 2nd ^ 4th -> 2nd
                if self._nRows != other._nRows or other._nCols != self._nRows:
                    raise ValueError("CTensor add_double_dot_product: dimension mismatch!")
                temp = [0.0] * self._nRows
                for i in range(self._nRows):
                    s = 0.0
                    for j in range(self._nRows):
                        s += self._data[j] * other._data[j * other._nCols + i]
                    temp[i] = s
                for i in range(self._nRows):
                    self._data[i] = factThis * self._data[i] + factOther * temp[i]
                return 0

            elif self._order == 4 and other._order == 4:
                # 4th ^ 4th -> 4th
                if self._nCols != other._nRows or other._nCols != self._nCols:
                    raise ValueError("CTensor add_double_dot_product: dimension mismatch!")
                n = self._nRows * self._nCols
                temp = [0.0] * n
                for i in range(self._nRows):
                    for j in range(self._nCols):
                        s = 0.0
                        for k in range(self._nCols):
                            s += self._data[i * self._nCols + k] * other._data[k * self._nCols + j]
                        temp[i * self._nCols + j] = s
                for i in range(n):
                    self._data[i] = factThis * self._data[i] + factOther * temp[i]
                return 0

            elif self._order == 4 and other._order == 2:
                raise ValueError("CTensor add_double_dot_product: 4th^2nd changes order, use ^ instead!")

        raise ValueError("CTensor add_double_dot_product: unsupported order combination!")

    def add_tensor_product(self, factThis, other, factOther, premultiply=True):
        """Fused self-dyadic accumulation: self_4th = factThis*self + factOther*(other (x) other)."""
        if self._order != 4 or other._order != 2:
            raise ValueError("CTensor add_tensor_product: self must be 4th, other must be 2nd!")
        if self._nRows != other._nRows or self._nCols != other._nRows:
            raise ValueError("CTensor add_tensor_product: dimension mismatch!")
        for i in range(other._nRows):
            for j in range(other._nRows):
                idx = i * self._nCols + j
                self._data[idx] = factThis * self._data[idx] + factOther * other._data[i] * other._data[j]
        return 0

    # ------------------------------------------------------------------
    # Scalar-tensor operators
    # ------------------------------------------------------------------

    def __add__(self, other):
        if isinstance(other, CTensor):
            return self._tensor_add(other)
        # scalar
        other = float(other)
        if other == 0.0:
            return self.copy()
        result = self.copy()
        n = result._nRows * result._nCols
        for i in range(n):
            result._data[i] += other
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, CTensor):
            return self._tensor_sub(other)
        other = float(other)
        if other == 0.0:
            return self.copy()
        result = self.copy()
        n = result._nRows * result._nCols
        for i in range(n):
            result._data[i] -= other
        return result

    def __rsub__(self, other):
        other = float(other)
        result = self.copy()
        n = result._nRows * result._nCols
        for i in range(n):
            result._data[i] = other - result._data[i]
        return result

    def __mul__(self, other):
        if isinstance(other, CTensor):
            return self._dyadic(other)
        other = float(other)
        result = self.copy()
        if other == 0.0:
            result.zero()
            return result
        n = result._nRows * result._nCols
        for i in range(n):
            result._data[i] *= other
        return result

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = float(other)
        if other == 0.0:
            raise ZeroDivisionError("CTensor: divide by zero!")
        result = self.copy()
        inv = 1.0 / other
        n = result._nRows * result._nCols
        for i in range(n):
            result._data[i] *= inv
        return result

    def __iadd__(self, other):
        if isinstance(other, CTensor):
            if other._repr != self._repr:
                raise ValueError("CTensor +=: representations do not match!")
            if self._nCols != other._nCols or self._nRows != other._nRows:
                raise ValueError("CTensor +=: dimensions do not match!")
            n = self._nRows * self._nCols
            for i in range(n):
                self._data[i] += other._data[i]
            return self
        other = float(other)
        if other == 0.0:
            return self
        n = self._nRows * self._nCols
        for i in range(n):
            self._data[i] += other
        return self

    def __isub__(self, other):
        if isinstance(other, CTensor):
            if other._repr != self._repr:
                raise ValueError("CTensor -=: representations do not match!")
            if self._nCols != other._nCols or self._nRows != other._nRows:
                raise ValueError("CTensor -=: dimensions do not match!")
            n = self._nRows * self._nCols
            for i in range(n):
                self._data[i] -= other._data[i]
            return self
        other = float(other)
        if other == 0.0:
            return self
        n = self._nRows * self._nCols
        for i in range(n):
            self._data[i] -= other
        return self

    def __imul__(self, other):
        other = float(other)
        if other == 1.0:
            return self
        if other == 0.0:
            self.zero()
            return self
        n = self._nRows * self._nCols
        for i in range(n):
            self._data[i] *= other
        return self

    def __itruediv__(self, other):
        other = float(other)
        if other == 0.0:
            raise ZeroDivisionError("CTensor: divide by zero!")
        inv = 1.0 / other
        n = self._nRows * self._nCols
        for i in range(n):
            self._data[i] *= inv
        return self

    def __neg__(self):
        result = self.copy()
        n = result._nRows * result._nCols
        for i in range(n):
            result._data[i] = -result._data[i]
        return result

    # ------------------------------------------------------------------
    # Tensor-tensor operators
    # ------------------------------------------------------------------

    def _tensor_add(self, other):
        if other._repr != self._repr:
            raise ValueError("CTensor +: representations do not match!")
        if self._nCols != other._nCols or self._nRows != other._nRows:
            raise ValueError("CTensor +: dimensions do not match!")
        result = self.copy()
        n = self._nRows * self._nCols
        for i in range(n):
            result._data[i] += other._data[i]
        return result

    def _tensor_sub(self, other):
        if other._repr != self._repr:
            raise ValueError("CTensor -: representations do not match!")
        if self._nCols != other._nCols or self._nRows != other._nRows:
            raise ValueError("CTensor -: dimensions do not match!")
        result = self.copy()
        n = self._nRows * self._nCols
        for i in range(n):
            result._data[i] -= other._data[i]
        return result

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def __eq__(self, other):
        if not isinstance(other, CTensor):
            return NotImplemented
        if (self._order != other._order or self._dim != other._dim
                or self._nRows != other._nRows or self._nCols != other._nCols
                or self._repr != other._repr):
            return False
        n = self._nRows * self._nCols
        return self._data[:n] == other._data[:n]

    def __ne__(self, other):
        eq = self.__eq__(other)
        if eq is NotImplemented:
            return NotImplemented
        return not eq

    # ------------------------------------------------------------------
    # Submatrix extraction
    # ------------------------------------------------------------------

    def submatrix(self, rows, cols):
        """Extract a sub-tensor by row and column index lists."""
        nRows = len(rows)
        nCols = len(cols)
        result = CTensor(nRows, nCols, self._repr)
        for i in range(nRows):
            for j in range(nCols):
                result._data[i * nCols + j] = self._data[rows[i] * self._nCols + cols[j]]
        return result

    # ------------------------------------------------------------------
    # Representation switching (private)
    # ------------------------------------------------------------------

    def _to_cov(self):
        if self._repr == 0:
            raise NotImplementedError("CTensor: Full-to-Cov not implemented!")
        elif self._repr == 1:
            return 0
        elif self._repr == 2:  # Contr -> Cov
            for i in range(self._nRows):
                for j in range(self._nCols):
                    idx = i * self._nCols + j
                    if i >= self._dim:
                        self._data[idx] *= 2.0
                    if j >= self._dim:
                        self._data[idx] *= 2.0
            self._repr = 1
            return 0
        elif self._repr == 3:  # CovContr -> Cov
            for i in range(self._nRows):
                for j in range(self._nCols):
                    if j >= self._dim:
                        self._data[i * self._nCols + j] *= 2.0
            self._repr = 1
            return 0
        elif self._repr == 4:  # ContrCov -> Cov
            for i in range(self._nRows):
                for j in range(self._nCols):
                    if i >= self._dim:
                        self._data[i * self._nCols + j] *= 2.0
            self._repr = 1
            return 0
        return -1

    def _to_full(self):
        raise NotImplementedError("CTensor: toFull() not implemented!")

    def _to_contr(self):
        if self._repr == 0:
            raise NotImplementedError("CTensor: Full-to-Contr not implemented!")
        elif self._repr == 1:  # Cov -> Contr
            for i in range(self._nRows):
                for j in range(self._nCols):
                    idx = i * self._nCols + j
                    if i >= self._dim:
                        self._data[idx] *= 0.5
                    if j >= self._dim:
                        self._data[idx] *= 0.5
            self._repr = 2
            return 0
        elif self._repr == 2:
            return 0
        elif self._repr == 3:  # CovContr -> Contr
            for i in range(self._nRows):
                for j in range(self._nCols):
                    if i >= self._dim:
                        self._data[i * self._nCols + j] *= 0.5
            self._repr = 2
            return 0
        elif self._repr == 4:  # ContrCov -> Contr
            for i in range(self._nRows):
                for j in range(self._nCols):
                    if j >= self._dim:
                        self._data[i * self._nCols + j] *= 0.5
            self._repr = 2
            return 0
        return -1

    def _to_cov_contr(self):
        if self._order != 4:
            raise ValueError("CTensor: cannot convert 2nd-order to mixed representation!")
        if self._repr == 0:
            raise NotImplementedError("CTensor: Full-to-CovContr not implemented!")
        elif self._repr == 1:  # Cov -> CovContr
            for i in range(self._nRows):
                for j in range(self._nCols):
                    if j >= self._dim:
                        self._data[i * self._nCols + j] *= 0.5
            self._repr = 3
            return 0
        elif self._repr == 2:  # Contr -> CovContr
            for i in range(self._nRows):
                for j in range(self._nCols):
                    if i >= self._dim:
                        self._data[i * self._nCols + j] *= 2.0
            self._repr = 3
            return 0
        elif self._repr == 3:
            return 0
        elif self._repr == 4:  # ContrCov -> CovContr
            for i in range(self._nRows):
                for j in range(self._nCols):
                    idx = i * self._nCols + j
                    if i >= self._dim:
                        self._data[idx] *= 2.0
                    if j >= self._dim:
                        self._data[idx] *= 0.5
            self._repr = 3
            return 0
        return -1

    def _to_contr_cov(self):
        if self._order != 4:
            raise ValueError("CTensor: cannot convert 2nd-order to mixed representation!")
        if self._repr == 0:
            raise NotImplementedError("CTensor: Full-to-ContrCov not implemented!")
        elif self._repr == 1:  # Cov -> ContrCov
            for i in range(self._nRows):
                for j in range(self._nCols):
                    if i >= self._dim:
                        self._data[i * self._nCols + j] *= 0.5
            self._repr = 4
            return 0
        elif self._repr == 2:  # Contr -> ContrCov
            for i in range(self._nRows):
                for j in range(self._nCols):
                    if j >= self._dim:
                        self._data[i * self._nCols + j] *= 2.0
            self._repr = 4
            return 0
        elif self._repr == 3:  # CovContr -> ContrCov
            for i in range(self._nRows):
                for j in range(self._nCols):
                    idx = i * self._nCols + j
                    if i >= self._dim:
                        self._data[idx] *= 0.5
                    if j >= self._dim:
                        self._data[idx] *= 2.0
            self._repr = 4
            return 0
        elif self._repr == 4:
            return 0
        return -1

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def __repr__(self):
        rep_name = self._REP_NAMES.get(self._repr, 'unknown')
        lines = [f"CTensor(order={self._order}, dim={self._dim}, rep={rep_name}, "
                 f"size={self._nRows}x{self._nCols})"]
        if self._order == 2:
            lines.append(str(self._data[:self._nRows]))
        elif self._order == 4 and self._nRows > 0:
            for i in range(self._nRows):
                row_start = i * self._nCols
                lines.append(str(self._data[row_start:row_start + self._nCols]))
        return '\n'.join(lines)

    # ==================================================================
    # Cached Constants (nested class)
    # ==================================================================

    class Constants:
        """Tensor-valued constants, cached by (name, dim, rep)."""
        _cache = {}

        @classmethod
        def I(cls, dim=3, rep=0):
            """2nd-order identity tensor (reps 0, 1, 2 only)."""
            if rep < 0 or rep > 2:
                raise ValueError("Constants.I: 2nd-order tensor has no mixed representation!")
            key = ('I', dim, rep)
            if key not in cls._cache:
                if rep == 0:
                    m = 9 if dim == 3 else 4
                else:
                    m = 6 if dim == 3 else 3
                t = CTensor(m, rep)
                for i in range(dim):
                    t[i] = 1.0
                cls._cache[key] = t
            return cls._cache[key].copy()

        @classmethod
        def IIsymm(cls, dim=3, rep=0):
            """4th-order symmetric identity operator."""
            if rep < 0 or rep > 4:
                raise ValueError("Constants.IIsymm: unsupported representation!")
            key = ('IIsymm', dim, rep)
            if key not in cls._cache:
                if rep == 0:
                    m = 9 if dim == 3 else 4
                    t = CTensor(m, m, rep)
                    for i in range(m):
                        # diagonal: 1.0 for normals, 0.5 for shears
                        t[i, i] = 1.0 - (1 if i >= dim else 0) * 0.5
                else:
                    m = 6 if dim == 3 else 3
                    t = CTensor(m, m, rep)
                    if rep == 1:    # Cov
                        for i in range(m):
                            t[i, i] = 1.0 + (1 if i >= dim else 0) * 1.0
                    elif rep == 2:  # Contr
                        for i in range(m):
                            t[i, i] = 1.0 - (1 if i >= dim else 0) * 0.5
                    else:           # CovContr(3) or ContrCov(4)
                        for i in range(m):
                            t[i, i] = 1.0
                cls._cache[key] = t
            return cls._cache[key].copy()

        @classmethod
        def IIvol(cls, dim=3, rep=0):
            """4th-order volumetric operator (I tensor I)."""
            if rep < 0 or rep > 4:
                raise ValueError("Constants.IIvol: unsupported representation!")
            key = ('IIvol', dim, rep)
            if key not in cls._cache:
                if rep == 0:
                    m = 9 if dim == 3 else 4
                else:
                    m = 6 if dim == 3 else 3
                t = CTensor(m, m, rep)
                if dim == 3:
                    for i in range(3):
                        for j in range(3):
                            t[i, j] = 1.0
                elif dim == 2:
                    t[0, 0] = 1.0; t[0, 1] = 1.0
                    t[1, 0] = 1.0; t[1, 1] = 1.0
                cls._cache[key] = t
            return cls._cache[key].copy()

        @classmethod
        def IIdev(cls, dim=3, rep=0):
            """4th-order deviatoric operator (IIsymm - 1/3*IIvol)."""
            if rep < 0 or rep > 4:
                raise ValueError("Constants.IIdev: unsupported representation!")
            key = ('IIdev', dim, rep)
            if key not in cls._cache:
                if rep == 0:
                    m = 9 if dim == 3 else 4
                    t = CTensor(m, m, rep)
                    if dim == 3:
                        t[0, 0] =  2.0/3.0; t[0, 1] = -1.0/3.0; t[0, 2] = -1.0/3.0
                        t[1, 0] = -1.0/3.0; t[1, 1] =  2.0/3.0; t[1, 2] = -1.0/3.0
                        t[2, 0] = -1.0/3.0; t[2, 1] = -1.0/3.0; t[2, 2] =  2.0/3.0
                        t[3, 3] = 0.5; t[4, 4] = 0.5; t[5, 5] = 0.5
                        t[6, 6] = 0.5; t[7, 7] = 0.5; t[8, 8] = 0.5
                    elif dim == 2:
                        t[0, 0] = 1.0/3.0;    t[0, 1] = -0.5/3.0
                        t[1, 0] = -0.5/3.0;   t[1, 1] = 1.0/3.0
                        t[2, 2] = 0.25; t[3, 3] = 0.25
                else:
                    m = 6 if dim == 3 else 3
                    t = CTensor(m, m, rep)
                    if rep == 1:  # Cov
                        if dim == 3:
                            t[0, 0] =  2.0/3.0; t[0, 1] = -1.0/3.0; t[0, 2] = -1.0/3.0
                            t[1, 0] = -1.0/3.0; t[1, 1] =  2.0/3.0; t[1, 2] = -1.0/3.0
                            t[2, 0] = -1.0/3.0; t[2, 1] = -1.0/3.0; t[2, 2] =  2.0/3.0
                            t[3, 3] = 2.0; t[4, 4] = 2.0; t[5, 5] = 2.0
                        elif dim == 2:
                            t[0, 0] = 1.0/3.0;    t[0, 1] = -0.5/3.0
                            t[1, 0] = -0.5/3.0;   t[1, 1] = 1.0/3.0
                            t[2, 2] = 1.0
                    elif rep == 2:  # Contr
                        if dim == 3:
                            t[0, 0] =  2.0/3.0; t[0, 1] = -1.0/3.0; t[0, 2] = -1.0/3.0
                            t[1, 0] = -1.0/3.0; t[1, 1] =  2.0/3.0; t[1, 2] = -1.0/3.0
                            t[2, 0] = -1.0/3.0; t[2, 1] = -1.0/3.0; t[2, 2] =  2.0/3.0
                            t[3, 3] = 0.5; t[4, 4] = 0.5; t[5, 5] = 0.5
                        elif dim == 2:
                            t[0, 0] = 1.0/3.0;    t[0, 1] = -0.5/3.0
                            t[1, 0] = -0.5/3.0;   t[1, 1] = 1.0/3.0
                            t[2, 2] = 0.25
                    else:  # CovContr(3) or ContrCov(4)
                        if dim == 3:
                            t[0, 0] =  2.0/3.0; t[0, 1] = -1.0/3.0; t[0, 2] = -1.0/3.0
                            t[1, 0] = -1.0/3.0; t[1, 1] =  2.0/3.0; t[1, 2] = -1.0/3.0
                            t[2, 0] = -1.0/3.0; t[2, 1] = -1.0/3.0; t[2, 2] =  2.0/3.0
                            t[3, 3] = 1.0; t[4, 4] = 1.0; t[5, 5] = 1.0
                        elif dim == 2:
                            t[0, 0] = 1.0/3.0;    t[0, 1] = -0.5/3.0
                            t[1, 0] = -0.5/3.0;   t[1, 1] = 1.0/3.0
                            t[2, 2] = 0.5
                cls._cache[key] = t
            return cls._cache[key].copy()
