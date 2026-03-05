##-----------------------------------------------------------------------##
#                                                                         #
#        #--oneFEM--#: One FEM software in a galaxy far far away          #
#                                                                         #
#                   Computational Mechanics 2022                          #
#                       University of Pavia                               #
#               Written by: Onur Deniz AKAN, IUSS Pavia                   #
#                         17 January 2022                                 #
#                                                                         #
##-----------------------------------------------------------------------##
#DOMAIN main object definition
#   Domain holds the model objects and the functions operate over them

from .node.main import Node
from .tseries.main import TSeries
from .pattern.main import Pattern
from .pattern.uniform_excitation import UniformExcitation
from .element.main import Element
from .material.main import Material
from .constraint.main import Constraint
from ..output.recorder.main import Recorder
from ..analysis.eigen.main import Eigen
from .._systools.data import Vector, Matrix
import numpy as np

class Domain(object):
    def __init__(self, nD=0):
        # Model objects
        self.__nodes = []       # Nodes (list)
        self.__patterns = []    # Load Pattern (list)
        self.__elements = []    # Elements (list)
        self.__constraints = [] # Constraints (list)
        self.__recorders = []   # Recorders (list)

        # Model variables
        self.__nDim = int(nD)   # model dimension 2D or 3D
        self.__nDof = int(0)    # default number of node dofs

        # Model statistics
        self.__nNds = 0     # total number of nodes
        self.__nDOF = 0     # total number of d.o.f.s
        self.__nElems = 0   # total number of elements
        self.__nCons = 0    # total number of constraints
        self.__nTS = 0      # total number of time series
        self.__nPtrns = 0   # total number of loading patterns
        self.__nMats = 0    # total number of root (user) materials
        self.__nRecs = 0    # total number of recorders

        # Node lookup dict for O(1) access during assembly
        self.__node_map = {}  # {nodeID: node_object}

        # Initialize global matrices and vectors
        self._K = None   # Global tangent stiffness matrix
        self._C = None   # Damping matrix
        self._M = None   # Mass matrix
        self._u = None   # Displacement vector
        self._v = None   # Velocity vector
        self._a = None   # Acceleration vector
        self._F = None   # Force vector

        # Rayleigh damping coefficients
        self.__alphaM = 0.0
        self.__betaK = 0.0

        # Eigen analysis results
        self.__eigenvalues = None   # numpy array of omega^2 values
        self.__eigenvectors = None  # numpy array (nFreeDOF x numModes)
        self.__eigenDOFs = None     # list of free DOF indices


    # Property accessors
    @property
    def nodes(self):
        return self.__nodes

    @property
    def elements(self):
        return self.__elements

    @property
    def patterns(self):
        return self.__patterns

    @property
    def constraints(self):
        return self.__constraints

    @property
    def recorders(self):
        return self.__recorders

    @property
    def nDOF(self):
        return self.__nDOF

    @property
    def nDim(self):
        return self.__nDim

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, value):
        self._K = value

    @property
    def F(self):
        return self._F

    @F.setter
    def F(self, value):
        self._F = value

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, value):
        self._u = value

    @property
    def M(self):
        return self._M

    @M.setter
    def M(self, value):
        self._M = value

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, value):
        self._C = value

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, value):
        self._v = value

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        self._a = value


    def setRayleighDamping(self, alphaM=0.0, betaK=0.0):
        self.__alphaM = alphaM
        self.__betaK = betaK


    def _domain(self):
        """Number DOFs sequentially and initialize elements."""
        self.__nNds = len(self.__nodes)
        if self.__nNds < 1:
            raise ValueError("Domain: No nodes in the domain!")

        self.__nElems = len(self.__elements)
        if self.__nElems < 1:
            raise ValueError("Domain: No elements in the domain!")

        # Build node lookup map
        self.__node_map = {node._ID: node for node in self.__nodes}

        # Sequential DOF numbering
        counter = 0
        for node in self.__nodes:
            nDOF = node.getNDOF()
            dofs = list(range(counter, counter + nDOF))
            node._setDOF(dofs)
            counter += nDOF
        self.__nDOF = counter

        # Call domain for all elements
        for element in self.__elements:
            element._domain()


    def _assemble(self, time=0.0):
        """Assemble global stiffness matrix, mass matrix, and force vector."""
        n = self.__nDOF
        self._K = Matrix(shape=[n, n])
        self._F = Vector(shape=n)

        # Preserve v and a across steps (only initialize if None or wrong size)
        if self._u is None or len(self._u) != n:
            self._u = Vector(shape=n)
        else:
            # Reset u for new solve (will be overwritten by algorithm)
            self._u = Vector(shape=n)

        # Check if any element has mass
        has_mass = any(e.getMass() is not None for e in self.__elements)
        if has_mass:
            self._M = Matrix(shape=[n, n])

        # Initialize v, a if needed (preserve across steps)
        if self._v is None or len(self._v) != n:
            self._v = Vector(shape=n)
        if self._a is None or len(self._a) != n:
            self._a = Vector(shape=n)

        # Assemble K and M for each element
        for element in self.__elements:
            ke = element.getStiffness()
            me = element.getMass()

            # Get global DOF indices for this element
            dofs = []
            for node in element.getNodes():
                node_dofs = node.getDOFs()
                if hasattr(node_dofs, 'tolist'):
                    dofs.extend(node_dofs.tolist())
                elif hasattr(node_dofs, 'data'):
                    dofs.extend(node_dofs.data.tolist())
                else:
                    dofs.extend(list(node_dofs))

            # Scatter element stiffness into global K
            for i_local, i_global in enumerate(dofs):
                for j_local, j_global in enumerate(dofs):
                    self._K[i_global, j_global] = self._K[i_global, j_global] + ke[i_local, j_local]

            # Scatter element mass into global M
            if me is not None:
                for i_local, i_global in enumerate(dofs):
                    for j_local, j_global in enumerate(dofs):
                        self._M[i_global, j_global] = self._M[i_global, j_global] + me[i_local, j_local]

        # Compute Rayleigh damping: C = alphaM * M + betaK * K
        if has_mass and (self.__alphaM != 0.0 or self.__betaK != 0.0):
            K_data = np.asarray(self._K)
            M_data = np.asarray(self._M)
            C_data = self.__alphaM * M_data + self.__betaK * K_data
            self._C = Matrix(init=C_data)
        elif has_mass:
            self._C = Matrix(shape=[n, n])
        else:
            self._C = None

        # Assemble F from patterns (nodal loads)
        for pattern in self.__patterns:
            nodal_loads = pattern.getNodalLoads(time)
            for nodeID, forces in nodal_loads.items():
                # Find the node by ID via dict lookup
                node = self.__node_map.get(nodeID)
                if node is None:
                    continue
                node_dofs = node.getDOFs()
                if hasattr(node_dofs, 'tolist'):
                    dof_list = node_dofs.tolist()
                elif hasattr(node_dofs, 'data'):
                    dof_list = node_dofs.data.tolist()
                else:
                    dof_list = list(node_dofs)
                for k, dof_idx in enumerate(dof_list):
                    if k < len(forces):
                        self._F[dof_idx] = self._F[dof_idx] + forces[k]

        # Apply UniformExcitation: F_eq = -M * r * a_g(t)
        # r is the influence vector (1.0 at the excitation DOF for each node)
        if has_mass:
            for pattern in self.__patterns:
                if isinstance(pattern, UniformExcitation):
                    a_g = pattern.getAcceleration(time)
                    if a_g != 0.0:
                        direction = pattern.direction  # 1-based
                        M_data = np.asarray(self._M)
                        F_data = np.asarray(self._F)
                        # Build influence vector r
                        r = np.zeros(n)
                        for node in self.__nodes:
                            node_dofs = node.getDOFs()
                            if hasattr(node_dofs, 'tolist'):
                                dof_list = node_dofs.tolist()
                            elif hasattr(node_dofs, 'data'):
                                dof_list = node_dofs.data.tolist()
                            else:
                                dof_list = list(node_dofs)
                            # direction is 1-based; DOF index within node
                            if direction <= len(dof_list):
                                r[dof_list[direction - 1]] = 1.0
                        # F_eq = -M * r * a_g
                        F_data -= a_g * M_data.dot(r)
                        self._F = Vector(list(F_data))

        # Apply imposed displacements from fixed nodes
        for node in self.__nodes:
            node_dofs = node.getDOFs()
            if hasattr(node_dofs, 'tolist'):
                dof_list = node_dofs.tolist()
            elif hasattr(node_dofs, 'data'):
                dof_list = node_dofs.data.tolist()
            else:
                dof_list = list(node_dofs)

            fix = node._fix
            for k, dof_idx in enumerate(dof_list):
                if fix[k]:
                    self._u[dof_idx] = 0.0  # imposed displacement = 0 for fixed DOFs


    def _commit(self, force, disp, vel=None, accel=None):
        """Update and commit all nodes with solved force and displacement vectors."""
        for node in self.__nodes:
            node_dofs = node.getDOFs()
            if hasattr(node_dofs, 'tolist'):
                dof_list = node_dofs.tolist()
            elif hasattr(node_dofs, 'data'):
                dof_list = node_dofs.data.tolist()
            else:
                dof_list = list(node_dofs)

            f_slice = Vector([force[d] for d in dof_list])
            u_slice = Vector([disp[d] for d in dof_list])

            if vel is not None and accel is not None:
                v_slice = Vector([vel[d] for d in dof_list])
                a_slice = Vector([accel[d] for d in dof_list])
                node._update(f_slice, u_slice, v_slice, a_slice)
            else:
                node._update(f_slice, u_slice)

            node._commitState()


    def getInternalForce(self):
        """Assemble global internal force vector from element force vectors."""
        n = self.__nDOF
        F_int = np.zeros(n)
        for element in self.__elements:
            fe = element.getForce()
            dofs = []
            for node in element.getNodes():
                node_dofs = node.getDOFs()
                if hasattr(node_dofs, 'tolist'):
                    dofs.extend(node_dofs.tolist())
                elif hasattr(node_dofs, 'data'):
                    dofs.extend(node_dofs.data.tolist())
                else:
                    dofs.extend(list(node_dofs))
            for i_local, i_global in enumerate(dofs):
                F_int[i_global] += float(fe[i_local])
        return F_int

    def getCommittedDisp(self):
        """Build global displacement vector from committed node displacements."""
        n = self.__nDOF
        u = np.zeros(n)
        for node in self.__nodes:
            node_dofs = node.getDOFs()
            if hasattr(node_dofs, 'tolist'):
                dof_list = node_dofs.tolist()
            elif hasattr(node_dofs, 'data'):
                dof_list = node_dofs.data.tolist()
            else:
                dof_list = list(node_dofs)
            u_commit = node._getCommitDisp()
            for k, dof_idx in enumerate(dof_list):
                u[dof_idx] = float(u_commit[k])
        return u

    def _record(self, time=None):
        """Call record() on all recorders.

        Parameters
        ----------
        time : float, optional
            Current simulation time passed to each recorder.
        """
        for recorder in self.__recorders:
            recorder.record(time)

    def _record_eigen(self):
        """Call record_eigen() on recorders that support it (e.g. ModeShapeRecorder)."""
        for recorder in self.__recorders:
            if hasattr(recorder, 'record_eigen'):
                recorder.record_eigen()


    def _revert(self):
        pass


    def _update(self):
        pass


    # Domain API methods
    def add(self, *objs):
        for obj in objs:
            if isinstance(obj, Element):
                self.__elements.append(obj)
            elif isinstance(obj, Node):
                self.__nodes.append(obj)
                self.__node_map[obj._ID] = obj
            elif isinstance(obj, Pattern):
                self.__patterns.append(obj)
            elif isinstance(obj, Constraint):
                self.__constraints.append(obj)
            elif isinstance(obj, Recorder):
                self.__recorders.append(obj)
            else:
                print(f"Domain: Unknown object! {obj}")


    def remove(self, obj):
        if isinstance(obj, Element):
            if obj in self.__elements:
                self.__elements.remove(obj)
            else:
                print("Domain: Element not found!")

        elif isinstance(obj, Node):
            if obj in self.__nodes:
                self.__nodes.remove(obj)
                self.__node_map.pop(obj._ID, None)
            else:
                print("Domain: Node not found!")

        elif isinstance(obj, Pattern):
            if obj in self.__patterns:
                self.__patterns.remove(obj)
            else:
                print("Domain: Pattern not found!")

        elif isinstance(obj, Constraint):
            if obj in self.__constraints:
                self.__constraints.remove(obj)
            else:
                print("Domain: Constraint not found!")

        elif isinstance(obj, Recorder):
            if obj in self.__recorders:
                self.__recorders.remove(obj)
            else:
                print("Domain: Recorder not found!")

        else:
            print("Domain: Unknown object!")


    # ---- Eigen analysis ----

    def eigen(self, numModes, solver='genBandArpack'):
        """Solve the generalized eigenvalue problem K*phi = lambda*M*phi.

        Parameters
        ----------
        numModes : int
            Number of modes to compute.
        solver : str
            'genBandArpack' (default, sparse shift-invert) or 'fullGenLapack' (dense).

        Returns
        -------
        eigenvalues : numpy array
            Array of omega^2 values for the requested modes.
        """
        # 1. Number DOFs, init elements
        self._domain()

        # 2. Assemble K and M
        self._assemble(time=0.0)

        if self._M is None:
            raise ValueError("Domain.eigen() - No mass matrix assembled. "
                             "Elements must have mass (rho > 0) for eigen analysis.")

        # 3. Identify free / fixed DOFs (same logic as Analysis._organize)
        uu = []  # free DOFs
        pp = []  # fixed DOFs
        for node in self.__nodes:
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
                    pp.append(dof_list[j])
                else:
                    uu.append(dof_list[j])

        if len(uu) == 0:
            raise ValueError("Domain.eigen() - No free DOFs. Cannot solve eigenvalue problem.")

        if numModes > len(uu):
            raise ValueError(
                f"Domain.eigen() - requested {numModes} modes but only {len(uu)} free DOFs.")

        # 4. Extract free-DOF submatrices
        K_full = np.asarray(self._K)
        M_full = np.asarray(self._M)
        uu_idx = np.array(uu)
        K_uu = K_full[np.ix_(uu_idx, uu_idx)]
        M_uu = M_full[np.ix_(uu_idx, uu_idx)]

        # 5. Solve
        solver_obj = Eigen(solver)
        eigenvalues, eigenvectors = solver_obj.solve(K_uu, M_uu, numModes)

        # 6. Store results
        self.__eigenvalues = eigenvalues
        self.__eigenvectors = eigenvectors
        self.__eigenDOFs = uu

        # 7. Trigger mode shape recorders
        self._record_eigen()

        return eigenvalues


    def modalProperties(self):
        """Compute and print modal properties: frequencies, periods,
        participation factors, and effective modal masses.

        Returns
        -------
        props : dict
            Dictionary with keys 'eigenvalues', 'omega', 'freq', 'period',
            'participation', 'effectiveMass', 'massParticipation'.
        """
        if self.__eigenvalues is None:
            raise ValueError("Domain.modalProperties() - Run eigen() first.")

        eigenvalues = self.__eigenvalues
        eigenvectors = self.__eigenvectors
        uu = self.__eigenDOFs
        numModes = len(eigenvalues)
        nDOF_free = len(uu)

        M_full = np.asarray(self._M)
        M_uu = M_full[np.ix_(np.array(uu), np.array(uu))]

        # Total mass per translational direction
        nDim = self.__nDim
        total_mass = np.zeros(nDim)
        for node in self.__nodes:
            node_dofs = node.getDOFs()
            if hasattr(node_dofs, 'tolist'):
                dof_list = node_dofs.tolist()
            elif hasattr(node_dofs, 'data'):
                dof_list = node_dofs.data.tolist()
            else:
                dof_list = list(node_dofs)
            for d in range(nDim):
                if d < len(dof_list):
                    global_dof = dof_list[d]
                    total_mass[d] += M_full[global_dof, global_dof]

        # Build influence vectors for each translational direction
        # r_dir has 1.0 at the DOF corresponding to direction d for each node
        uu_set = {dof: idx for idx, dof in enumerate(uu)}

        omega = np.zeros(numModes)
        freq = np.zeros(numModes)
        period = np.zeros(numModes)
        participation = np.zeros((numModes, nDim))
        eff_mass = np.zeros((numModes, nDim))

        for i in range(numModes):
            lam = eigenvalues[i]
            omega[i] = np.sqrt(abs(lam))
            freq[i] = omega[i] / (2.0 * np.pi)
            period[i] = 1.0 / freq[i] if freq[i] > 0.0 else float('inf')

            phi = eigenvectors[:, i]

            for d in range(nDim):
                # Build influence vector for direction d (in free-DOF space)
                r = np.zeros(nDOF_free)
                for node in self.__nodes:
                    node_dofs = node.getDOFs()
                    if hasattr(node_dofs, 'tolist'):
                        dof_list = node_dofs.tolist()
                    elif hasattr(node_dofs, 'data'):
                        dof_list = node_dofs.data.tolist()
                    else:
                        dof_list = list(node_dofs)

                    if d < len(dof_list):
                        global_dof = dof_list[d]
                        if global_dof in uu_set:
                            r[uu_set[global_dof]] = 1.0

                # Gamma = phi^T * M * r (mass-normalized phi => phi^T*M*phi = 1)
                gamma = phi.dot(M_uu.dot(r))
                participation[i, d] = gamma
                eff_mass[i, d] = gamma ** 2

        # Cumulative mass participation ratio
        cum_mass_ratio = np.zeros((numModes, nDim))
        for d in range(nDim):
            cumsum = 0.0
            for i in range(numModes):
                cumsum += eff_mass[i, d]
                if total_mass[d] > 0.0:
                    cum_mass_ratio[i, d] = cumsum / total_mass[d]

        # Print table
        dir_labels = ['X', 'Y', 'Z'][:nDim]
        print("\n" + "=" * 72)
        print("  MODAL PROPERTIES")
        print("=" * 72)
        print(f"  {'Mode':>4s}  {'omega':>12s}  {'f [Hz]':>12s}  {'T [s]':>12s}")
        print("-" * 72)
        for i in range(numModes):
            print(f"  {i+1:4d}  {omega[i]:12.6f}  {freq[i]:12.6f}  {period[i]:12.6f}")

        print()
        header = f"  {'Mode':>4s}"
        for d in range(nDim):
            header += f"  {'Gamma_'+dir_labels[d]:>10s}  {'Meff_'+dir_labels[d]:>10s}  {'%Cum_'+dir_labels[d]:>10s}"
        print(header)
        print("-" * 72)
        for i in range(numModes):
            line = f"  {i+1:4d}"
            for d in range(nDim):
                line += f"  {participation[i,d]:10.4f}  {eff_mass[i,d]:10.4f}  {cum_mass_ratio[i,d]*100:9.2f}%"
            print(line)
        print("=" * 72 + "\n")

        return {
            'eigenvalues': eigenvalues,
            'omega': omega,
            'freq': freq,
            'period': period,
            'participation': participation,
            'effectiveMass': eff_mass,
            'massParticipation': cum_mass_ratio,
            'totalMass': total_mass
        }


    def getEigenvalue(self, mode):
        """Return eigenvalue (omega^2) for the given mode (1-based)."""
        if self.__eigenvalues is None:
            raise ValueError("Domain.getEigenvalue() - Run eigen() first.")
        if mode < 1 or mode > len(self.__eigenvalues):
            raise ValueError(
                f"Domain.getEigenvalue() - Mode {mode} out of range [1, {len(self.__eigenvalues)}].")
        return self.__eigenvalues[mode - 1]


    def getEigenvector(self, mode):
        """Return eigenvector for the given mode (1-based).
        Returns full-DOF vector (zeros at fixed DOFs)."""
        if self.__eigenvectors is None:
            raise ValueError("Domain.getEigenvector() - Run eigen() first.")
        if mode < 1 or mode > self.__eigenvectors.shape[1]:
            raise ValueError(
                f"Domain.getEigenvector() - Mode {mode} out of range [1, {self.__eigenvectors.shape[1]}].")

        phi_free = self.__eigenvectors[:, mode - 1]
        phi_full = np.zeros(self.__nDOF)
        for idx, dof in enumerate(self.__eigenDOFs):
            phi_full[dof] = phi_free[idx]
        return phi_full
