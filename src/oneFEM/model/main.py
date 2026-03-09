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
from .._systools.data import Vector
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

        # Global state vectors (owned by Domain, written by Analysis)
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

        # System of equations (set by Analysis for assembly/solve)
        self._system = None


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
        """Lazy accessor: returns assembled K from the system (requires Analysis setup)."""
        if self._system is not None:
            K_sp = self._system.getK()
            if K_sp is not None:
                return K_sp.toarray()
        return None

    @property
    def M(self):
        """Lazy accessor: returns assembled M from the system (requires Analysis setup)."""
        if self._system is not None:
            M_sp = self._system.getM()
            if M_sp is not None:
                return M_sp.toarray()
        return None

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

    def getRayleighCoeffs(self):
        return self.__alphaM, self.__betaK


    def _domain(self, numberer=None):
        """Number DOFs and initialize elements.

        Parameters
        ----------
        numberer : Numberer, optional
            If provided, the numberer assigns DOF numbers (e.g. RCM ordering).
            If None, uses sequential numbering (default, backward compatible).
        """
        self.__nNds = len(self.__nodes)
        if self.__nNds < 1:
            raise ValueError("Domain: No nodes in the domain!")

        self.__nElems = len(self.__elements)
        if self.__nElems < 1:
            raise ValueError("Domain: No elements in the domain!")

        # Build node lookup map
        self.__node_map = {node._ID: node for node in self.__nodes}

        if numberer is not None and hasattr(numberer, 'number'):
            # Numberer assigns DOF numbers (e.g. RCM)
            self.__nDOF = numberer.number(self)
        else:
            # Sequential DOF numbering (default)
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

        Uses SparseGeneral for assembly (same system as Analysis).

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
        from ..analysis.system.sparse_general import SparseGeneral

        # 1. Number DOFs, init elements
        self._domain()

        # 2. Build DOF maps and create assembly system
        node_dof_map = {}
        for nd in self.__nodes:
            dof_data = nd.getDOFs()
            if hasattr(dof_data, 'tolist'):
                node_dof_map[nd._ID] = np.array(dof_data.tolist(), dtype=np.int32)
            else:
                node_dof_map[nd._ID] = np.array(list(dof_data), dtype=np.int32)

        elem_dof_map = {}
        for elem in self.__elements:
            edofs = []
            for nd in elem.getNodes():
                edofs.extend(node_dof_map[nd._ID].tolist())
            elem_dof_map[elem._ID] = np.array(edofs, dtype=np.int32)

        system = SparseGeneral()
        system.setSize(self.__nDOF, elem_dof_map, self.__elements)
        self._system = system

        # 3. Assemble K and M into system
        system.zeroA()
        for elem in self.__elements:
            system.addA(np.asarray(elem.getStiffness()), elem._ID)
            me = elem.getMass()
            if me is not None:
                system.addM(np.asarray(me), elem._ID)

        M_sp = system.getM()
        if M_sp is None:
            raise ValueError("Domain.eigen() - No mass matrix assembled. "
                             "Elements must have mass (rho > 0) for eigen analysis.")

        # 4. Identify free / fixed DOFs
        uu = []
        pp = []
        for node in self.__nodes:
            nDOF = node.getNDOF()
            fix = node._fix
            dof_list = node_dof_map[node._ID].tolist()
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

        # 5. Extract free-DOF submatrices
        K_full = system.getK().toarray()
        M_full = M_sp.toarray()
        uu_idx = np.array(uu)
        K_uu = K_full[np.ix_(uu_idx, uu_idx)]
        M_uu = M_full[np.ix_(uu_idx, uu_idx)]

        # 6. Solve
        solver_obj = Eigen(solver)
        eigenvalues, eigenvectors = solver_obj.solve(K_uu, M_uu, numModes)

        # 7. Store results
        self.__eigenvalues = eigenvalues
        self.__eigenvectors = eigenvectors
        self.__eigenDOFs = uu

        # 8. Trigger mode shape recorders
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

        if self._system is None:
            raise ValueError("Domain.modalProperties() - No assembly system. Run eigen() first.")
        M_sp = self._system.getM()
        if M_sp is None:
            raise ValueError("Domain.modalProperties() - No mass matrix in system.")
        M_full = M_sp.toarray()
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
