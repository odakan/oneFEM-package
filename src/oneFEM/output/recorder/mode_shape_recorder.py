##-----------------------------------------------------------------------##
#                                                                         #
#        #--oneFEM--#: One FEM software in a galaxy far far away          #
#                                                                         #
#                   Computational Mechanics 2022                          #
#                       University of Pavia                               #
#               Written by: Onur Deniz AKAN, IUSS Pavia                   #
#                         04 March 2026                                   #
#                                                                         #
##-----------------------------------------------------------------------##
#
# MODE SHAPE RECORDER
#   Records eigenvector mode shapes from a Domain after eigen() is solved.
#   Activated by Domain._record_eigen() (called at the end of eigen()).
#
#   Usage:
#       rec = ModeShapeRecorder(1, model, nodes=[nd1, nd2, nd3],
#                               dofs=[1, 2, 3], modes=[1, 2, 3])
#       model.add(rec)
#       model.eigen(3)          # triggers _record_eigen() automatically
#       rec.data['mode_1']      # {nodeID: [phi_dof1, phi_dof2, phi_dof3]}
#
#   File output:
#       rec.save('mode_shapes.out')
#       # Format per mode block:
#       #   # Mode 1  omega=...  freq=...  T=...
#       #   nodeID  phi_dof1  phi_dof2  ...
#       #   nodeID  phi_dof1  phi_dof2  ...
#

import numpy as np
from .main import Recorder


class ModeShapeRecorder(Recorder):
    def __init__(self, recID, domain=None, nodes=None, dofs=None, modes=None,
                 file=None):
        """
        Parameters
        ----------
        recID : int
            Recorder ID.
        domain : Domain
            The Domain object to read eigenvectors from.
        nodes : list of Node
            Nodes whose mode shape values to record.
            If None, all nodes in the domain are used.
        dofs : list of int
            DOF indices to record per node (1-based, e.g. [1,2,3]).
            If None, all translational DOFs (1..nDim) are used.
        modes : list of int
            Mode numbers to record (1-based).
            If None, all computed modes are recorded.
        file : str, optional
            File path for output (written via save()).
        """
        super().__init__(recID)
        self._domain = domain
        self._nodes = nodes
        self._dofs = dofs
        self._modes = modes
        self._file = file

    def record_eigen(self):
        """Capture mode shapes from the domain's eigen results.

        Called by Domain._record_eigen() after eigen() completes.
        Only records if eigenvalues are available.
        """
        if self._domain is None:
            return

        # Check if eigen has been solved
        try:
            self._domain.getEigenvalue(1)
        except (ValueError, AttributeError):
            return

        # Resolve nodes: default to all domain nodes
        nodes = self._nodes if self._nodes is not None else self._domain.nodes

        # Resolve dofs: default to translational DOFs (1..nDim)
        if self._dofs is not None:
            dofs = self._dofs
        else:
            dofs = list(range(1, self._domain.nDim + 1))

        # Resolve modes: default to all computed modes
        if self._modes is not None:
            modes = self._modes
        else:
            # Determine number of modes from eigenvalues
            mode = 1
            modes = []
            while True:
                try:
                    self._domain.getEigenvalue(mode)
                    modes.append(mode)
                    mode += 1
                except ValueError:
                    break

        # Clear previous data
        self.data = {}

        for mode in modes:
            phi_full = self._domain.getEigenvector(mode)
            lam = self._domain.getEigenvalue(mode)
            omega = np.sqrt(abs(lam))

            mode_data = {}
            for nd in nodes:
                node_dofs = nd.getDOFs()
                if hasattr(node_dofs, 'tolist'):
                    dof_list = node_dofs.tolist()
                elif hasattr(node_dofs, 'data'):
                    dof_list = node_dofs.data.tolist()
                else:
                    dof_list = list(node_dofs)

                vals = []
                for d in dofs:
                    if d - 1 < len(dof_list):
                        vals.append(phi_full[dof_list[d - 1]])
                    else:
                        vals.append(0.0)
                mode_data[nd._ID] = vals

            key = f'mode_{mode}'
            self.data[key] = mode_data
            # Store eigenvalue metadata
            self.data[f'eigenvalue_{mode}'] = lam
            self.data[f'omega_{mode}'] = omega
            self.data[f'freq_{mode}'] = omega / (2.0 * np.pi)

    def record(self, time=None):
        """No-op for time-history recording. Mode shapes are captured
        via record_eigen(), not the per-step record() call."""
        pass

    def save(self, filepath=None):
        """Write mode shapes to file.

        Format (per mode block):
            # Mode 1  eigenvalue=...  omega=...  freq=...  T=...
            nodeID  phi_dof1  phi_dof2  ...
            nodeID  phi_dof1  phi_dof2  ...
            <blank line>

        Parameters
        ----------
        filepath : str, optional
            Output file path. Defaults to self._file.
        """
        path = filepath or self._file
        if path is None:
            raise ValueError("ModeShapeRecorder.save() - No file path specified.")

        # Collect mode numbers from data keys
        mode_keys = sorted([k for k in self.data if k.startswith('mode_')],
                           key=lambda k: int(k.split('_')[1]))

        with open(path, 'w') as f:
            for key in mode_keys:
                mode_num = int(key.split('_')[1])
                mode_data = self.data[key]
                lam = self.data.get(f'eigenvalue_{mode_num}', 0.0)
                omega = self.data.get(f'omega_{mode_num}', 0.0)
                freq = self.data.get(f'freq_{mode_num}', 0.0)
                T = 1.0 / freq if freq > 0.0 else float('inf')

                f.write(f"# Mode {mode_num}  eigenvalue={lam:.10e}  "
                        f"omega={omega:.6f}  freq={freq:.6f}  T={T:.6f}\n")

                for nid in sorted(mode_data.keys()):
                    vals = mode_data[nid]
                    val_str = "  ".join(f"{v:+.10e}" for v in vals)
                    f.write(f"{nid:6d}  {val_str}\n")

                f.write("\n")

    def getNodeModeShape(self, mode, nodeID):
        """Get mode shape values for a specific node and mode.

        Parameters
        ----------
        mode : int
            Mode number (1-based).
        nodeID : int
            Node ID.

        Returns
        -------
        vals : list of float
            Mode shape values at the recorded DOFs.
        """
        key = f'mode_{mode}'
        if key not in self.data:
            raise ValueError(
                f"ModeShapeRecorder.getNodeModeShape() - Mode {mode} not recorded.")
        mode_data = self.data[key]
        if nodeID not in mode_data:
            raise ValueError(
                f"ModeShapeRecorder.getNodeModeShape() - Node {nodeID} not recorded.")
        return mode_data[nodeID]
