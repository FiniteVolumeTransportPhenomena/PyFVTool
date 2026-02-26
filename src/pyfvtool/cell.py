# CellVariable class definition and operator overloading
#
# CellVariable stores values on cell centres as a flat array of size
# (num_cells + num_ghost_cells).  The first num_cells entries are the
# interior cell values; the remaining entries are ghost-cell values
# (one per boundary face).

from copy import deepcopy
from typing import overload

import numpy as np

from .mesh import MeshStructure
from .utilities import TrackedArray


class CellVariable:
    """A scalar field defined on the cell centres of a mesh.

    Parameters
    ----------
    mesh : MeshStructure
        The mesh on which this variable lives.
    cell_value : float or ndarray
        Initial value(s).  A scalar broadcasts to all cells.  An array
        of shape ``(num_cells,)`` sets interior cells directly.  An
        array of shape ``(num_cells + num_ghost_cells,)`` sets both
        interior and ghost cells.  Multi-dimensional arrays (e.g. from
        structured meshes) are flattened in C order before size checks.
    """

    def __init__(self, mesh: MeshStructure, cell_value=0.0, *_extra_args):
        # *_extra_args: legacy 3-arg form CellVariable(mesh, value, BC)
        # The BC argument is silently ignored — use phi.BCs instead.
        self.domain = mesh
        self._BCs = None  # lazy; created on first .BCs access
        N = mesh.num_cells
        Ng = mesh.num_ghost_cells
        total = N + Ng

        def _try_array(arr):
            """Attempt to set self._value from a numpy array, with
            automatic flattening of multi-dimensional arrays."""
            if arr.size == 1:
                return TrackedArray(float(arr.flat[0]) * np.ones(total))
            # Flatten multi-dimensional arrays (e.g. shape (Nx, Ny) for 2D)
            flat = arr.ravel()
            if flat.shape == (N,):
                buf = np.zeros(total)
                buf[:N] = flat
                return TrackedArray(buf)
            elif flat.shape == (total,):
                return TrackedArray(flat.copy())
            else:
                raise ValueError(
                    f"cell_value shape {arr.shape} (flat size {flat.size}) "
                    f"is not valid for a mesh with {N} cells and {Ng} "
                    f"ghost cells. Expected scalar, ({N},), or ({total},)."
                )

        if np.isscalar(cell_value):
            self._value = TrackedArray(float(cell_value) * np.ones(total))
        elif isinstance(cell_value, np.ndarray):
            self._value = _try_array(cell_value)
        else:
            arr_val = np.asarray(cell_value, dtype=float)
            self._value = _try_array(arr_val)

        self._value.modified = False

    @property
    def value(self):
        """Interior cell values as a flat array (excludes ghost cells).

        Returns a view of shape ``(num_cells,)``.  For structured
        multi-dimensional indexing, use :attr:`shaped_value` instead.
        """
        return self._value[: self.domain.num_cells]

    @property
    def shaped_value(self):
        """Interior cell values reshaped to match structured mesh dims.

        For structured meshes returns a view of shape ``(Nx,)`` for 1D,
        ``(Nx, Ny)`` for 2D, or ``(Nx, Ny, Nz)`` for 3D.  Falls back
        to the flat ``(num_cells,)`` array for unstructured meshes.
        """
        flat = self._value[: self.domain.num_cells]
        dims = getattr(self.domain, "dims", None)
        if dims is not None and len(dims) > 1:
            return flat.reshape(tuple(dims))
        return flat

    @value.setter
    def value(self, values):
        self._value[: self.domain.num_cells] = np.asarray(values).ravel()

    @property
    def BCs(self):
        """Boundary conditions (lazy, auto-created on first access).

        Returns a ``BoundaryConditions`` instance associated with this
        CellVariable.  If one hasn't been set, a default Neumann BC
        object is created automatically (backward compatibility).
        """
        if self._BCs is None:
            from .boundary import BoundaryConditions

            self._BCs = BoundaryConditions(self.domain)
        return self._BCs

    @BCs.setter
    def BCs(self, bc):
        self._BCs = bc

    @property
    def ghost_value(self):
        """Ghost cell values."""
        return self._value[self.domain.num_cells :]

    @ghost_value.setter
    def ghost_value(self, values):
        self._value[self.domain.num_cells :] = values

    @property
    def cellvolume(self):
        """Cell volumes as a flat array (read-only)."""
        return self.domain.cell_volumes

    @property
    def cellcenters(self):
        """Cell centre coordinates (read-only)."""
        return self.domain.cellcenters

    # ------------------------------------------------------------------
    #  Arithmetic operators
    # ------------------------------------------------------------------

    def _transfer_BCs(self, result):
        """Copy BCs from self to result if they exist."""
        if self._BCs is not None:
            result._BCs = deepcopy(self._BCs)
        return result

    def _binop(self, other, op):
        if isinstance(other, CellVariable):
            result = CellVariable(self.domain, op(self.value, other.value))
        else:
            result = CellVariable(self.domain, op(self.value, other))
        return self._transfer_BCs(result)

    def _rbinop(self, other, op):
        if isinstance(other, CellVariable):
            result = CellVariable(self.domain, op(other.value, self.value))
        else:
            result = CellVariable(self.domain, op(other, self.value))
        return self._transfer_BCs(result)

    def __add__(self, other):
        return self._binop(other, np.add)

    def __radd__(self, other):
        return self._rbinop(other, np.add)

    def __sub__(self, other):
        return self._binop(other, np.subtract)

    def __rsub__(self, other):
        return self._rbinop(other, np.subtract)

    def __mul__(self, other):
        return self._binop(other, np.multiply)

    def __rmul__(self, other):
        return self._rbinop(other, np.multiply)

    def __truediv__(self, other):
        return self._binop(other, np.true_divide)

    def __rtruediv__(self, other):
        return self._rbinop(other, np.true_divide)

    def __pow__(self, other):
        return self._binop(other, np.power)

    def __rpow__(self, other):
        return self._rbinop(other, np.power)

    def __neg__(self):
        return self._transfer_BCs(CellVariable(self.domain, -self.value))

    def __pos__(self):
        return self._transfer_BCs(CellVariable(self.domain, self.value.copy()))

    def __abs__(self):
        return self._transfer_BCs(CellVariable(self.domain, np.abs(self.value)))

    # ------------------------------------------------------------------
    #  Comparison operators
    # ------------------------------------------------------------------

    def __gt__(self, other):
        return self._binop(other, np.greater)

    def __ge__(self, other):
        return self._binop(other, np.greater_equal)

    def __lt__(self, other):
        return self._binop(other, np.less)

    def __le__(self, other):
        return self._binop(other, np.less_equal)

    def __and__(self, other):
        return self._binop(other, np.logical_and)

    def __or__(self, other):
        return self._binop(other, np.logical_or)

    # ------------------------------------------------------------------
    #  Utility methods
    # ------------------------------------------------------------------

    def copy(self):
        """Return a deep copy of this CellVariable."""
        c = CellVariable(self.domain, self._value.copy())
        if self._BCs is not None:
            c._BCs = deepcopy(self._BCs)
        return c

    def update_value(self, new_cell):
        """Copy values from another CellVariable into this one."""
        np.copyto(self._value, new_cell._value)
        self._value.modified = True

    def apply_BCs(self):
        """Apply boundary conditions to update ghost-cell values.

        Calls ``boundary.apply_BCs(self, self.BCs)`` and then resets
        the *modified* flags on both the BCs and the value array, so
        that subsequent checks see a "clean" state.
        """
        from .boundary import apply_BCs as _apply_BCs

        _apply_BCs(self, self.BCs)
        # Reset all modified flags: BCs have been applied, ghost cells
        # are up-to-date, and the state is considered "clean".
        self.BCs.modified = False
        self._value._modified = False

    def plotprofile(self):
        """Return data suitable for plotting this CellVariable.

        Returns
        -------
        For 1-D meshes: ``(x, phi)``
        For 2-D meshes: ``(x, y, phi)``
        For 3-D meshes: ``(x, y, z, phi)``

        The coordinate arrays are the face locations (including
        boundary faces), and the value array includes ghost-cell
        values so it has size ``N+2`` in each direction.  This matches
        the original MATLAB FVTool ``plotprofile()`` convention used
        by ``visualizeCells``.
        """
        from .boundary import cellValuesWithBoundaries

        phi_all = cellValuesWithBoundaries(self, self.BCs)
        mesh = self.domain
        dim = mesh.dimension
        dims = getattr(mesh, "dims", None)

        if dim == 1 and dims is not None:
            Nx = int(dims[0])
            # phi_all has Nx interior + 2 ghost cells (left, right)
            # Reconstruct [boundary_left, interior..., boundary_right]
            # At boundary positions we use the interpolated face value
            # (average of owner cell and ghost cell) rather than the
            # raw ghost cell value, so the plot matches the BC exactly.
            phi_int = phi_all[:Nx]
            phi_ghost_left = phi_all[Nx]
            phi_ghost_right = phi_all[Nx + 1]
            phi_plot = np.empty(Nx + 2)
            phi_plot[1:-1] = phi_int
            phi_plot[0] = 0.5 * (phi_int[0] + phi_ghost_left)
            phi_plot[-1] = 0.5 * (phi_int[-1] + phi_ghost_right)
            # x-axis: [left_boundary, cell_centers..., right_boundary]
            x = np.empty(Nx + 2)
            x[0] = mesh._face_loc_x[0]
            x[1:-1] = mesh._cell_center_x
            x[-1] = mesh._face_loc_x[-1]
            return x, phi_plot

        elif dim == 2 and dims is not None:
            Nx, Ny = int(dims[0]), int(dims[1])
            N = Nx * Ny
            # Interior cells are stored in row-major (C) order: index [i,j] = i*Ny + j
            phi_int = phi_all[:N].reshape(Nx, Ny)

            # Ghost cells are appended after interior cells.
            # For 2D structured meshes:
            #   boundary_tags: left (Ny faces), right (Ny faces),
            #                  bottom (Nx faces), top (Nx faces)
            g = N  # ghost cell start index
            left_ghosts = phi_all[g : g + Ny]  # Ny ghosts for left boundary
            g += Ny
            right_ghosts = phi_all[g : g + Ny]  # Ny ghosts for right boundary
            g += Ny
            bottom_ghosts = phi_all[g : g + Nx]  # Nx ghosts for bottom boundary
            g += Nx
            top_ghosts = phi_all[g : g + Nx]  # Nx ghosts for top boundary

            # Build (Nx+2, Ny+2) array
            phi_plot = np.zeros((Nx + 2, Ny + 2))
            phi_plot[1:-1, 1:-1] = phi_int
            phi_plot[0, 1:-1] = left_ghosts
            phi_plot[-1, 1:-1] = right_ghosts
            phi_plot[1:-1, 0] = bottom_ghosts
            phi_plot[1:-1, -1] = top_ghosts
            # Corners: average of adjacent ghosts (approximate)
            phi_plot[0, 0] = 0.5 * (left_ghosts[0] + bottom_ghosts[0])
            phi_plot[0, -1] = 0.5 * (left_ghosts[-1] + top_ghosts[0])
            phi_plot[-1, 0] = 0.5 * (right_ghosts[0] + bottom_ghosts[-1])
            phi_plot[-1, -1] = 0.5 * (right_ghosts[-1] + top_ghosts[-1])

            # x-axis: [left_boundary, cell_centers_x..., right_boundary]
            x = np.empty(Nx + 2)
            x[0] = mesh._face_loc_x[0]
            x[1:-1] = mesh._cell_center_x
            x[-1] = mesh._face_loc_x[-1]
            # y-axis: [bottom_boundary, cell_centers_y..., top_boundary]
            y = np.empty(Ny + 2)
            y[0] = mesh._face_loc_y[0]
            y[1:-1] = mesh._cell_center_y
            y[-1] = mesh._face_loc_y[-1]
            return x, y, phi_plot

        elif dim == 3 and dims is not None:
            Nx, Ny, Nz = int(dims[0]), int(dims[1]), int(dims[2])
            N = Nx * Ny * Nz
            phi_int = phi_all[:N].reshape(Nx, Ny, Nz)

            g = N
            left_ghosts = phi_all[g : g + Ny * Nz].reshape(Ny, Nz)
            g += Ny * Nz
            right_ghosts = phi_all[g : g + Ny * Nz].reshape(Ny, Nz)
            g += Ny * Nz
            bottom_ghosts = phi_all[g : g + Nx * Nz].reshape(Nx, Nz)
            g += Nx * Nz
            top_ghosts = phi_all[g : g + Nx * Nz].reshape(Nx, Nz)
            g += Nx * Nz
            back_ghosts = phi_all[g : g + Nx * Ny].reshape(Nx, Ny)
            g += Nx * Ny
            front_ghosts = phi_all[g : g + Nx * Ny].reshape(Nx, Ny)

            phi_plot = np.zeros((Nx + 2, Ny + 2, Nz + 2))
            phi_plot[1:-1, 1:-1, 1:-1] = phi_int
            phi_plot[0, 1:-1, 1:-1] = left_ghosts
            phi_plot[-1, 1:-1, 1:-1] = right_ghosts
            phi_plot[1:-1, 0, 1:-1] = bottom_ghosts
            phi_plot[1:-1, -1, 1:-1] = top_ghosts
            phi_plot[1:-1, 1:-1, 0] = back_ghosts
            phi_plot[1:-1, 1:-1, -1] = front_ghosts

            x = np.empty(Nx + 2)
            x[0] = mesh._face_loc_x[0]
            x[1:-1] = mesh._cell_center_x
            x[-1] = mesh._face_loc_x[-1]
            y = np.empty(Ny + 2)
            y[0] = mesh._face_loc_y[0]
            y[1:-1] = mesh._cell_center_y
            y[-1] = mesh._face_loc_y[-1]
            z = np.empty(Nz + 2)
            z[0] = mesh._face_loc_z[0]
            z[1:-1] = mesh._cell_center_z
            z[-1] = mesh._face_loc_z[-1]
            return x, y, z, phi_plot

        else:
            # Fallback for unstructured meshes: return flat arrays
            cc = mesh._cell_centers
            if dim == 1:
                return cc[:, 0], phi_all[: mesh.num_cells]
            elif dim == 2:
                return cc[:, 0], cc[:, 1], phi_all[: mesh.num_cells]
            else:
                return cc[:, 0], cc[:, 1], cc[:, 2], phi_all[: mesh.num_cells]

    def domainIntegral(self) -> float:
        """Compute the finite-volume integral over the entire domain.

        Returns
        -------
        float
            Sum of (cell_volume * cell_value) over all interior cells.
        """
        return float(np.sum(self.cellvolume * self.value))

    def __repr__(self):
        return (
            f"CellVariable(mesh={self.domain.__class__.__name__}, "
            f"num_cells={self.domain.num_cells})"
        )

    def __str__(self):
        return repr(self)


# ------------------------------------------------------------------
#  Module-level helper functions
# ------------------------------------------------------------------


def cellLocations(m: MeshStructure):
    """Return CellVariable(s) containing cell centre coordinates.

    Parameters
    ----------
    m : MeshStructure
        The mesh.

    Returns
    -------
    For 1-D meshes: ``X``
    For 2-D meshes: ``(X, Y)``
    For 3-D meshes: ``(X, Y, Z)``

    Each is a CellVariable holding the corresponding coordinate.
    """
    dim = m.dimension
    cc = m._cell_centers  # (num_cells, dim)
    if dim == 1:
        return CellVariable(m, cc[:, 0])
    elif dim == 2:
        X = CellVariable(m, cc[:, 0])
        Y = CellVariable(m, cc[:, 1])
        return X, Y
    elif dim == 3:
        X = CellVariable(m, cc[:, 0])
        Y = CellVariable(m, cc[:, 1])
        Z = CellVariable(m, cc[:, 2])
        return X, Y, Z
    raise TypeError(f"Unsupported mesh dimension: {dim}")


def funceval(f, *args):
    """Apply a function to the values of one or more CellVariables.

    Parameters
    ----------
    f : callable
        Function to apply element-wise.
    *args : CellVariable
        Input CellVariables.

    Returns
    -------
    CellVariable
        Result of ``f(arg0.value, arg1.value, ...)``.
    """
    values = [a.value for a in args]
    result_val = f(*values)
    result = CellVariable(args[0].domain, result_val)
    # Transfer BCs from the first CellVariable argument
    args[0]._transfer_BCs(result)
    return result


def celleval(f, *args):
    """Alias for :func:`funceval`."""
    return funceval(f, *args)
