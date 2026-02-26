# Diffusion term — connectivity-based unified implementation
#
# A single ``diffusionTerm`` function that works for ALL mesh types
# (structured and unstructured, every coordinate system) by operating
# only on the connectivity arrays stored on the mesh: owner, neighbor,
# d_CF, face_areas, cell_volumes.
#
# The face-level diffusion flux is:
#
#   F_f = D_f * A_f * (phi_N - phi_O) / d_CF
#
# For cell P the diffusion contribution is (1/V_P) * sum_faces(F_f * sign).
# This leads to the matrix coefficient for face f:
#
#   coeff_f = D_f * A_f / d_CF_f
#
# Owner row O gets:  M[O, N] += coeff / V_O,  M[O, O] -= coeff / V_O
# Neighbor row N gets: M[N, O] += coeff / V_N,  M[N, N] -= coeff / V_N
#
# For boundary faces the neighbor is a ghost cell (index >= num_cells).
# Both owner- and ghost-column entries are created, but the ghost *row*
# is not touched — it is handled by ``boundaryConditionsTerm``.

import numpy as np
from scipy.sparse import csr_array

from .face import FaceVariable


def diffusionTerm(D: FaceVariable) -> csr_array:
    """Discretised diffusion operator in matrix form.

    Builds and returns a sparse matrix ``M`` such that the diffusion
    term ``div(D grad(phi))`` is represented as ``M @ phi_all``, where
    ``phi_all`` is the flat vector of all cell values (interior +
    ghost).

    Parameters
    ----------
    D : FaceVariable
        Diffusion coefficient defined on face centres.

    Returns
    -------
    M : csr_array, shape (N_total, N_total)
        Sparse matrix.  ``N_total = num_cells + num_ghost_cells``.

    Notes
    -----
    Only interior-cell rows (indices ``0 .. num_cells-1``) receive
    diffusion entries.  Ghost-cell rows are left empty — they must be
    filled by ``boundaryConditionsTerm``.

    The formula works for every coordinate system because face areas
    and cell volumes already include all metric scale factors.

    Examples
    --------
    >>> import pyfvtool as pf
    >>> m = pf.Grid1D(10, 1.0)
    >>> D = pf.FaceVariable(m, 1.0)
    >>> M = pf.diffusionTerm(D)
    """
    m = D.domain
    N = m.num_cells
    N_total = N + m.num_ghost_cells
    nf = m.num_faces

    owner = m.owner  # (nf,)  cell indices
    neighbor = m.neighbor  # (nf,)  cell indices (ghost for boundary)
    d_CF = m.d_CF  # (nf,)
    A = m.face_areas  # (nf,)
    V = m.cell_volumes  # (N,)   interior only

    D_val = D._value  # (nf,)  diffusion coeff on faces

    # Face-level conductance: D_f * A_f / d_CF_f
    coeff = D_val * A / d_CF  # (nf,)

    # Mask for internal faces (neighbor is an interior cell)
    is_internal = neighbor < N  # (nf,) bool

    # ------------------------------------------------------------------
    # Build COO entries
    # Each face contributes entries to the owner row.  Internal faces
    # also contribute to the neighbor row.
    # ------------------------------------------------------------------
    # Owner side (all faces):
    #   M[O, N_nbr] += coeff / V_O   (off-diagonal)
    #   M[O, O]     -= coeff / V_O   (diagonal)
    coeff_over_Vo = coeff / V[owner]  # safe: owner is always interior

    # Neighbor side (internal faces only):
    #   M[N_nbr, O] += coeff / V_N
    #   M[N_nbr, N_nbr] -= coeff / V_N
    int_idx = np.where(is_internal)[0]
    coeff_int = coeff[int_idx]
    nbr_int = neighbor[int_idx]
    own_int = owner[int_idx]
    coeff_over_Vn = coeff_int / V[nbr_int]

    # Assemble COO arrays
    # Owner off-diagonal: (owner, neighbor, +coeff/V_O)
    row_o_off = owner
    col_o_off = neighbor
    val_o_off = coeff_over_Vo

    # Owner diagonal: (owner, owner, -coeff/V_O)
    row_o_diag = owner
    col_o_diag = owner
    val_o_diag = -coeff_over_Vo

    # Neighbor off-diagonal: (neighbor, owner, +coeff/V_N)
    row_n_off = nbr_int
    col_n_off = own_int
    val_n_off = coeff_over_Vn

    # Neighbor diagonal: (neighbor, neighbor, -coeff/V_N)
    row_n_diag = nbr_int
    col_n_diag = nbr_int
    val_n_diag = -coeff_over_Vn

    # Stack everything
    rows = np.concatenate([row_o_off, row_o_diag, row_n_off, row_n_diag])
    cols = np.concatenate([col_o_off, col_o_diag, col_n_off, col_n_diag])
    vals = np.concatenate([val_o_off, val_o_diag, val_n_off, val_n_diag])

    M = csr_array((vals, (rows, cols)), shape=(N_total, N_total))
    return M
