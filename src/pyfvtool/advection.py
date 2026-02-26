# Advection/convection terms — connectivity-based unified implementation
#
# Three convection-term functions that work for ALL mesh types
# (structured and unstructured, every coordinate system) by operating
# only on the connectivity arrays stored on the mesh.
#
# The convection flux at a face is:
#
#   F_f = u_f * phi_f * A_f
#
# where u_f is the face-normal velocity component (from a FaceVariable),
# A_f is the face area, and phi_f is the interpolated face value of phi.
#
# For cell P the convection contribution is (1/V_P) * sum_faces(F_f * sign):
#
#   For the owner (O) of face f:  F_f contributes +u_f * phi_f * A_f / V_O
#   For the neighbor (N) of face f: F_f contributes -u_f * phi_f * A_f / V_N
#
# (The convention is that the face normal points from owner to neighbor.)
#
# Available schemes:
#
#   convectionTerm          — central (linear) differencing, second-order
#   convectionUpwindTerm    — first-order upwind (donor-cell)
#   convectionTvdTerm       — TVD deferred correction (upwind implicit +
#                             flux-limiter explicit RHS correction)

import numpy as np
from scipy.sparse import csr_array

from .face import FaceVariable
from .cell import CellVariable


def _fsign(phi_in, eps=1e-16):
    """Safe-sign function to avoid division by zero in gradient ratios.

    Returns the input where ``|phi_in| >= eps``, and ``eps * sign(phi_in)``
    for values very close to or equal to zero.
    """
    result = phi_in.copy()
    small = np.abs(phi_in) < eps
    result[small] = eps * np.sign(phi_in[small])
    result[phi_in == 0.0] = eps
    return result


# -----------------------------------------------------------------------
#  Central (linear) convection term
# -----------------------------------------------------------------------


def convectionTerm(u: FaceVariable) -> csr_array:
    """Discretised convection operator using central (linear) differencing.

    Builds and returns a sparse matrix ``M`` such that the convection
    term ``div(u * phi)`` is represented as ``M @ phi_all``, where
    ``phi_all`` is the flat vector of all cell values (interior + ghost).

    The face value of phi is linearly interpolated:

        ``phi_f = w * phi[owner] + (1 - w) * phi[neighbor]``

    where ``w = face_interpolation_weight = d_fF / d_CF``.

    Parameters
    ----------
    u : FaceVariable
        Face-normal velocity (or mass flux per unit area).  Positive
        direction is owner -> neighbor (aligned with face normal).

    Returns
    -------
    M : csr_array, shape (N_total, N_total)
        Sparse convection matrix.

    Notes
    -----
    Central differencing is second-order accurate but may produce
    oscillations near discontinuities.  For bounded solutions, use
    :func:`convectionUpwindTerm` or :func:`convectionTvdTerm`.

    Only interior-cell rows (indices ``0 .. num_cells-1``) receive
    entries.  Ghost-cell rows are left empty.

    Face areas and cell volumes already include all metric scale
    factors (cylindrical ``r``, spherical ``r^2 sin(theta)``, etc.),
    so this single function handles all coordinate systems.

    Examples
    --------
    >>> import pyfvtool as pf
    >>> m = pf.Grid1D(10, 1.0)
    >>> u = pf.FaceVariable(m, 1.0)
    >>> M = pf.convectionTerm(u)
    """
    m = u.domain
    N = m.num_cells
    N_total = N + m.num_ghost_cells
    nf = m.num_faces

    owner = m.owner  # (nf,)
    neighbor = m.neighbor  # (nf,)
    A = m.face_areas  # (nf,)
    V = m.cell_volumes  # (N,) interior only
    w = m.face_interpolation_weight  # (nf,)

    u_val = u._value  # (nf,) face velocity

    # Face flux times area: F = u * A
    F = u_val * A  # (nf,)

    # Linear interpolation: phi_f = w*phi_O + (1-w)*phi_N
    # Convection flux for owner row:
    #   +F * (w * phi_O + (1-w) * phi_N) / V_O
    # = owner-owner coeff: +F*w / V_O
    # = owner-neighbor coeff: +F*(1-w) / V_O
    #
    # For neighbor row (if interior):
    #   -F * (w * phi_O + (1-w) * phi_N) / V_N
    # = neighbor-owner coeff: -F*w / V_N
    # = neighbor-neighbor coeff: -F*(1-w) / V_N

    F_over_Vo = F / V[owner]  # safe: owner always interior

    # Mask for internal faces (neighbor is an interior cell)
    is_internal = neighbor < N
    int_idx = np.where(is_internal)[0]

    # Owner side (all faces)
    row_o_oo = owner  # owner-row, owner-col (diagonal)
    col_o_oo = owner
    val_o_oo = F_over_Vo * w  # +F*w/V_O

    row_o_on = owner  # owner-row, neighbor-col (off-diagonal)
    col_o_on = neighbor
    val_o_on = F_over_Vo * (1.0 - w)  # +F*(1-w)/V_O

    # Neighbor side (internal faces only)
    F_int = F[int_idx]
    nbr_int = neighbor[int_idx]
    own_int = owner[int_idx]
    w_int = w[int_idx]
    F_over_Vn = F_int / V[nbr_int]

    row_n_no = nbr_int  # neighbor-row, owner-col
    col_n_no = own_int
    val_n_no = -F_over_Vn * w_int

    row_n_nn = nbr_int  # neighbor-row, neighbor-col (diagonal)
    col_n_nn = nbr_int
    val_n_nn = -F_over_Vn * (1.0 - w_int)

    # Stack and build sparse matrix
    rows = np.concatenate([row_o_oo, row_o_on, row_n_no, row_n_nn])
    cols = np.concatenate([col_o_oo, col_o_on, col_n_no, col_n_nn])
    vals = np.concatenate([val_o_oo, val_o_on, val_n_no, val_n_nn])

    return csr_array((vals, (rows, cols)), shape=(N_total, N_total))


# -----------------------------------------------------------------------
#  First-order upwind convection term
# -----------------------------------------------------------------------


def convectionUpwindTerm(u: FaceVariable, u_upwind: FaceVariable = None) -> csr_array:
    """Discretised convection operator using first-order upwind.

    For each face:
    - If ``u_f > 0`` (flow owner -> neighbor):  ``phi_f = phi[owner]``
    - If ``u_f < 0`` (flow neighbor -> owner):  ``phi_f = phi[neighbor]``
    - If ``u_f == 0``:  no contribution.

    Parameters
    ----------
    u : FaceVariable
        Face-normal velocity (or mass flux per unit area).
    u_upwind : FaceVariable, optional
        Velocity used to determine upwind direction.  If not given,
        ``u`` is used for both magnitude and direction.

    Returns
    -------
    M : csr_array, shape (N_total, N_total)
        Sparse upwind convection matrix.

    Notes
    -----
    First-order upwind is unconditionally bounded but numerically
    diffusive (first-order accurate).

    Examples
    --------
    >>> import pyfvtool as pf
    >>> m = pf.Grid1D(10, 1.0)
    >>> u = pf.FaceVariable(m, 1.0)
    >>> M = pf.convectionUpwindTerm(u)
    """
    if u_upwind is None:
        u_upwind = u

    m = u.domain
    N = m.num_cells
    N_total = N + m.num_ghost_cells
    nf = m.num_faces

    owner = m.owner
    neighbor = m.neighbor
    A = m.face_areas
    V = m.cell_volumes

    u_val = u._value  # (nf,) actual velocity magnitude
    u_dir = u_upwind._value  # (nf,) velocity for direction determination

    F = u_val * A  # face flux (magnitude * area)

    # Split flux based on upwind direction
    # F_pos: flux when flow is owner -> neighbor (u_dir > 0)
    # F_neg: flux when flow is neighbor -> owner (u_dir < 0)
    F_pos = np.where(u_dir > 0, F, 0.0)
    F_neg = np.where(u_dir < 0, F, 0.0)

    # When u_dir > 0: phi_f = phi_O => flux = F_pos * phi_O
    #   Owner row: +F_pos * phi_O / V_O  =>  M[O, O] += F_pos/V_O
    #   Neighbor row (int): -F_pos * phi_O / V_N  =>  M[N, O] -= F_pos/V_N
    #
    # When u_dir < 0: phi_f = phi_N => flux = F_neg * phi_N
    #   Owner row: +F_neg * phi_N / V_O  =>  M[O, N] += F_neg/V_O
    #   Neighbor row (int): -F_neg * phi_N / V_N  =>  M[N, N] -= F_neg/V_N

    is_internal = neighbor < N
    int_idx = np.where(is_internal)[0]

    # ---------- Owner side (all faces) ----------
    # Positive flux: M[O, O] += F_pos / V_O
    F_pos_over_Vo = F_pos / V[owner]
    row_o_oo = owner
    col_o_oo = owner
    val_o_oo = F_pos_over_Vo

    # Negative flux: M[O, N] += F_neg / V_O
    F_neg_over_Vo = F_neg / V[owner]
    row_o_on = owner
    col_o_on = neighbor
    val_o_on = F_neg_over_Vo

    # ---------- Neighbor side (internal faces only) ----------
    F_pos_int = F_pos[int_idx]
    F_neg_int = F_neg[int_idx]
    nbr_int = neighbor[int_idx]
    own_int = owner[int_idx]

    # Positive flux: M[N, O] -= F_pos / V_N
    F_pos_over_Vn = F_pos_int / V[nbr_int]
    row_n_no = nbr_int
    col_n_no = own_int
    val_n_no = -F_pos_over_Vn

    # Negative flux: M[N, N] -= F_neg / V_N
    F_neg_over_Vn = F_neg_int / V[nbr_int]
    row_n_nn = nbr_int
    col_n_nn = nbr_int
    val_n_nn = -F_neg_over_Vn

    rows = np.concatenate([row_o_oo, row_o_on, row_n_no, row_n_nn])
    cols = np.concatenate([col_o_oo, col_o_on, col_n_no, col_n_nn])
    vals = np.concatenate([val_o_oo, val_o_on, val_n_no, val_n_nn])

    return csr_array((vals, (rows, cols)), shape=(N_total, N_total))


# -----------------------------------------------------------------------
#  TVD convection term (deferred correction)
# -----------------------------------------------------------------------


def convectionTvdTerm(
    u: FaceVariable, phi: CellVariable, FL, u_upwind: FaceVariable = None
):
    """TVD convection: upwind matrix + flux-limiter RHS correction.

    Returns both the implicit (upwind) matrix and the explicit
    (deferred correction) RHS vector.  The TVD scheme is:

        ``M_upwind @ phi_new = ... + RHS_tvd_correction``

    where ``M_upwind`` is the first-order upwind matrix and
    ``RHS_tvd_correction`` adds anti-diffusive flux controlled by the
    flux limiter ``FL``.

    Parameters
    ----------
    u : FaceVariable
        Face-normal velocity.
    phi : CellVariable
        Current (known) value of the solution variable.  Ghost-cell
        values must be up-to-date.
    FL : callable
        Flux limiter function ``FL(r)`` returning the limiter value.
        See :func:`utilities.fluxLimiter` for available limiters.
    u_upwind : FaceVariable, optional
        Velocity for upwind direction determination.  Defaults to ``u``.

    Returns
    -------
    M : csr_array, shape (N_total, N_total)
        Upwind convection matrix (implicit part).
    RHS : ndarray, shape (N_total,)
        TVD correction (explicit part, to be added to the RHS).

    Notes
    -----
    Usage in a time-stepping loop::

        FL = pf.fluxLimiter('VanLeer')
        for step in range(nsteps):
            M_conv, RHS_conv = pf.convectionTvdTerm(u, phi, FL)
            phi = pf.solvePDE(phi, BC, [(-M_conv, -RHS_conv), ...])

    The TVD correction is evaluated explicitly using the current
    ``phi`` values.  For steady-state problems, iterate until
    convergence.

    The anti-diffusive correction at each face is:

        ``psi_f = 0.5 * FL(r) * (phi_downwind - phi_upwind)``

    where ``r`` is the ratio of consecutive gradients (measuring
    solution smoothness).

    Examples
    --------
    >>> import pyfvtool as pf
    >>> m = pf.Grid1D(20, 1.0)
    >>> FL = pf.fluxLimiter('VanLeer')
    >>> u = pf.FaceVariable(m, 1.0)
    >>> phi = pf.CellVariable(m)
    >>> M, RHS = pf.convectionTvdTerm(u, phi, FL)
    """
    if u_upwind is None:
        u_upwind = u

    m = u.domain
    N = m.num_cells
    N_total = N + m.num_ghost_cells
    nf = m.num_faces

    owner = m.owner
    neighbor = m.neighbor
    A = m.face_areas
    V = m.cell_volumes
    d_CF = m.d_CF

    u_val = u._value
    u_dir = u_upwind._value
    phi_val = phi._value  # includes ghost cells

    F = u_val * A  # face flux

    # ----- Compute gradient at each face -----
    # dphi_f = (phi_N - phi_O) / d_CF
    dphi_face = (phi_val[neighbor] - phi_val[owner]) / d_CF  # (nf,)

    # ----- Build face adjacency for gradient ratio computation -----
    # For each face f, we need the "upstream face" to compute the
    # gradient ratio r.  This requires knowing which faces are
    # adjacent to the upstream cell (beyond the current face).
    #
    # For structured meshes, this is straightforward: the upstream face
    # of cell i in direction +x is the west face of cell i.
    #
    # For the general connectivity-based approach, we use the fact that
    # for each interior cell, we can compute the gradient at that cell's
    # center, and then use that to estimate the upstream gradient.
    #
    # Alternative approach (simpler, works for all meshes):
    # Compute a cell-centered gradient, then use it to estimate the
    # "upstream gradient" for the gradient ratio.
    #
    # For face f between O and N:
    #   When u > 0 (flow O -> N), upstream cell is O.
    #     The "upstream gradient" at O projected onto the O-N direction:
    #       r = (2 * grad_O . d_Of / d_CF - dphi_face) / dphi_face
    #     Simplification: r = 2 * (phi_O - phi_UU) / (phi_N - phi_O) - 1
    #     where phi_UU is the value at the cell upstream of O.
    #
    # Since we don't easily have access to the "face upstream of the
    # upstream cell" in the general connectivity, we use the compact
    # stencil approach: approximate the upstream gradient using the
    # cell-centered gradient at the upstream cell projected onto the
    # face direction.
    #
    # A cleaner approach for all meshes: For each face, compute
    #   r_pos = (phi_O - phi_OO) / _fsign(phi_N - phi_O)
    # where phi_OO is reconstructed as phi_OO = 2*phi_O - phi_N
    # (extrapolation through O using the local gradient).
    # This reduces to the standard TVD formulation on uniform meshes.
    #
    # The most robust general approach:
    # For each cell, precompute the least-squares gradient, then project.
    # This is expensive but works on unstructured meshes.
    #
    # For NOW, we use the approach that works on structured meshes and
    # gives a reasonable approximation on unstructured:
    #
    # r = 2 * d_Cf * grad_O . n_hat / (phi_N - phi_O) - 1  (for u > 0)
    # where grad_O is estimated from adjacent face gradients.
    #
    # Actually, let's use the proven approach from the old code:
    # On structured meshes, faces are ordered so that we can directly
    # compute consecutive gradient ratios.  For the general case, we
    # build per-cell "upwind face" and "downwind face" pairs.

    # --- General TVD using cell-centered gradient reconstruction ---
    # For each cell i, compute grad_phi_i via the Green-Gauss theorem:
    #   grad_phi_i = (1/V_i) * sum_faces(phi_f * A_f * n_hat_f * sign_f)
    # where phi_f is the face value (linear mean) and sign_f is +1 if i
    # is the owner, -1 if i is the neighbor.

    # Compute phi at faces (linear interp) for gradient estimation
    w_interp = m.face_interpolation_weight
    phi_face = w_interp * phi_val[owner] + (1.0 - w_interp) * phi_val[neighbor]

    # Build cell-centered gradient using Green-Gauss
    dim = m.dimension
    face_normals = m.face_normals  # (nf, dim)

    # Flux contribution of each face to the gradient
    grad_flux = phi_face[:, None] * face_normals * A[:, None]  # (nf, dim)

    # Accumulate into cells
    cell_grad = np.zeros((N_total, dim))
    for d in range(dim):
        np.add.at(cell_grad[:, d], owner, grad_flux[:, d])
        np.add.at(cell_grad[:, d], neighbor, -grad_flux[:, d])

    # Divide by cell volume (interior only)
    cell_grad[:N, :] /= V[:, None]
    # Ghost cell gradients are not well-defined; leave as zero

    # --- Compute gradient ratio r for each face ---
    # For face f between O and N with face unit normal n_hat:
    #   dphi_f = phi_N - phi_O  (the face gradient * d_CF, unnormalized)
    #
    # When u > 0 (flow O -> N):
    #   "upstream gradient" at cell O projected onto face direction:
    #     grad_upstream = cell_grad[O] . (x_N - x_O) / |x_N - x_O|
    #   r = 2 * grad_upstream * d_CF / dphi_f - 1
    #
    # When u < 0 (flow N -> O):
    #   "upstream gradient" at cell N projected onto face direction:
    #     grad_upstream = cell_grad[N] . (x_N - x_O) / |x_N - x_O|
    #   r = 2 * grad_upstream * d_CF / dphi_f - 1

    # Face direction vector (O -> N), already captured by face_normals
    # But face_normals may have unit length; we need the actual O->N direction.
    # For the gradient ratio, we just need the dot product of cell_grad with
    # the face normal direction.

    dphi = phi_val[neighbor] - phi_val[owner]  # (nf,)
    dphi_safe = _fsign(dphi)

    # Project cell gradient onto face normal direction * d_CF
    # This gives the gradient at the cell center projected onto the face direction
    # times the face distance, giving an estimate of phi_N - phi_O from the cell's
    # perspective.
    grad_O_proj = np.sum(cell_grad[owner] * face_normals, axis=1) * d_CF  # (nf,)
    grad_N_proj = np.sum(cell_grad[neighbor] * face_normals, axis=1) * d_CF  # (nf,)

    # Gradient ratio for positive flow (upstream cell = O):
    # r_pos = 2 * grad_O_proj / dphi_safe - 1
    # This is: r = (phi_O - phi_OO) / (phi_N - phi_O) in the 1D uniform case
    r_pos = 2.0 * grad_O_proj / dphi_safe - 1.0

    # Gradient ratio for negative flow (upstream cell = N):
    # r_neg = 2 * grad_N_proj / dphi_safe - 1
    r_neg = 2.0 * grad_N_proj / dphi_safe - 1.0

    # Apply flux limiter
    psi_pos = 0.5 * FL(r_pos) * dphi  # correction when u > 0
    psi_neg = 0.5 * FL(r_neg) * dphi  # correction when u < 0
    # Note: for negative flow, the correction should use -(phi_O - phi_N)
    # but since we want the correction to be additive to the upwind flux,
    # we keep the same sign convention. The sign is handled by the velocity
    # splitting below.

    # --- Split flux by upwind direction ---
    F_pos = np.where(u_dir > 0, F, 0.0)  # flow O -> N
    F_neg = np.where(u_dir < 0, F, 0.0)  # flow N -> O

    # --- Compute the TVD correction per face ---
    # The anti-diffusive flux correction at each face:
    #   For u > 0: correction = F_pos * psi_pos  (adds to the upwind flux)
    #   For u < 0: correction = F_neg * psi_neg
    # Total face correction flux:
    face_correction = F_pos * psi_pos + F_neg * psi_neg  # (nf,)

    # --- Accumulate into RHS ---
    # The correction flux enters with the same sign convention as the
    # convection term itself:
    #   Owner row: -correction / V_O  (minus because the upwind matrix
    #              already captures the base flux, and the correction is the
    #              difference between TVD and upwind)
    #   Neighbor row: +correction / V_N
    RHS = np.zeros(N_total)
    corr_over_Vo = face_correction / V[owner]
    np.add.at(RHS, owner, -corr_over_Vo)

    is_internal = neighbor < N
    int_idx = np.where(is_internal)[0]
    corr_int = face_correction[int_idx]
    nbr_int = neighbor[int_idx]
    corr_over_Vn = corr_int / V[nbr_int]
    np.add.at(RHS, nbr_int, corr_over_Vn)

    # --- Build the upwind matrix (implicit part) ---
    M = convectionUpwindTerm(u, u_upwind)

    return M, RHS


# -----------------------------------------------------------------------
#  Public API convenience function (backward compatibility)
# -----------------------------------------------------------------------


def convectionTVDupwindRHSTerm(
    u: FaceVariable, phi: CellVariable, FL, u_upwind: FaceVariable = None
) -> np.ndarray:
    """Compute only the TVD correction RHS (without the upwind matrix).

    This is useful when the upwind matrix is built separately. Returns
    just the RHS correction vector.

    Parameters
    ----------
    u : FaceVariable
        Face-normal velocity.
    phi : CellVariable
        Current solution variable.
    FL : callable
        Flux limiter function.
    u_upwind : FaceVariable, optional
        Velocity for upwind direction determination.

    Returns
    -------
    RHS : ndarray, shape (N_total,)
        TVD correction vector (explicit part).
    """
    _, RHS = convectionTvdTerm(u, phi, FL, u_upwind)
    return RHS
