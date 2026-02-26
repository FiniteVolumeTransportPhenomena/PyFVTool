# PDE solvers — connectivity-based unified implementation
#
# Three solver entry points:
#
#   solvePDE          — high-level solver that auto-assembles BC terms
#   solveMatrixPDE    — expert-level solver using pre-assembled M and RHS
#   solveExplicitPDE  — explicit (forward Euler) time stepping
#
# The sign convention for ``eqnterms`` follows the original PyFVTool
# (MATLAB FVTool compatibility):
#
#   ``diffusionTerm(D)`` returns a matrix encoding ``div(D grad(phi))``
#   (positive Laplacian — positive off-diag, negative diag).  Users
#   negate it to set up ``-div(D grad(phi)) = S``::
#
#       eqnterms = [-diffusionTerm(D), constantSourceTerm(S)]
#
#   ``solvePDE`` sums all (M, RHS) contributions and the BC term, then
#   solves ``M_total @ phi = RHS_total``.

import numpy as np

from scipy.sparse import csr_array
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import use_solver

use_solver(useUmfpack=False)
# For reproducibility, do not automatically use any installed `scikits.umfpack`
# solver. Always use the built-in SuperLU by default. In PyFVTool, the
# `scikits.umfpack.spsolve` solver (if installed) should be supplied via
# the `externalsolver` keyword argument of `solvePDE`, as is the case for
# the `pypardiso.spsolve` solver (and any other external solvers).

from .mesh import MeshStructure
from .cell import CellVariable
from .boundary import BoundaryConditions, boundaryConditionsTerm, apply_BCs
from .utilities import TrackedArray


def solvePDE(
    phi: CellVariable, BC_or_eqnterms=None, eqnterms=None, externalsolver=None
) -> CellVariable:
    """
    Solve a PDE using the finite volume method.

    Supports two calling conventions:

    - **New API**: ``solvePDE(phi, BC, eqnterms)``
    - **Legacy API**: ``solvePDE(phi, eqnterms)`` — BCs are taken from
      ``phi.BCs``.

    Constructs the matrix equation from the FVM-discretised terms and
    the boundary conditions, then solves the resulting linear system.
    The terms are provided by the user as a list, each term being the
    output of a prior call to the appropriate ``xxxTerm()`` function.

    The terms provided by the user should *not* include any terms related
    to the boundary conditions — those are handled automatically.

    The CellVariable ``phi`` is updated in-place with the solution values,
    and a reference to it is returned.  If the original values of ``phi``
    must be preserved, copy it beforehand.

    Parameters
    ----------
    phi : CellVariable
        Solution variable (initial guess / previous time-step value).
    BC_or_eqnterms : BoundaryConditions or list
        Either a BoundaryConditions object (new API) or a list of
        equation terms (legacy API).
    eqnterms : list, optional
        List of equation terms (new API only).
    externalsolver : callable, optional
        If specified, use an external sparse solver with the same
        interface as ``scipy.sparse.linalg.spsolve``.

    Returns
    -------
    CellVariable
        The updated ``phi``.

    Examples
    --------
    >>> import pyfvtool as pf
    >>> m = pf.Grid1D(10, 1.0)
    >>> BC = pf.BoundaryConditions(m)
    >>> BC.left.a[:] = 0; BC.left.b[:] = 1; BC.left.c[:] = 0.0
    >>> BC.right.a[:] = 0; BC.right.b[:] = 1; BC.right.c[:] = 1.0
    >>> D = pf.FaceVariable(m, 1.0)
    >>> phi = pf.CellVariable(m)
    >>> phi = pf.solvePDE(phi, BC, [-pf.diffusionTerm(D)])
    """
    # Detect calling convention
    if isinstance(BC_or_eqnterms, list):
        # Legacy: solvePDE(phi, eqnterms)
        BC = phi.BCs
        terms = BC_or_eqnterms
    elif isinstance(BC_or_eqnterms, BoundaryConditions):
        # New API: solvePDE(phi, BC, eqnterms)
        BC = BC_or_eqnterms
        terms = eqnterms if eqnterms is not None else []
    elif BC_or_eqnterms is None and eqnterms is not None:
        # Keyword-only: solvePDE(phi, eqnterms=eqnterms)
        BC = phi.BCs
        terms = eqnterms
    else:
        raise TypeError(
            "solvePDE: expected solvePDE(phi, BC, eqnterms) or solvePDE(phi, eqnterms)"
        )
    if externalsolver is None:
        solver = spsolve
    else:
        solver = externalsolver

    # Build BC matrix and RHS
    Mbc, RHSbc = boundaryConditionsTerm(BC)

    # Initialise overall system from BCs
    M = Mbc.copy()
    RHS = RHSbc.copy()

    # Accumulate all equation terms
    for term in terms:
        if isinstance(term, tuple):
            Mterm, RHSterm = term
            if (Mterm.ndim != 2) or (RHSterm.ndim != 1):
                raise TypeError("Unknown term")
            M += Mterm
            RHS += RHSterm
        elif hasattr(term, "ndim"):
            if term.ndim == 1:
                RHS += term
            elif term.ndim == 2:
                M += term
            else:
                raise TypeError("Unknown term")
        else:
            raise TypeError("Unknown term")

    # Solve the linear system
    phi_new_values = solver(M, RHS)

    # Update phi in-place (flat array)
    phi._value = TrackedArray(phi_new_values)
    phi._value.modified = False

    # Recompute ghost cells from BCs
    apply_BCs(phi, BC)

    return phi


def solveMatrixPDE(
    m: MeshStructure, M: csr_array, RHS: np.ndarray, externalsolver=None
) -> CellVariable:
    """
    Solve the PDE from a pre-assembled matrix system.

    This 'expert-level' solver uses the ``M`` matrix and ``RHS`` vector
    directly.  These should be constructed beforehand by combining all
    discretisation terms, **including** the boundary condition terms.

    Returns a new CellVariable; the input is not modified.

    Parameters
    ----------
    m : MeshStructure
        Mesh structure.
    M : csr_array
        Matrix of the linear system.
    RHS : ndarray
        Right-hand side of the linear system.
    externalsolver : callable, optional
        If provided, use an external sparse solver with the same
        interface as ``scipy.sparse.linalg.spsolve``.

    Returns
    -------
    CellVariable
        Solution of the PDE (newly created).
    """
    if externalsolver is None:
        solver = spsolve
    else:
        solver = externalsolver

    phi_vals = solver(M, RHS)
    return CellVariable(m, phi_vals)  # flat array of size N_total


def solveExplicitPDE(
    phi_old: CellVariable, BC_or_dt=None, dt_or_RHS=None, RHS=None
) -> CellVariable:
    """
    Solve the PDE using an explicit (forward Euler) finite volume method.

    Supports two calling conventions:

    - **New API**: ``solveExplicitPDE(phi_old, BC, dt, RHS)``
    - **Legacy API**: ``solveExplicitPDE(phi_old, dt, RHS)`` — BCs are
      taken from ``phi_old.BCs``.

    Parameters
    ----------
    phi_old : CellVariable
        Solution at the previous time step.
    BC_or_dt : BoundaryConditions or float
        Either a BoundaryConditions object (new API) or the time step
        (legacy API).
    dt_or_RHS : float or np.ndarray
        Either dt (new API) or RHS (legacy API).
    RHS : np.ndarray, optional
        Right-hand side vector (new API only).

    Returns
    -------
    CellVariable
        Solution at the new time step (newly created).
    """
    if isinstance(BC_or_dt, BoundaryConditions):
        # New API: solveExplicitPDE(phi_old, BC, dt, RHS)
        BC = BC_or_dt
        dt = dt_or_RHS
        rhs = RHS
    elif isinstance(BC_or_dt, (int, float, np.floating)):
        # Legacy: solveExplicitPDE(phi_old, dt, RHS)
        BC = phi_old.BCs
        dt = BC_or_dt
        rhs = dt_or_RHS
    else:
        raise TypeError(
            "solveExplicitPDE: expected solveExplicitPDE(phi_old, BC, dt, RHS) "
            "or solveExplicitPDE(phi_old, dt, RHS)"
        )

    # Apply BCs to phi_old first
    apply_BCs(phi_old, BC)

    # Forward Euler step
    x = phi_old._value + dt * rhs
    phi = CellVariable(phi_old.domain, x)

    # Update ghost cells from BCs
    apply_BCs(phi, BC)

    return phi
