# Source terms — connectivity-based unified implementation
#
# With the new flat-array architecture, source terms operate on the
# interior-cell portion of the value arrays (indices 0..num_cells-1).
# Ghost-cell rows are left untouched (handled by boundaryConditionsTerm).

import numpy as np
from scipy.sparse import csr_array

from .cell import CellVariable


def constantSourceTerm(gamma: CellVariable) -> np.ndarray:
    """Constant source term in a PDE.

    The value of this source ``gamma`` may be different in each cell,
    but is constant during the evolution of the PDE.  Returns a RHS
    vector of shape ``(N_total,)`` with the source values in the
    interior-cell rows.

    Parameters
    ----------
    gamma : CellVariable
        Value of the source term.

    Returns
    -------
    RHS : ndarray, shape (N_total,)
        Right-hand side contribution.

    Examples
    --------
    >>> import pyfvtool as pf
    >>> m = pf.Grid1D(10, 1.0)
    >>> gamma = pf.CellVariable(m, 1.0)
    >>> RHS = pf.constantSourceTerm(gamma)
    """
    m = gamma.domain
    N = m.num_cells
    N_total = N + m.num_ghost_cells
    RHS = np.zeros(N_total)
    RHS[:N] = gamma.value  # .value returns _value[:num_cells]
    return RHS


def linearSourceTerm(beta: CellVariable) -> csr_array:
    """Linear source term in a PDE.

    The linear source term takes the form ``beta * phi`` where ``phi``
    is the solution variable and ``beta`` a coefficient.  Returns a
    sparse diagonal matrix of shape ``(N_total, N_total)`` with the
    ``beta`` values on the diagonal for interior cells.

    Parameters
    ----------
    beta : CellVariable
        Multiplicative coefficient for the solution variable.

    Returns
    -------
    M : csr_array, shape (N_total, N_total)
        Sparse matrix contribution.

    Examples
    --------
    >>> import pyfvtool as pf
    >>> m = pf.Grid1D(10, 1.0)
    >>> beta = pf.CellVariable(m, 1.0)
    >>> M = pf.linearSourceTerm(beta)
    """
    m = beta.domain
    N = m.num_cells
    N_total = N + m.num_ghost_cells
    diag_vals = beta.value  # shape (N,)
    row_idx = np.arange(N)
    return csr_array((diag_vals, (row_idx, row_idx)), shape=(N_total, N_total))


def transientTerm(phi_old: CellVariable, dt, alfa=1.0):
    """Transient (time-derivative) term of a PDE.

    Discretizes ``alfa * dphi/dt`` using backward Euler.  Returns a
    matrix and RHS vector such that the transient contribution is::

        M @ phi_new = RHS

    where ``M = diag(alfa / dt)`` and ``RHS = alfa * phi_old / dt``.

    Parameters
    ----------
    phi_old : CellVariable
        Solution at the previous time step.
    dt : float
        Time step size.
    alfa : float or CellVariable, optional
        Coefficient of the transient term (default 1.0).

    Returns
    -------
    M : csr_array, shape (N_total, N_total)
        Sparse matrix contribution.
    RHS : ndarray, shape (N_total,)
        Right-hand side contribution.

    Examples
    --------
    >>> import pyfvtool as pf
    >>> m = pf.Grid1D(10, 1.0)
    >>> phi_old = pf.CellVariable(m, 0.0)
    >>> M, RHS = pf.transientTerm(phi_old, 1.0, 1.0)
    """
    if not isinstance(alfa, CellVariable):
        a = CellVariable(phi_old.domain, alfa)
    else:
        a = alfa
    M = linearSourceTerm(a / dt)
    RHS = constantSourceTerm(a * phi_old / dt)
    return M, RHS
