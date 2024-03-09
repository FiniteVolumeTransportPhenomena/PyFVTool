import numpy as np
from scipy.sparse import csr_array

from .mesh import Grid1D, Grid2D, Grid3D
from .cell import CellVariable
from .boundary import BoundaryConditions



def constantSourceTerm(gamma: CellVariable):
    """
    Constant source term in a PDE.
    
    The value of this source `gamma` may be different in each cell, but is 
    constant during the evolution of the PDE.
    
    Parameters
    ----------
    gamma: CellVariable
        Value of the source term
    
    Returns
    -------
    RHS: np.ndarray
        Right hand side of the source term
    
    Examples
    --------
    >>> import pyfvtool as pf
    >>> m = pf.Grid1D(10, 1.0)
    >>> gamma = pf.CellVariable(m, 1.0)
    >>> RHS = pf.constantSourceTerm(gamma)
    """
    m = gamma.domain
    if issubclass(type(m), Grid1D):
        Nx = m.dims[0]
        G = m.cell_numbers()
        row_index = G[1:Nx+1]  # main diagonal (only internal cells)
        RHS = np.zeros(Nx+2)
        RHS[row_index] = gamma.value[1:-1]
    elif issubclass(type(m), Grid2D):
        Nx, Ny = m.dims
        G = m.cell_numbers()
        # main diagonal (only internal cells)
        row_index = G[1:Nx+1, 1:Ny+1].ravel()
        RHS = np.zeros((Nx+2)*(Ny+2))
        RHS[row_index] = gamma.value[1:-1, 1:-1].ravel()
    elif issubclass(type(m), Grid3D):
        Nx, Ny, Nz = m.dims
        G = m.cell_numbers()
        # main diagonal (only internal cells)
        row_index = G[1:Nx+1, 1:Ny+1, 1:Nz+1].ravel()
        RHS = np.zeros((Nx+2)*(Ny+2)*(Nz+2))
        RHS[row_index] = gamma.value[1:-1, 1:-1, 1:-1].ravel()
    return RHS


def linearSourceTerm(beta: CellVariable):
    """
    Linear source term in a PDE.

    The linear source term takes the form `beta*phi` where `phi` is the
    solution variable and the factor `beta` a coefficient.

    Parameters
    ----------
    beta: CellVariable
        Multiplicative coefficient for the solution variable
    
    Returns
    -------
    RHS: np.ndarray
        Right hand side of the source term
    
    Examples
    --------
    >>> import pyfvtool as pf
    >>> m = pf.Grid1D(10, 1.0)
    >>> beta = pf.CellVariable(m, 1.0)
    >>> RHS = pf.linearSourceTerm(beta)
    """
    m = beta.domain
    if issubclass(type(m), Grid1D):
        Nx = m.dims[0]
        G = m.cell_numbers()
        AP_diag = beta.value[1:-1]
        row_index = G[1:Nx+1]  # main diagonal (only internal cells)
        return csr_array((AP_diag, (row_index, row_index)),
                         shape=((Nx+2), (Nx+2)))
    elif issubclass(type(m), Grid2D):
        Nx, Ny = m.dims
        G = m.cell_numbers()
        AP_diag = beta.value[1:-1, 1:-1].ravel()
        # main diagonal (only internal cells)
        row_index = G[1:Nx+1, 1:Ny+1].ravel()
        return csr_array((AP_diag, (row_index, row_index)),
                         shape=((Nx+2)*(Ny+2), (Nx+2)*(Ny+2)))
    elif issubclass(type(m), Grid3D):
        Nx, Ny, Nz = m.dims
        G = m.cell_numbers()
        AP_diag = beta.value[1:-1, 1:-1, 1:-1].ravel()
        # main diagonal (only internal cells)
        row_index = G[1:Nx+1, 1:Ny+1, 1:Nz+1].ravel()
        return csr_array((AP_diag, (row_index, row_index)),
                         shape=((Nx+2)*(Ny+2)*(Nz+2), (Nx+2)*(Ny+2)*(Nz+2)))


def transientTerm(phi_old: CellVariable, dt, alfa):
    """
    Transient term of a PDE.

    Parameters
    ----------
    phi_old: CellVariable
        Old value of the variable
    dt: float
        Time step size
    alfa: float or CellVariable
        Coefficient of the transient term
    
    Returns
    -------
    M: csr_array
        Matrix of the transient term
    RHS: np.ndarray
        Right hand side of the transient term
    
    Examples
    --------
    >>> import pyfvtool as pf
    >>> m = pf.Grid1D(10, 1.0)
    >>> phi_old = pf.CellVariable(m, 0.0)
    >>> M, RHS = pf.transientTerm(phi_old, 1.0, 1.0)
    """
    if not (type(alfa) is CellVariable):
        a = CellVariable(phi_old.domain, alfa,
                         BoundaryConditions(phi_old.domain))
    else:
        a = alfa
    return linearSourceTerm(a/dt), constantSourceTerm(a*phi_old/dt)

