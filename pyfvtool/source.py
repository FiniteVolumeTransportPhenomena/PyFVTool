
import numpy as np
from scipy.sparse import csr_array
from .mesh import *
from .utilities import *
from .cell import *
from .face import *


def constantSourceTerm(phi0: CellVariable):
    """
    Constant source term of a PDE.

    Parameters
    ----------
    phi0: CellVariable
        Value of the source term
    
    Returns
    -------
    RHS: np.ndarray
        Right hand side of the source term
    
    Examples
    --------
    >>> from pyfvtool import *
    >>> m = createMesh1D(10, 1.0)
    >>> phi0 = createCellVariable(m, 1.0)
    >>> RHS = constantSourceTerm(phi0)
    """
    m = phi0.domain
    if issubclass(type(m), Mesh1D):
        Nx = m.dims[0]
        G = m.cell_numbers()
        row_index = G[1:Nx+1]  # main diagonal (only internal cells)
        RHS = np.zeros(Nx+2)
        RHS[row_index] = phi0.value[1:-1]
    elif issubclass(type(m), Mesh2D):
        Nx, Ny = m.dims
        G = m.cell_numbers()
        # main diagonal (only internal cells)
        row_index = G[1:Nx+1, 1:Ny+1].ravel()
        RHS = np.zeros((Nx+2)*(Ny+2))
        RHS[row_index] = phi0.value[1:-1, 1:-1].ravel()
    elif issubclass(type(m), Mesh3D):
        Nx, Ny, Nz = m.dims
        G = m.cell_numbers()
        # main diagonal (only internal cells)
        row_index = G[1:Nx+1, 1:Ny+1, 1:Nz+1].ravel()
        RHS = np.zeros((Nx+2)*(Ny+2)*(Nz+2))
        RHS[row_index] = phi0.value[1:-1, 1:-1, 1:-1].ravel()
    return RHS


def linearSourceTerm(betta0: CellVariable):
    """
    Linear source term of a PDE.

    Parameters
    ----------
    betta0: CellVariable
        Value of the source term
    
    Returns
    -------
    RHS: np.ndarray
        Right hand side of the source term
    
    Examples
    --------
    >>> from pyfvtool import *
    >>> m = createMesh1D(10, 1.0)
    >>> betta0 = createCellVariable(m, 1.0)
    >>> RHS = linearSourceTerm(betta0)
    """
    m = betta0.domain
    if issubclass(type(m), Mesh1D):
        Nx = m.dims[0]
        G = m.cell_numbers()
        AP_diag = betta0.value[1:-1]
        row_index = G[1:Nx+1]  # main diagonal (only internal cells)
        return csr_array((AP_diag, (row_index, row_index)),
                         shape=((Nx+2), (Nx+2)))
    elif issubclass(type(m), Mesh2D):
        Nx, Ny = m.dims
        G = m.cell_numbers()
        AP_diag = betta0.value[1:-1, 1:-1].ravel()
        # main diagonal (only internal cells)
        row_index = G[1:Nx+1, 1:Ny+1].ravel()
        return csr_array((AP_diag, (row_index, row_index)),
                         shape=((Nx+2)*(Ny+2), (Nx+2)*(Ny+2)))
    elif issubclass(type(m), Mesh3D):
        Nx, Ny, Nz = m.dims
        G = m.cell_numbers()
        AP_diag = betta0.value[1:-1, 1:-1, 1:-1].ravel()
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
    >>> from pyfvtool import *
    >>> m = createMesh1D(10, 1.0)
    >>> phi_old = createCellVariable(m, 0.0)
    >>> M, RHS = transientTerm(phi_old, 1.0, 1.0)
    """
    if not (type(alfa) is CellVariable):
        a = createCellVariable(phi_old.domain, alfa, createBC(phi_old.domain))
    else:
        a = alfa
    return linearSourceTerm(a/dt), constantSourceTerm(a*phi_old/dt)