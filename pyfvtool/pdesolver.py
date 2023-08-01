import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.linalg import spsolve
from .mesh import *
from .utilities import *
from .cell import *
from .face import *



def solvePDE(m: MeshStructure, M:csr_array, RHS: np.ndarray) -> CellVariable:
    """
    Solve the PDE using the finite volume method.
    
    Parameters
    ----------
    m: MeshStructure
        Mesh structure
    M: csr_array
        Matrix of the linear system
    RHS: np.ndarray
        Right hand side of the linear system
    
    Returns
    -------
    phi: CellVariable
        Solution of the PDE
    """
    phi = spsolve(M, RHS)
    return CellVariable(m, np.reshape(phi, m.dims+2))

def solveExplicitPDE(phi_old: CellVariable, dt: float, RHS: np.ndarray, BC: BoundaryCondition) -> CellVariable:
    """
    Solve the PDE using the finite volume method.

    Parameters
    ----------
    phi_old: CellVariable
        Solution of the previous time step
    dt: float
        Time step
    RHS: np.ndarray
        Right hand side of the linear system
    BC: BoundaryCondition
        Boundary condition
    
    Returns
    -------
    phi: CellVariable
        Solution of the PDE
    """
    

    x = phi_old.value + dt*RHS.reshape(phi_old.value.shape)
    phi= createCellVariable(phi_old.domain, 0.0)
    phi.value = x
    phi.update_bc_cells(BC)
    return phi