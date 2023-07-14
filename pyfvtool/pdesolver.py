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