import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.linalg import spsolve
from .mesh import *
from .utilities import *
from .cell import *
from .face import *
from .boundary import *


def solvePDE(m: MeshStructure, M:csr_array, RHS: np.ndarray,
             externalsolver = None) -> CellVariable:
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
    externalsolver: function (optional)
        If provided, use an external sparse solver via a function call
        having the same interface as the default solver
        scipy.sparse.linalg.spsolve.
    
    Returns
    -------
    phi: CellVariable
        Solution of the PDE
    """
    if externalsolver is None:
        solver = spsolve
    else:
        solver = externalsolver
    phi = solver(M, RHS)
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

#
#
# Below is an alternative (untested) implementation by gmweir that reproduces
# the original Matlab code.
#
# https://github.com/simulkade/PyFVTool/commit/6f3f7825ff4cb851dd9db84748a174dd6fe03661
#
# It uses the deprecated coding of the dimensionality (1, 1.5, 2, 2.5, 2.8)
# and should be rewritten and tested.
#
# I keep it here so that we can come back to this later if necessary.
#
#


# def solveExplicitPDE(phi_old: CellVariable, dt: float, RHS: np.ndarray, BC: BoundaryCondition, *args, **kwargs) -> CellVariable:
#     """
#     Solve a PDE (or system of PDEs) with an explicit discretization scheme (in time).
    
#         phi = solveExplicitPDE(phi_old, dt, RHS, BC, varargin)
    
#     SolveExplicitPDE solves for the new value of variable \phi in an explicit discretization scheme: 
#         \phi_new = \phi_old + dt/ * RHS
        
#         The real equation solved here is d\phi/dt = RHS

#     The code calculates the new values for the internal cells. Then it uses the boundary condition to calculate the values for the ghost cells.


#     Parameters
#     ----------
#     phi_old : {CellVariable object}
#         Solution variable defined at mesh-nodes
        
#     Returns
#     -------
#     out : {CellVariable object}
#         Updated solution variable defined at mesh-nodes with ghost cells
        

#     See Also
#     --------

#     Notes
#     -----

#     Examples
#     --------
    
#     """
#     # this is not the most beautiful implementation, but it works.
#     if len(args)>0:
#         x = phi_old.value[:] + dt*RHS/args[0].value[:]
#     else:
#         x = phi_old.value[:]+dt*RHS

#     d = phi_old.domain.dimension
#     N = phi_old.domain.dims

#     if (d>=2):
#         phi_val = x.reshape(N+2, order='F').copy()
#     else:
#         phi_val = x.reshape([N[0]+2, 1])


#     if (d == 1) or (d == 1.5):
#         phi_val = phi_val[1:N[0]] # (2:N(1)+1)
        
#     elif (d == 2) or (d == 2.5) or (d==2.8):
#         phi_val = phi_val[1:N[0], 1:N[0]]   # (2:N(1)+1, 2:N(2)+1)
        
#     elif (d == 3) or (d==3.2):
#         phi_val = phi_val[1:N[0], 1:N[1], 1:N[2]]  # (2:N(1)+1, 2:N(2)+1, 2:N(3)+1)

#     return createCellVariable(phi_old.domain, phi_val, BC)


