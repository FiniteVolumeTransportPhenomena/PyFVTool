"""
Boundary condition classes
"""

import numpy as np
from scipy.sparse import csr_array
from mesh import *
from utilities import *

class Boundary:
    def __init__(self, a:np.ndarray, b:np.ndarray, c:np.ndarray, periodic=False):
        self.a = a
        self.b = b
        self.c = c
        self.periodic = periodic
    def __str__(self):
        temp = vars(self)
        for item in temp:
            print(item, ':', temp[item])
        return ""
    def __repr__(self):
        temp = vars(self)
        for item in temp:
            print(item, ':', temp[item])
        return ""


class BoundaryCondition:
    def __init__(self, mesh: MeshStructure, 
                 left: Boundary, right: Boundary, 
                 bottom: Boundary, top: Boundary, 
                 back:Boundary, front:Boundary):
        self.domain = mesh
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.back = back
        self.front = front
    def __str__(self):
        temp = vars(self)
        for item in temp:
            print(item, ':', temp[item])
        return ""
    def __repr__(self):
        temp = vars(self)
        for item in temp:
            print(item, ':', temp[item])
        return ""

class BoundaryCondition1D(BoundaryCondition):
    def __init__(self, mesh:Mesh1D):
        left = Boundary(np.array([1.0]), np.array([0.0]), np.array([0.0]))
        right = Boundary(np.array([1.0]), np.array([0.0]), np.array([0.0]))
        bottom = Boundary(np.array([]), np.array([]), np.array([]))
        top = Boundary(np.array([]), np.array([]), np.array([]))
        back = Boundary(np.array([]), np.array([]), np.array([]))
        front = Boundary(np.array([]), np.array([]), np.array([]))
        BoundaryCondition.__init__(self, mesh, left, right, bottom, top, back, front)

class BoundaryCondition2D(BoundaryCondition):
    def __init__(self, mesh: Mesh2D):
        Nx, Ny = mesh.dims
        left = Boundary(np.ones(1,Ny), np.zeros(1,Ny), np.zeros(1,Ny))
        right = Boundary(np.ones(1,Ny), np.zeros(1,Ny), np.zeros(1,Ny))
        bottom = Boundary(np.ones(Nx,1), np.zeros(Nx,1), np.zeros(Nx,1))
        top = Boundary(np.ones(Nx,1), np.zeros(Nx,1), np.zeros(Nx,1))
        back = Boundary(np.array([]), np.array([]), np.array([]))
        front = Boundary(np.array([]), np.array([]), np.array([]))
        BoundaryCondition.__init__(mesh, left, right, bottom, top, back, front)

class BoundaryCondition3D(BoundaryCondition):
    def __init__(self, mesh: Mesh3D):
        Nx, Ny, Nz = mesh.dims
        left = Boundary(np.ones(Ny, Nz), np.zeros(Ny, Nz), np.zeros(Ny, Nz))
        right = Boundary(np.ones(Ny, Nz), np.zeros(Ny, Nz), np.zeros(Ny, Nz))
        bottom = Boundary(np.ones(Nx, Nz), np.zeros(Nx, Nz), np.zeros(Nx, Nz))
        top = Boundary(np.ones(Nx, Nz), np.zeros(Nx, Nz), np.zeros(Nx, Nz))
        back = Boundary(np.ones(Nx, Ny), np.zeros(Nx, Ny), np.zeros(Nx, Ny))
        front = Boundary(np.ones(Nx, Ny), np.zeros(Nx, Ny), np.zeros(Nx, Ny))
        BoundaryCondition.__init__(mesh, left, right, bottom, top, back, front)

def createBC(mesh: MeshStructure):
    if issubclass(type(mesh), Mesh1D):
        return BoundaryCondition1D(mesh)
    elif issubclass(type(mesh), Mesh2D):
        return BoundaryCondition2D(mesh)
    elif issubclass(type(mesh), Mesh3D):
        return BoundaryCondition3D(mesh)

"""
Discretizing boundary conditions to csr array
Example:
row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
csr_array((data, (row, col)), shape=(3, 3)).toarray()
array([[1, 0, 2],
       [0, 0, 3],
       [4, 5, 6]])
"""

def boundaryCondition1D(BC: BoundaryCondition1D):
    Nx = BC.domain.dims[0]
    dx_1 = BC.domain.cellsize.x[1]
    dx_end = BC.domain.cellsize.x[-1]
    G = int_range(1, Nx+2)-1

    nb = 8 # number of boundary nodes
    ii = np.zeros(nb, dtype=int)
    jj = np.zeros(nb, dtype=int)
    s = np.zeros(nb, dtype=float)

    BCRHS = np.zeros(Nx+2) # RHS vector
    q = -1 # a counter in case I'm wrong with my count of nonzero elements

    if (not BC.left.periodic) or (not BC.right.periodic):
        # right boundary
        i = Nx+1 # -1 for python 0-based indexing
        q+=1
        ii[q] = G[i]
        jj[q] = G[i]  
        s[q] = BC.right.b/2 + BC.right.a/dx_end
        q=q+1
        ii[q] = G[i]  
        jj[q] = G[i-1]
        s[q] = BC.right.b/2 - BC.right.a/dx_end
        BCRHS[G[i]] = BC.right.c

        # Left boundary
        i = 0
        q+=1
        ii[q] = G[i]
        jj[q] = G[i+1]
        s[q] = -(BC.left.b/2 + BC.left.a/dx_1)
        q=q+1
        ii[q] = G[i]
        jj[q] = G[i]
        s[q] = -(BC.left.b/2 - BC.left.a/dx_1)
        BCRHS[G[i]] = -BC.left.c
    elif BC.right.periodic or BC.left.periodic: # periodic boundary condition
        # Right boundary
        i = Nx+1
        q=q+1
        ii[q] = G[i]  
        jj[q] = G[i]  
        s[q] = 1
        q=q+1
        ii[q] = G[i]  
        jj[q] = G[i-1]
        s[q] = -1
        q=q+1
        ii[q] = G[i]  
        jj[q] = G[0] 
        s[q] = dx_end/dx_1
        q=q+1
        ii[q] = G[i]  
        jj[q] = G[1]
        s[q] = -dx_end/dx_1
        BCRHS[G[i]] = 0
        # Left boundary
        i = 0
        q=q+1
        ii[q] = G[i]  
        jj[q] = G[i]  
        s[q] = 1.0
        q=q+1
        ii[q] = G[i]  
        jj[q] = G[1]  
        s[q] = 1.0
        q=q+1
        ii[q] = G[i]  
        jj[q] = G[Nx] 
        s[q] = -1.0
        q=q+1
        ii[q] = G[i]  
        jj[q] = G[Nx+1] 
        s[q] = -1.0
        BCRHS[G[i]] = 0
    # Build the sparse matrix of the boundary conditions
    q+=1
    BCMatrix = csr_array((s[0:q], (ii[0:q], jj[0:q])), shape=(Nx+2, Nx+2))
    return BCMatrix, BCRHS