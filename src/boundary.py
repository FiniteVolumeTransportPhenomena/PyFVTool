"""
Boundary condition classes
"""

import numpy as np
from scipy.sparse import csr_array
from mesh import *
from utilities import *


class Boundary:
    def __init__(self, a: np.ndarray, b: np.ndarray, c: np.ndarray, periodic=False):
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
                 back: Boundary, front: Boundary):
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
    def __init__(self, mesh: Mesh1D):
        left = Boundary(np.array([1.0]), np.array([0.0]), np.array([0.0]))
        right = Boundary(np.array([1.0]), np.array([0.0]), np.array([0.0]))
        bottom = Boundary(np.array([]), np.array([]), np.array([]))
        top = Boundary(np.array([]), np.array([]), np.array([]))
        back = Boundary(np.array([]), np.array([]), np.array([]))
        front = Boundary(np.array([]), np.array([]), np.array([]))
        BoundaryCondition.__init__(
            self, mesh, left, right, bottom, top, back, front)


class BoundaryCondition2D(BoundaryCondition):
    def __init__(self, mesh: Mesh2D):
        Nx, Ny = mesh.dims
        left = Boundary(np.ones((1, Ny)), np.zeros((1, Ny)), np.zeros((1, Ny)))
        right = Boundary(np.ones((1, Ny)), np.zeros((1, Ny)), np.zeros((1, Ny)))
        bottom = Boundary(np.ones((Nx, 1)), np.zeros((Nx, 1)), np.zeros((Nx, 1)))
        top = Boundary(np.ones((Nx, 1)), np.zeros((Nx, 1)), np.zeros((Nx, 1)))
        back = Boundary(np.array([]), np.array([]), np.array([]))
        front = Boundary(np.array([]), np.array([]), np.array([]))
        BoundaryCondition.__init__(self, mesh, left, right, bottom, top, back, front)


class BoundaryCondition3D(BoundaryCondition):
    def __init__(self, mesh: Mesh3D):
        Nx, Ny, Nz = mesh.dims
        left = Boundary(np.ones((Ny, Nz)), np.zeros((Ny, Nz)), np.zeros((Ny, Nz)))
        right = Boundary(np.ones((Ny, Nz)), np.zeros((Ny, Nz)), np.zeros((Ny, Nz)))
        bottom = Boundary(np.ones((Nx, Nz)), np.zeros((Nx, Nz)), np.zeros((Nx, Nz)))
        top = Boundary(np.ones((Nx, Nz)), np.zeros((Nx, Nz)), np.zeros((Nx, Nz)))
        back = Boundary(np.ones((Nx, Ny)), np.zeros((Nx, Ny)), np.zeros((Nx, Ny)))
        front = Boundary(np.ones((Nx, Ny)), np.zeros((Nx, Ny)), np.zeros((Nx, Ny)))
        BoundaryCondition.__init__(self, mesh, left, right, bottom, top, back, front)


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
    dx_1 = BC.domain.cellsize.x[0]
    dx_end = BC.domain.cellsize.x[-1]
    G = int_range(1, Nx+2)-1

    nb = 8  # number of boundary nodes
    ii = np.zeros(nb, dtype=int)
    jj = np.zeros(nb, dtype=int)
    s = np.zeros(nb, dtype=float)

    BCRHS = np.zeros(Nx+2)  # RHS vector
    q = -1  # a counter in case I'm wrong with my count of nonzero elements

    if (not BC.left.periodic) and (not BC.right.periodic):
        # right boundary
        i = Nx+1  # -1 for python 0-based indexing
        q += 1
        ii[q] = G[i]
        jj[q] = G[i]
        s[q] = BC.right.b/2 + BC.right.a/dx_end
        q = q+1
        ii[q] = G[i]
        jj[q] = G[i-1]
        s[q] = BC.right.b/2 - BC.right.a/dx_end
        BCRHS[G[i]] = BC.right.c

        # Left boundary
        i = 0
        q += 1
        ii[q] = G[i]
        jj[q] = G[i+1]
        s[q] = -(BC.left.b/2 + BC.left.a/dx_1)
        q = q+1
        ii[q] = G[i]
        jj[q] = G[i]
        s[q] = -(BC.left.b/2 - BC.left.a/dx_1)
        BCRHS[G[i]] = -BC.left.c
    elif BC.right.periodic or BC.left.periodic:  # periodic boundary condition
        # Right boundary
        i = Nx+1
        q = q+1
        ii[q] = G[i]
        jj[q] = G[i]
        s[q] = 1
        q = q+1
        ii[q] = G[i]
        jj[q] = G[i-1]
        s[q] = -1
        q = q+1
        ii[q] = G[i]
        jj[q] = G[0]
        s[q] = dx_end/dx_1
        q = q+1
        ii[q] = G[i]
        jj[q] = G[1]
        s[q] = -dx_end/dx_1
        BCRHS[G[i]] = 0
        # Left boundary
        i = 0
        q = q+1
        ii[q] = G[i]
        jj[q] = G[i]
        s[q] = 1.0
        q = q+1
        ii[q] = G[i]
        jj[q] = G[1]
        s[q] = 1.0
        q = q+1
        ii[q] = G[i]
        jj[q] = G[Nx]
        s[q] = -1.0
        q = q+1
        ii[q] = G[i]
        jj[q] = G[Nx+1]
        s[q] = -1.0
        BCRHS[G[i]] = 0
    # Build the sparse matrix of the boundary conditions
    q += 1
    BCMatrix = csr_array((s[0:q], (ii[0:q], jj[0:q])), shape=(Nx+2, Nx+2))
    return BCMatrix, BCRHS


def boundaryCondition2D(BC: BoundaryCondition1D):
    Nx, Ny = BC.domain.dims
    dx_1 = BC.domain.cellsize.x[0]
    dx_end = BC.domain.cellsize.x[-1]
    dy_1 = BC.domain.cellsize.y[0]
    dy_end = BC.domain.cellsize.y[-1]
    G = BC.domain.cell_numbers()

    nb = 8*(Nx+Ny+2)  # number of boundary nodes
    ii = np.zeros(nb, dtype=int)
    jj = np.zeros(nb, dtype=int)
    s = np.zeros(nb, dtype=float)

    BCRHS = np.zeros((Nx+2)*(Ny+2))  # RHS vector
    q = -1  # a counter in case I'm wrong with my count of nonzero elements
    #  assign value to the corner nodes (useless cells)
    q = int_range(0, 3)
    ii[q] = BC.domain.corners
    jj[q] = BC.domain.corners
    s[q] = np.max(BC.top.b/2 + BC.top.a/dy_end)
    BCRHS[BC.domain.corners] = 0.0

    if (not BC.top.periodic) and (not BC.bottom.periodic):
        # top boundary
        j=Ny+1
        i=int_range(1,Nx)
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,j]  
        s[q] = (BC.top.b/2 + BC.top.a/dy_end).squeeze()
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,j-1] 
        s[q] = (BC.top.b/2 - BC.top.a/dy_end).squeeze()
        BCRHS[G[i,j]] = (BC.top.c).squeeze()

        # Bottom boundary
        j=0
        # i=1:Nx already defined
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,j+1]  
        s[q] = -(BC.bottom.b/2 + BC.bottom.a/dy_1).squeeze()
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,j] 
        s[q] = -(BC.bottom.b/2 - BC.bottom.a/dy_1).squeeze()
        BCRHS[G[i,j]] = (-BC.bottom.c).squeeze()
    elif BC.top.periodic or BC.bottom.periodic: # periodic boundary
        # top boundary
        j=Ny+1
        # i=int_range(1,Nx)
        i=int_range(1,Nx)
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,j]  
        s[q] = 1
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,j-1]  
        s[q] = -1
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,0] 
        s[q] = dy_end/dy_1
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,1] 
        s[q] = -dy_end/dy_1
        BCRHS[G[i,j]] = 0

        # Bottom boundary
        j=0
        # i=int_range(1,Nx)
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,j]  
        s[q] = 1
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,j+1]  
        s[q] = 1
        q = q[-1]+i
        ii[q] = G[i,j]
        jj[q] = G[i,Ny+1] 
        s[q] = -1
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,Ny+2]
        s[q] = -1
        BCRHS[G[i,j]] = 0

    if (not BC.left.periodic) and (not BC.right.periodic):
        # right boundary
        i = Nx+1  # -1 for python 0-based indexing
        j = int_range(1, Ny)
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[i,j]
        s[q] = BC.right.b/2 + BC.right.a/dx_end
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[i-1,j]
        s[q] = BC.right.b/2 - BC.right.a/dx_end
        BCRHS[G[i,j]] = BC.right.c

        # Left boundary
        i = 0
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[i+1,j]
        s[q] = -(BC.left.b/2 + BC.left.a/dx_1)
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[i,j]
        s[q] = -(BC.left.b/2 - BC.left.a/dx_1)
        BCRHS[G[i,j]] = -BC.left.c
    elif BC.right.periodic or BC.left.periodic:  # periodic boundary condition
        # Right boundary
        i = Nx+1
        j = int_range(1, Ny)
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[i,j]
        s[q] = 1
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[i-1,j]
        s[q] = -1
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[0,j]
        s[q] = dx_end/dx_1
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[1,j]
        s[q] = -dx_end/dx_1
        BCRHS[G[i,j]] = 0
        # Left boundary
        i = 0
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[i,j]
        s[q] = 1.0
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[i+1,j]
        s[q] = 1.0
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[Nx,j]
        s[q] = -1.0
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[Nx+1,j]
        s[q] = -1.0
        BCRHS[G[i,j]] = 0.0
    # Build the sparse matrix of the boundary conditions
    q = q[-1] + 1
    BCMatrix = csr_array((s[0:q], (ii[0:q], jj[0:q])), 
                         shape=((Nx+2)*(Ny+2), (Nx+2)*(Ny+2)))
    return BCMatrix, BCRHS

def boundaryCondition3D(BC: BoundaryCondition3D):
    # extract data from the mesh structure
    Nx, Ny, Nz = BC.domain.dims
    G=BC.domain.cell_numbers()
    dx_1 = BC.domain.cellsize.x[0]
    dx_end = BC.domain.cellsize.x[-1]
    dy_1 = BC.domain.cellsize.y[0]
    dy_end = BC.domain.cellsize.y[-1]
    dz_1 = BC.domain.cellsize.z[0]
    dz_end = BC.domain.cellsize.z[-1]

    i_ind = int_range(1,Nx)[:, np.newaxis, np.newaxis]
    j_ind = int_range(1,Ny)[np.newaxis, :, np.newaxis]
    k_ind = int_range(1,Nz)[np.newaxis, np.newaxis, :]
    # number of boundary nodes (axact number is 2[(m+1)(n+1)*(n+1)*(p+1)+(m+1)*p+1]:
    nb = 8*((Nx+1)*(Ny+1)+(Nx+1)*(Nz+1)+(Ny+1)*(Nz+1))

    # define the vectors to be used for the creation of the sparse matrix
    ii = np.zeros(nb, dtype=int)
    jj = np.zeros(nb, dtype=int)
    s = np.zeros(nb, dtype=float)

    # define the RHS column vector
    BCRHS = np.zeros((Nx+2)*(Ny+2)*(Nz+2))

    # assign value to the corner nodes (useless cells)
    q = int_range(0, 7)
    ii[q] = BC.domain.corners 
    jj[q] = BC.domain.corners
    s[q] = 1.0
    BCRHS[BC.domain.corners] = 0.0

    # assign values to the edges (useless cells)
    q = q[-1]+int_range(1, np.size(BC.domain.edges))
    ii[q] = BC.domain.edges 
    jj[q] = BC.domain.edges
    s[q] = 1.0
    BCRHS[BC.domain.edges] = 0.0

    # Assign values to the boundary condition matrix and the RHS vector based
    # on the BC structure
    if (not BC.top.periodic) and (not BC.bottom.periodic):
        # top boundary
        j=Ny+1
        i = i_ind
        k = k_ind
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = (BC.top.b/2 + BC.top.a/dy_end).ravel()
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j-1,k].ravel()
        s[q] = (BC.top.b/2 - BC.top.a/dy_end).ravel()
        BCRHS[G[i,j,k].ravel()] = (BC.top.c).ravel()

        # Bottom boundary
        j=0
        i=i_ind
        k=k_ind
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j+1,k].ravel()
        s[q] = -(BC.bottom.b/2 + BC.bottom.a/dy_1).ravel()
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = -(BC.bottom.b/2 - BC.bottom.a/dy_1).ravel()
        BCRHS[G[i,j,k].ravel()] = -(BC.bottom.c).ravel()
    elif BC.top.periodic or BC.bottom.periodic: # periodic
        # top boundary
        j=Ny+1
        i=i_ind
        k=k_ind
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = 1.0
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j-1,k].ravel()
        s[q] = -1.0
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,0,k].ravel()
        s[q] = dy_end/dy_1
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,1,k].ravel()
        s[q] = -dy_end/dy_1
        BCRHS[G[i,j,k].ravel()] = 0.0

        # Bottom boundary
        j=0
        i=i_ind
        k=k_ind
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = 1.0
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j+1,k].ravel()
        s[q] = 1.0
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,Ny,k].ravel()
        s[q] = -1.0
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,Ny+1,k].ravel()
        s[q] = -1.0
        BCRHS[G[i,j,k].ravel()] = 0.0
    if (not BC.right.periodic) and (not BC.left.periodic):
        # Right boundary
        i=Nx+1
        j=j_ind
        k=k_ind
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = (BC.right.b/2 + BC.right.a/dx_end).ravel()
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i-1,j,k].ravel()
        s[q] = (BC.right.b/2 - BC.right.a/dx_end).ravel()
        BCRHS[G[i,j,k].ravel()] = (BC.right.c).ravel()

        # Left boundary
        i = 0
        j=j_ind
        k=k_ind
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i+1,j,k].ravel()
        s[q] = -(BC.left.b/2 + BC.left.a/dx_1).ravel()
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = -(BC.left.b/2 - BC.left.a/dx_1).ravel()
        BCRHS[G[i,j,k].ravel()] = -(BC.left.c).ravel()
    elif BC.right.periodic or BC.left.periodic: # periodic
        # Right boundary
        i=Nx+1
        j=j_ind
        k=k_ind
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = 1.0
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i-1,j,k].ravel()
        s[q] = -1.0
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[0,j,k].ravel()
        s[q] = dx_end/dx_1
        q = q[-1]+int_range[1,Ny*Nz]
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[1,j,k].ravel()
        s[q] = -dx_end/dx_1
        BCRHS[G[i,j,k].ravel()] = 0.0

        # Left boundary
        i = 0
        j=j_ind
        k=k_ind
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = 1.0
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i+1,j,k].ravel()
        s[q] = 1.0
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[Nx,j,k].ravel()
        s[q] = -1.0
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[Nx+1,j,k].ravel()
        s[q] = -1.0
        BCRHS[G[i,j,k].ravel()] = 0.0
    if (not BC.front.periodic) and (not BC.back.periodic):
        # Front boundary
        k=Nz+1
        i = i_ind
        j = j_ind
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = (BC.front.b/2 + BC.front.a/dz_end).ravel()
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k-1].ravel()
        s[q] = (BC.front.b/2 - BC.front.a/dz_end).ravel()
        BCRHS[G[i,j,k].ravel()] = (BC.front.c).ravel()

        # Back boundary
        k=0
        i = i_ind
        j = j_ind
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k+1].ravel()
        s[q] = -(BC.back.b/2 + BC.back.a/dz_1).ravel()
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = -(BC.back.b/2 - BC.back.a/dz_1).ravel()
        BCRHS[G[i,j,k].ravel()] = -(BC.back.c).ravel()
    elif BC.front.periodic or BC.back.periodic: # periodic
        # Front boundary
        k=Nz+1
        i = i_ind
        j = j_ind
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = 1.0
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k-1].ravel()
        s[q] = -1.0
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,0].ravel()
        s[q] = dz_end/dz_1
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,1].ravel()
        s[q] = -dz_end/dz_1
        BCRHS[G[i,j,k].ravel()] = 0.0

        # Back boundary
        k=0
        i = i_ind
        j = j_ind
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = 1.0
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k+1].ravel()
        s[q] = 1.0
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,Nz].ravel()
        s[q] = -1.0
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,Nz+1].ravel()
        s[q] = -1.0
        BCRHS[G[i,j,k].ravel()] = 0.0

    # Build the sparse matrix of the boundary conditions
    q = q[-1]+1
    BCMatrix = csr_array((s[0:q], (ii[0:q], jj[0:q])), 
                         shape=((Nx+2)*(Ny+2)*(Nz+2), (Nx+2)*(Ny+2)*(Nz+2)))
    return BCMatrix, BCRHS