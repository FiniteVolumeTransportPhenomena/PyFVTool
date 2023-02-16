import numpy as np
from importlib import reload
import mesh, boundary
reload(mesh)
reload(boundary)
from mesh import *
from boundary import *

# Development story
m1 = MeshCylindrical1D(10, 1.0)
m2 = MeshCylindrical2D(10, 5, 1.0, 1.0)
m3 = Mesh3D(4, 5, 6, 1.0, 2.0, 3.0)
print(m3)
print(m3.cell_numbers())
# TBD: call all mesh functions
# Could create some of the meshes

# lesson: specified types in fuctions considered as hint by python
bc1 = BoundaryCondition1D(m1)
M, rhs = boundaryCondition1D(bc1)

bc2 = BoundaryCondition2D(m2)
M, rhs = boundaryCondition2D(bc2)
print(M)
print(rhs)
# the code runs fine after some squeezing of arrays
# the next stage is to implement more BC functions
# Now I test the 3D BC
bc3 = BoundaryCondition3D(m3)
M, rhs = boundaryCondition3D(bc3)
print(M)
print(rhs)

# learning: use ravel instead of flatten to make the code faster
# there are other functions for flattening the array including a flat that is the 
# flat form of the array for indexing and reshape(-1) that also flattens the array 
# apparently without making a copy

# session 3 of development
# numpy broadcasts arrays
m_rad = MeshRadial2D(7,9, 1, 2*np.pi)
bc_rad = BoundaryCondition2D(m_rad)
M, rhs = boundaryConditionRadial2D(bc_rad)
print(M)
print(rhs)

# It runs now
# the [:, np.newaxis] is just amazing!
# The code is not clean. I'm still doing lots of copy and pasting with tons of
# boiler plate. But remember any answer is better than no answer.
# The final boundary: 3D cylindrical
m_cyl = MeshCylindrical3D(7,9,10, 1, 2*np.pi,3)
bc_cyl = BoundaryCondition3D(m_cyl)
M, rhs = boundaryConditionCylindrical3D(bc_cyl)
print(M)
print(rhs)

# It runs now! Now I will add all the cell and face variable creations.
# the saga continues
# Now I'm creating the CellVariables, for which I need cellBoundary functions.
# Here they are being tested:
Nx = 5
Ny = 6
Nz = 7
Lx = 1.0
Ly = 2.0 
Lz = 3.0
m1 = Mesh1D(Nx, Lx)
BC1 = createBC(m1)
phi = np.random.rand(Nx)
phi_cell = cellBoundary1D(phi, BC1)
print(phi)
print(phi_cell)

m2 = Mesh2D(Nx, Ny, Lx, Ly)
BC2 = createBC(m2)
phi = np.random.rand(Nx, Ny)
phi_cell = cellBoundary2D(phi, BC2)
print(phi)
print(phi_cell)

m3 = Mesh3D(Nx, Ny, Nz, Lx, Ly, Lz)
BC3 = createBC(m3)
phi = np.ones((Nx, Ny, Nz))
phi_cell = cellBoundary3D(phi, BC3)
print(phi)
print(phi_cell)

m3 = MeshCylindrical3D(Nx, Ny, Nz, Lx, Ly, Lz)
BC3 = createBC(m3)
phi = np.ones((Nx, Ny, Nz))
phi_cell = cellBoundaryCylindrical3D(phi, BC3)
print(phi)
print(phi_cell)

m2 = MeshRadial2D(Nx, Ny, Lx, Ly)
BC2 = createBC(m2)
phi = np.random.rand(Nx, Ny)
phi_cell = cellBoundaryRadial2D(phi, BC2)
print(phi)
print(phi_cell)

# I just finished the boundary values and the create cell functions.
# the overloading functions must wait until later this afternoon.
# for overloading, python has __add__ and __radd__ for the second argument and __iadd__ for +=
# It will be a lot of writing today!
# now I'm moving to porting the diffusion and advection terms.
# It is going to be fun because we are very close to having the first 1D problems solved!
# I will also write the first tests for the cell and face creation here:
