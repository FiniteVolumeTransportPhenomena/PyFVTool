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
