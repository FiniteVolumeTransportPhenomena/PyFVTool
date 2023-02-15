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
