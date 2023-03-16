from pyfvtool import *
import matplotlib.pyplot as plt
import numpy as np
# a tutorial adapted from the fipy convection diffusion 1D example
# see: http://www.ctcms.nist.gov/fipy/examples/convection/index.html
# Written by Ali A. Eftekhari
# Last checked: June 2021
## define the domain
L = 1.0  # domain length
Nx = 25 # number of cells
meshstruct = createMesh1D(Nx, L)
BC = createBC(meshstruct) # all Neumann boundary condition structure
BC.left.a[:] = 0 
BC.left.b[:] = 1 
BC.left.c[:] = 0 # left boundary
BC.right.a[:] = 0 
BC.right.b[:] = 1 
BC.right.c[:] = 1 # right boundary
x = meshstruct.cellcenters.x
## define the transfer coeffs
D_val = -1
D = createCellVariable(meshstruct, D_val)
Dave = harmonicMean(D) # convert a cell variable to face variable
alfa = createCellVariable(meshstruct, 1)
u = -10
u_face = createFaceVariable(meshstruct, u)
## solve
Mconv =  convectionTerm(u_face)
Mconvupwind =  convectionUpwindTerm(u_face)
Mdiff = diffusionTerm(Dave)
[Mbc, RHSbc] = boundaryConditionTerm(BC)
M = Mconv-Mdiff-Mbc
Mupwind = Mconvupwind-Mdiff-Mbc
RHS = -RHSbc
c = solvePDE(meshstruct, M, RHS)
c_upwind = solvePDE(meshstruct, Mupwind, RHS)
c_analytical = (1-np.exp(u*x/D_val))/(1-np.exp(u*L/D_val))
plt.plot(x, c.value[1:Nx+1], label="Central")
plt.plot(x, c_upwind.value[1:Nx+1], label="Upwind")
plt.plot(x, c_analytical, '^', label = "Analytical")
plt.legend()
plt.show()