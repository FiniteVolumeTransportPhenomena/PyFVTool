# A tutorial example adapted from the FiPy 1D diffusion example
# see: http://www.ctcms.nist.gov/fipy/examples/diffusion/index.html
# Converted from Matlab FVTool to Python PyFVTool

import pyfvtool as pf
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

# define the domain
L = 5.0  # domain length
Nx = 100 # number of cells
meshstruct = pf.createMesh1D(Nx, L)
BC = pf.createBC(meshstruct) # all Neumann boundary condition structure
BC.left.a[:] = 0 
BC.left.b[:]=1 
BC.left.c[:]=1 # left boundary
BC.right.a[:] = 0 
BC.right.b[:]=1 
BC.right.c[:]=0 # right boundary
x = meshstruct.cellcenters.x
## define the transfer coeffs
D_val = 1.0
alfa = pf.createCellVariable(meshstruct, 1)
Dave = pf.createFaceVariable(meshstruct, D_val)
## define initial values
c_old = pf.createCellVariable(meshstruct, 0, BC) # initial values
c = pf.createCellVariable(meshstruct, 0, BC) # working values
## loop
dt = 0.001 # time step
final_t = 0.5
for t in np.arange(dt, final_t, dt):
    # step 1: calculate divergence term
    RHS = pf.divergenceTerm(Dave*pf.gradientTerm(c_old))
    # step 2: calculate the new value for internal cells
    c = pf.solveExplicitPDE(c_old, dt, RHS, BC)
    c_old.update_value(c)

# analytical solution
c_analytical = 1-erf(x/(2*np.sqrt(D_val*t)))
plt.plot(x, c.internalCells(), x, c_analytical, 'r--')
plt.show()
