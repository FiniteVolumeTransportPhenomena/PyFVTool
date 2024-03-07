# A tutorial adapted from the FiPy 1D diffusion example
# see: http://www.ctcms.nist.gov/fipy/examples/diffusion/index.html
# Converted from Matlab FVTool to Python PyFVTool

import pyfvtool as pf
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

# define the domain
L = 5.0  # domain length
Nx = 100 # number of cells
meshstruct = pf.createMeshCylindrical2D(Nx, Nx, L, L)
BC = pf.BoundaryConditions(meshstruct) # all Neumann boundary condition structure
BC.bottom.a[:] = 0.0 
BC.bottom.b[:] = 1.0 
BC.bottom.c[:] = 1.0 # bottom boundary
BC.top.a[:] = 0.0 
BC.top.b[:] = 1.0 
BC.top.c[:] = 0.0 # top boundary
x = meshstruct.cellcenters.x
## define the transfer coeffs
D_val = 1.0
alfa = pf.CellVariable(meshstruct, 1.0)
Dave = pf.FaceVariable(meshstruct, D_val)
## define initial values
c_old = pf.CellVariable(meshstruct, 0.0, BC) # initial values
c = pf.CellVariable(meshstruct, 0.0, BC) # working values
## loop
dt = 0.001 # time step
final_t = 100*dt
for t in np.arange(dt, final_t, dt):
    # step 1: calculate divergence term
    RHS = pf.divergenceTerm(Dave*pf.gradientTerm(c_old))
    # step 2: calculate the new value for internal cells
    c = pf.solveExplicitPDE(c_old, dt, RHS, BC)
    c_old.update_value(c)

# analytical solution
c_analytical = 1-erf(x/(2*np.sqrt(D_val*t)))

plt.figure(1)
plt.clf()
plt.plot(x, c.internalCellValues[2,:], 'k', label='PyFVTool')
plt.plot(x, c_analytical, 'r--', label='analytic')
plt.legend()
plt.show()
