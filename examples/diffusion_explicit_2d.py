# a tutorial adapted from the fipy diffusion 1D example
# see: http://www.ctcms.nist.gov/fipy/examples/diffusion/index.html
# Converted from Matlab FVTool to Python PyFVTool
from pyfvtool import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

# define the domain
L = 5.0  # domain length
Nx = 100 # number of cells
meshstruct = createMeshCylindrical2D(Nx, Nx, L, L)
BC = createBC(meshstruct) # all Neumann boundary condition structure
BC.bottom.a[:] = 0 
BC.bottom.b[:]=1 
BC.bottom.c[:]=1 # bottom boundary
BC.top.a[:] = 0 
BC.top.b[:]=1 
BC.top.c[:]=0 # top boundary
x = meshstruct.cellcenters.x
## define the transfer coeffs
D_val = 1.0
alfa = createCellVariable(meshstruct, 1)
Dave = createFaceVariable(meshstruct, D_val)
## define initial values
c_old = createCellVariable(meshstruct, 0, BC) # initial values
c = createCellVariable(meshstruct, 0, BC) # working values
## loop
dt = 0.001 # time step
final_t = 100*dt
for t in np.arange(dt, final_t, dt):
    # step 1: calculate divergence term
    RHS = divergenceTerm(Dave*gradientTerm(c_old))
    # step 2: calculate the new value for internal cells
    c = solveExplicitPDE(c_old, dt, RHS, BC)
    c_old.update_value(c)

# analytical solution
c_analytical = 1-erf(x/(2*np.sqrt(D_val*t)))
plt.plot(x, c.internalCells()[2,:], x, c_analytical, 'r--')
plt.show()
