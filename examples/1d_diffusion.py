from pyfvtool import *

# Solving a 1D diffusion equation with a fixed concentration 
# at the left boundary and a closed boundary on the right side
Nx = 20 # number of finite volume cells
Lx = 1.0 # [m] length of the domain 
c_left = 1.0 # left boundary concentration
c_init = 0.0 # initial concentration
D_val = 1e-5 # diffusion coefficient (gas phase)
t_simulation = 3600.0 # [s] simulation time
dt = 60.0 # [s] time step

m1 = createMesh1D(Nx, Lx) # mesh object
bc = createBC(m1) # Neumann boundary condition by default

# switch the left boundary to Dirichlet: fixed concentration
bc.left.a[:] = 0.0
bc.left.b[:] = 1.0
bc.left.c[:] = c_left

# create a cell variable with initial concentration
c_old = createCellVariable(m1, c_init, bc)

# assign diffusivity to cells
D_cell = createCellVariable(m1, D_val, createBC(m1))
D_face = geometricMean(D_cell) # average value of diffusivity at the interfaces between cells

# Discretization
Mdiff = diffusionTerm(D_face)
Mbc, RHSbc = boundaryConditionTerm(bc)

# time loop
t = 0
while t<t_simulation:
    t+=dt
    Mt, RHSt = transientTerm(c_old, dt, 1.0)
    c_new = solvePDE(m1, Mt-Mdiff+Mbc, RHSbc+RHSt)
    c_old.update_value(c_new)

visualizeCells(c_old)

