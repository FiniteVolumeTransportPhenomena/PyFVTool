import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf


# Instead of using the broad-spectrum `import pyfvtools as pf`,
# we import each name explicitly in this particular script. This gives us
# an idea of the coverage, and can also serve as a guide to writing more
# elaborate documentation for PyFVTool functions and classes.

from pyfvtool import Grid1D, Grid2D, Grid3D
from pyfvtool import CylindricalGrid1D, CylindricalGrid2D, PolarGrid2D
from pyfvtool import CylindricalGrid3D
from pyfvtool import CellVariable, FaceVariable
from pyfvtool import BoundaryConditions
from pyfvtool import boundaryConditionsTerm, diffusionTerm
from pyfvtool import convectionTerm, convectionUpwindTerm, convectionTvdRHSTerm
from pyfvtool import gradientTerm, divergenceTerm
from pyfvtool import linearSourceTerm, constantSourceTerm
from pyfvtool import transientTerm
from pyfvtool import solveMatrixPDE, solveExplicitPDE
from pyfvtool import harmonicMean, linearMean, arithmeticMean, geometricMean
from pyfvtool import upwindMean
from pyfvtool import fluxLimiter
from pyfvtool import visualizeCells



print('***** HELLO ******')


# Test script, run this before committing changes & creating pull requests...

#TODO: Further organize this test script
#TODO: There is some overlap with test_runs.py => check and merge
#TODO: Include more visualizeCells testing?


# Start test
# The only actual test is to see if this script terminates successfully.
# More detailed tests (constraints on output, for example) could be added.
successful_finish = False 


Lx= 1.0
Ly= 2*np.pi
Lz= 2.0
Nx=5
Ny=7
Nz=9
X=np.array([0.01, 0.1, 0.3, 0.5, 0.55, 1.0])
Y= np.array([0.0, 0.1, 1.0, 1.5, 2.9, 3.0, np.pi, 2*np.pi])
Z= np.array([0.0, 0.01, 0.1, 0.5, 0.7, 0.95, 1.0, 1.25, 1.39, 2.0])
N_mesh=7
# create nonuniform mesh
mesh_nonuniform= []
mesh_nonuniform.append(Grid1D(X))
mesh_nonuniform.append(Grid2D(X, Y))
mesh_nonuniform.append(Grid3D(X, Y, Z))
mesh_nonuniform.append(CylindricalGrid1D(X))
mesh_nonuniform.append(CylindricalGrid2D(X, Y))
mesh_nonuniform.append(CylindricalGrid3D(X, Y, Z))
mesh_nonuniform.append(PolarGrid2D(X, Y))
print("Non-uniform mesh created successfully!")
## Part II: create cell and face variables
c_val= 1.0
D_val = 0.5
# nonuniform
c_n= [CellVariable(m, c_val, BoundaryConditions(m)) for m in mesh_nonuniform]
D_n= [CellVariable(m, D_val, BoundaryConditions(m)) for m in mesh_nonuniform]
print("Cells of fixed values over nonuniform mesh created successfully!")
c_r= [CellVariable(m, np.random.random_sample(m.dims), BoundaryConditions(m)) for m in mesh_nonuniform]
print("Cells of random values over nonuniform mesh created successfully!")
## Part III: create face variables
f_val= 0.5
# nonuniform
f_n = [FaceVariable(m, [f_val,0.0,0.0]) for m in mesh_nonuniform]
print("Face variable over nonuniform mesh created successfully!")
## Part IV: Test boundary conditions
BC_n = []
for m in mesh_nonuniform:
    BC=BoundaryConditions(m)
    BC.left.a[:]=0.0
    BC.left.b[:]=1.0
    BC.left.c[:]=1.0
    # BC.right.a[:]=0.0
    # BC.right.b[:]=1.0
    # BC.right.c[:]=0.0
    BC_n.append(BC)
print("Boundary condition over a nonuniform mesh created successfully!")
## Part V: solve a steady-state diffusion equation
c_dif=[]
M_bc=[]
M_dif=[]
RHS_bc=[]
for i in range(len(mesh_nonuniform)):
    Mbc, RHSbc= boundaryConditionsTerm(BC_n[i])
    M_bc.append(Mbc)
    RHS_bc.append(RHSbc)
    Md = diffusionTerm(f_n[i])
    M_dif.append(Md)
    print(i)
    c_dif.append(solveMatrixPDE(mesh_nonuniform[i], -M_dif[i]+M_bc[i], RHS_bc[i]))


L = 1.0  # domain length
Nx = 25 # number of cells
meshstruct = Grid1D(Nx, L)
x = meshstruct.cellcenters.x # extract the cell center positions
##
# The next step is to define the boundary condition:
BC = BoundaryConditions(meshstruct) # all Neumann boundary condition structure
# BC.left.a[:] = 0 
# BC.left.b[:] = 1 # switch the left boundary to Dirichlet
# BC.left.c[:] = 0 # value = 0 at the left boundary
BC.right.a[:] = 0 
BC.right.b[:] =1 # switch the right boundary to Dirichlet
BC.right.c[:] =1 # value = 1 at the right boundary
##
# Now we define the transfer coefficients:
D_val = 1.0 # diffusion coefficient value
D = CellVariable(meshstruct, D_val, BoundaryConditions(meshstruct)) # assign dif. coef. to all the cells
Dave = harmonicMean(D) # convert a cell variable to face variable
u = -10.0 # velocity value
u_face = FaceVariable(meshstruct, u) # assign velocity value to cell faces

Mconv =  convectionTerm(u_face) # convection term, central, second order
Mconvupwind = convectionUpwindTerm(u_face) # convection term, upwind, first order
Mdiff = diffusionTerm(Dave) # diffusion term
Mbc, RHSbc = boundaryConditionsTerm(BC) # boundary condition discretization
M = Mconv-Mdiff+Mbc # matrix of coefficient for central scheme
Mupwind = Mconvupwind-Mdiff+Mbc # matrix of coefficient for upwind scheme
RHS = RHSbc # right hand side vector
c = solveMatrixPDE(meshstruct, M, RHS) # solve for the central scheme
c_upwind = solveMatrixPDE(meshstruct, Mupwind, RHS) # solve for the upwind scheme
c_analytical = (1-np.exp(u*x/D_val))/(1-np.exp(u*L/D_val)) # analytical solution
# plt.figure(5)
# plt.plot(x, c.internalCellValues) 
# plt.plot(x, c_upwind.internalCellValues, '--')
# plt.plot(x, c_analytical, '.')
# plt.show()
# plt.legend('central', 'upwind', 'analytical')

## Part VI: solve convection diffucion equation
# nonuniform
c_conv=[]
M_bc=[]
M_dif=[]
M_conv=[]
RHS_bc=[]
for i in range(len(mesh_nonuniform)):
    M, RHS= boundaryConditionsTerm(BC_n[i])
    M_bc.append(M)
    RHS_bc.append(RHS)
    M=diffusionTerm(f_n[i])
    M_dif.append(M)
    M_conv.append(convectionTerm(0.1*f_n[i]))
    c_conv.append(solveMatrixPDE(mesh_nonuniform[i], M_conv[i]-M_dif[i]+M_bc[i], RHS_bc[i]))
    # print(c_conv[i].value)
# # visualize
# # figure(2)
# # for i=1:N_mesh
# #     subplot(3, 3, i)
# #     visualizeCells(c_conv[i])
# # end
# # println("Convection-Diffusion equation solved and visualized successfully")
# ## Part VII: test the calculus fanctions
grad_c=[]
for i in range(len(mesh_nonuniform)):
    grad_c.append(gradientTerm(c_dif[i]))

div_c=[]
for i in range(len(mesh_nonuniform)):
    div_c.append(divergenceTerm(grad_c[i]))
print("Gradient and divergence functions work fine!")
# ## Solve a dynamic equation
dt=0.1
FL1=fluxLimiter("SUPERBEE")
c_old=c_n
c_trans=[]
M_bc=[]
M_dif=[]
M_conv=[]
RHS_bc=[]
M_ls=[]
RHS_s=[]
RHS_tvd=[]
for i in range(len(mesh_nonuniform)):
    M, RHS= boundaryConditionsTerm(BC_n[i])
    M_bc.append(M)
    RHS_bc.append(RHS)
    M=diffusionTerm(f_n[i])
    M_dif.append(M)
    M_conv.append(convectionUpwindTerm(0.1*f_n[i]))
    RHS_tvd.append(convectionTvdRHSTerm(0.01*f_n[i], c_old[i], FL1)) #only called, not used
    M_ls.append(linearSourceTerm(0.1*c_n[i]))
    RHS_s.append(constantSourceTerm(0.2*c_n[i]))

for i in range(len(mesh_nonuniform)):
    for j in range(1,10):
        M_t, RHS_t=transientTerm(c_old[i], dt, 1.0)
        c_new=solveMatrixPDE(mesh_nonuniform[i],
            M_t+M_ls[i]+M_conv[i]-M_dif[i]+M_bc[i], RHS_t+RHS_s[i]+RHS_bc[i])
        c_old[i].update_value(c_new)
    c_trans.append(c_old[i])
# # visualize
# # figure(3)
# #for i=1:N_mesh
# #     visualizeCells(c_trans[i])
# #     pause(1.5)
# #end
# println("Transient convection-diffucion-reaction solved successfully!")
## Part VIII: test the utilities
# only test the averaging, don"t save the result
for i in range(len(mesh_nonuniform)):
    linearMean(c_trans[i])
    arithmeticMean(c_trans[i])
    geometricMean(D_n[i])
    harmonicMean(D_n[i])
    upwindMean(c_trans[i], f_n[i])
    # tvdMean(c_trans[i], f_n[i], FL1)
print("Averaging functions run smoothly!")
# ## Part IX: test the classes and operators

# end # end function

# the explicit solver
# define the domain
L = 5.0  # domain length
Nx = 100 # number of cells
meshstruct = Grid1D(Nx, L)
BC = BoundaryConditions(meshstruct) # all Neumann boundary condition structure
BC.left.a[:] = 0 
BC.left.b[:]=1 
BC.left.c[:]=1 # left boundary
BC.right.a[:] = 0 
BC.right.b[:]=1 
BC.right.c[:]=0 # right boundary
x = meshstruct.cellcenters.x
## define the transfer coeffs
D_val = 1.0
alfa = CellVariable(meshstruct, 1)
Dave = FaceVariable(meshstruct, D_val)
## define initial values
c_old = CellVariable(meshstruct, 0, BC) # initial values
c = CellVariable(meshstruct, 0, BC) # working values
## loop
dt = 0.001 # time step
final_t = 0.5
for t in np.arange(dt, final_t, dt):
    # step 1: calculate divergence term
    RHS = divergenceTerm(Dave*gradientTerm(c_old))
    # step 2: calculate the new value for internal cells
    c = solveExplicitPDE(c_old, dt, RHS, BC)
    c_old.update_value(c)

# analytical solution
c_analytical = 1-erf(x/(2*np.sqrt(D_val*t)))
plt.figure(1)
plt.clf()
plt.plot(x, c.internalCellValues, x, c_analytical, 'r--')
# plt.show()



# Testing 1D visualization...
# (code from README.md )

# Solving a 1D diffusion equation with a fixed concentration 
# at the left boundary and a closed boundary on the right side
Nx = 20 # number of finite volume cells
Lx = 1.0 # [m] length of the domain 
c_left = 1.0 # left boundary concentration
c_init = 0.0 # initial concentration
D_val = 1e-5 # diffusion coefficient (gas phase)
t_simulation = 3600.0 # [s] simulation time
dt = 60.0 # [s] time step

m1 = Grid1D(Nx, Lx) # mesh object
bc = BoundaryConditions(m1) # Neumann boundary condition by default

# switch the left boundary to Dirichlet: fixed concentration
bc.left.a[:] = 0.0
bc.left.b[:] = 1.0
bc.left.c[:] = c_left

# create a cell variable with initial concentration
c_old = CellVariable(m1, c_init, bc)

# assign diffusivity to cells
D_cell = CellVariable(m1, D_val)
D_face = geometricMean(D_cell) # average value of diffusivity at the interfaces between cells

# Discretization
Mdiff = diffusionTerm(D_face)
Mbc, RHSbc = boundaryConditionsTerm(bc)

# time loop
t = 0
while t<t_simulation:
    t+=dt
    Mt, RHSt = transientTerm(c_old, dt, 1.0)
    c_new = solveMatrixPDE(m1, Mt-Mdiff+Mbc, RHSbc+RHSt)
    c_old.update_value(c_new)

plt.figure(2)
plt.clf()
visualizeCells(c_old)

# Testing 2D visualization
#
mm = Grid2D(50, 50, 5*np.pi, 5*np.pi)
XX, YY = np.meshgrid(mm.cellcenters.x, mm.cellcenters.y)
vv = CellVariable(mm, np.cos(XX)*np.sin(YY))
plt.figure(3)
plt.clf()
visualizeCells(vv)

# Only use show() at the end of the script, since only a single call is needed
# to display all created Figures. `visualizeCells` does not call plt.show() anymore
#
# In Notebooks and in Spyder, plt.show() is not necessary. It is only needed
# when running scripts in a stand-alone Python interpreter.
#
# plt.show() # show figures and wait for user to close windows

# Replaced plt.show() with plt.pause(3) which does not necessitate user 
# interaction to close windows.
#
plt.pause(3) # show figures and pause for 3 seconds before continuing


# end test (if the scripts run until here, it should be OK)
successful_finish = True


# pytest
def test_success():
    assert successful_finish
