# This test tests the basic working of most functions in PyFVTool

import numpy as np

import pyfvtool as pf

# This script calls all the functions of the pyfvtool package
## Part I: creating an array of different mesh types:
# domain size

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
mesh_nonuniform.append(pf.Grid1D(X))
mesh_nonuniform.append(pf.Grid2D(X, Y))
mesh_nonuniform.append(pf.Grid3D(X, Y, Z))
mesh_nonuniform.append(pf.CylindricalGrid1D(X))
mesh_nonuniform.append(pf.CylindricalGrid2D(X, Y))
mesh_nonuniform.append(pf.CylindricalGrid3D(X, Y, Z))
mesh_nonuniform.append(pf.PolarGrid2D(X, Y))
print("Non-uniform mesh created successfully!")
## Part II: create cell and face variables
c_val= 1.0
D_val = 0.5
# nonuniform
c_n= [pf.CellVariable(m, c_val, pf.BoundaryConditions(m)) for m in mesh_nonuniform]
D_n= [pf.CellVariable(m, D_val, pf.BoundaryConditions(m)) for m in mesh_nonuniform]
print("Cells of fixed values over nonuniform mesh created successfully!")
c_r= [pf.CellVariable(m, np.random.random_sample(m.dims), pf.BoundaryConditions(m))\
      for m in mesh_nonuniform]
print("Cells of random values over nonuniform mesh created successfully!")
## Part III: create face variables
f_val= 0.5
# nonuniform
f_n = [pf.FaceVariable(m, f_val) for m in mesh_nonuniform]
print("Face variable over nonuniform mesh created successfully!")
## Part IV: Test boundary conditions
BC_n = []
for m in mesh_nonuniform:
    BC = pf.BoundaryConditions(m)
    # BC.left.a[:]=0.0
    # BC.left.b[:]=1.0
    # BC.left.c[:]=1.0
    BC.right.a[:]=0.0
    BC.right.b[:]=1.0
    BC.right.c[:]=0.0
    BC_n.append(BC)
print("Boundary condition over a nonuniform mesh created successfully!")
## Part V: solve a steady-state diffusion equation
c_dif=[]
M_bc=[]
M_dif=[]
RHS_bc=[]
for i in range(len(mesh_nonuniform)):
    Mbc, RHSbc= pf.boundaryConditionsTerm(BC_n[i])
    M_bc.append(Mbc)
    RHS_bc.append(RHSbc)
    Md = pf.diffusionTerm(f_n[i])
    if type(Md) is tuple:
        M_dif.append(Md[0])
    else:
        M_dif.append(Md)
    print(i)
    c_dif.append(pf.solvePDE(mesh_nonuniform[i], -M_dif[i]+M_bc[i], RHS_bc[i]))

## Part VI: solve convection diffucion equation
# nonuniform
c_conv=[]
M_bc=[]
M_dif=[]
M_conv=[]
RHS_bc=[]
for i in range(len(mesh_nonuniform)):
    M, RHS= pf.boundaryConditionsTerm(BC_n[i])
    M_bc.append(M)
    RHS_bc.append(RHS)
    M=pf.diffusionTerm(f_n[i])
    M_dif.append(M)
    M_conv.append(pf.convectionTerm(0.1*f_n[i]))
    c_conv.append(pf.solvePDE(mesh_nonuniform[i], 
                           M_conv[i]-M_dif[i]+M_bc[i],
                           RHS_bc[i]))
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
    grad_c.append(pf.gradientTerm(c_dif[i]))

div_c=[]
for i in range(len(mesh_nonuniform)):
    div_c.append(pf.divergenceTerm(grad_c[i]))
print("Gradient and divergence functions work fine!")
# ## Solve a dynamic equation
dt=0.1
FL1=pf.fluxLimiter("SUPERBEE")
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
    M, RHS= pf.boundaryConditionsTerm(BC_n[i])
    M_bc.append(M)
    RHS_bc.append(RHS)
    M=pf.diffusionTerm(f_n[i])
    M_dif.append(M)
    M_conv.append(pf.convectionUpwindTerm(0.1*f_n[i]))
    RHS_tvd.append(pf.convectionTvdRHSTerm(0.01*f_n[i], c_old[i], FL1)) #only called, not used
    M_ls.append(pf.linearSourceTerm(0.1*c_n[i]))
    RHS_s.append(pf.constantSourceTerm(0.2*c_n[i]))

for i in range(len(mesh_nonuniform)):
    for j in range(1,10):
        M_t, RHS_t=pf.transientTerm(c_old[i], dt, 1.0)
        c_new=pf.solvePDE(mesh_nonuniform[i],
                          M_t+M_ls[i]+M_conv[i]-M_dif[i]+M_bc[i],
                          RHS_t+RHS_s[i]+RHS_bc[i])
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
    pf.linearMean(c_trans[i])
    pf.arithmeticMean(c_trans[i])
    pf.geometricMean(D_n[i])
    pf.harmonicMean(D_n[i])
    pf.upwindMean(c_trans[i], f_n[i])
    # tvdMean(c_trans[i], f_n[i], FL1)
print("Averaging functions run smoothly!")

# end test (if the scripts run until here, it should be OK)
successful_finish = True


# pytest
def test_success():
    assert successful_finish
