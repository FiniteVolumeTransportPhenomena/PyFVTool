# Heat transfer example based on a Julia example
# https://docs.sciml.ai/MethodOfLines/stable/tutorials/heat/

import numpy as np
import matplotlib.pyplot as plt
import pyfvtool as pf


def T_analytic(x,t):
    return np.exp(-t)*np.cos(x)

# Parameters
L = 1.0 # domain length
alfa = 1.0 # heat diffusion
qs = 1000.0 # [W/m^2]
t_sim = L**2/(20*alfa) # [s]
time_steps = 50
dt = t_sim/time_steps # 
Nx = 20 # number of cells
m = pf.createMesh1D(Nx, L)
left_bc = "Dirichlet"
# Boundary condition
BC = pf.BoundaryConditions(m)
BC.left.a[:] = 0.0
BC.left.b[:] = 1.0
BC.right.a[:] = 0.0
BC.right.b[:] = 1.0

# Initial condition
T0 = np.cos(m.cellcenters.x) # cosine temperature profile, amplitude = 1.0 K
T_init = pf.createCellVariable(m, T0, BC) # initial condition

# physical parameters
alfa_cell = pf.createCellVariable(m, alfa, pf.BoundaryConditions(m))
alfa_face = pf.harmonicMean(alfa_cell)

M_diff = pf.diffusionTerm(alfa_face)


t=0
while t<t_sim:
    t +=dt
    [M_trans, RHS_trans] = pf.transientTerm(T_init, dt, 1.0)
    BC.left.c[:] = np.exp(-t)
    BC.right.c[:] = np.exp(-t)*np.cos(L)
    [M_bc, RHS_bc] = pf.boundaryConditionsTerm(BC)
    T_val = pf.solvePDE(m, M_bc+M_trans-M_diff, RHS_bc+RHS_trans)
    T_init.update_value(T_val)

x = m.facecenters.x
T_face = pf.linearMean(T_val)
T_num = T_face.xvalue
T_an = T_analytic(x, t_sim)
er = np.sum(np.abs(T_num-T_an)/T_an)/Nx
print(er)
plt.plot(x, T_an, x, T_num, 'o')
plt.legend({'Analytical', 'Numerical'})
plt.xlabel('x [m]')
plt.ylabel('deltaT [K]')
plt.show()