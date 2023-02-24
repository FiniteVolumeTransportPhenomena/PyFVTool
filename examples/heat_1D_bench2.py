
from pyfvtool import *
import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

# From a Julia example
# https://docs.sciml.ai/MethodOfLines/stable/tutorials/heat/

def T_analytic(x,t):
    return np.exp(-t)*np.cos(x)

# Parameters
L = 1.0 # domain length
alfa = 1.0 # heat diffusion
Ts = 350.0 # [K]
qs = 1000 # [W/m^2]
t_sim = L**2/(20*alfa) # [s]
time_steps = 50
dt = t_sim/time_steps # 
Nx = 20 # number of cells
m = Mesh1D(Nx, L)
left_bc = "Dirichlet"
# Boundary condition
BC = createBC(m)
BC.left.a[:] = 0.0
BC.left.b[:] = 1.0
BC.right.a[:] = 0.0
BC.right.b[:] = 1.0
# Initial condition
T0 = np.cos(m.cellcenters.x)
T_init = createCellVariable(m, T0, BC) # initial condition
# physical parameters
alfa_cell = createCellVariable(m, alfa, createBC(m))
alfa_face = harmonicMean(alfa_cell)

M_diff = diffusionTerm(alfa_face)


t=0
while t<t_sim:
    t +=dt
    [M_trans, RHS_trans] = transientTerm(T_init, dt, 1.0)
    BC.left.c[:] = np.exp(-t)
    BC.right.c[:] = np.exp(-t)*np.cos(L)
    [M_bc, RHS_bc] = boundaryConditionTerm(BC)
    T_val = solvePDE(m, M_bc+M_trans-M_diff, RHS_bc+RHS_trans)
    T_init.update_value(T_val)

x = m.facecenters.x
T_face = linearMean(T_val)
T_num = T_face.xvalue
plt.plot(x, T_analytic(x, t_sim), x, T_num, 'o')
plt.legend({'Analytical', 'Numerical'})
plt.xlabel('x [m]')
plt.ylabel('T [K]')
plt.show()