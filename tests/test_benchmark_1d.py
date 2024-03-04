import unittest
import numpy as np
from scipy.special import erf

import pyfvtool as pf

    

OUTPUT_DIAGNOSTICS = False


def T_analytical_dirichlet(x,t, alfa, T0, Ts):
        return (T0-Ts)*erf(x/np.sqrt(4*alfa*t))+Ts

def T_analytical_neuman(x,t, alfa, T0, k, qs):
    return T0+qs/k*np.sqrt(4*alfa*t/np.pi)*np.exp(-x**2/(4*alfa*t))-qs/k*x*(1-erf(x/np.sqrt(4*alfa*t)))

def T_numerical(left_bc: str) -> float:
    L = 1.0 # [m] domain length
    k = 20.0 # 0.6 for water, 0.025 for air W/m/K
    rho = 8000.0 # kg/m^3
    c = 500.0 # J/kg/K (4200 for water, 1000 for air)
    alfa = k/(rho*c) # heat diffusion
    T0 = 300.0 # [K]
    Ts = 350.0 # [K]
    qs = 1000 # [W/m^2]
    t_sim = L**2/(20*alfa) # [s]
    time_steps = 50
    dt = t_sim/time_steps # 
    Nx = 50 # number of cells
    m = pf.createMesh1D(Nx, L)
    # Boundary condition
    BC = pf.BoundaryConditions(m)
    if left_bc == "Dirichlet":
        BC.left.a[:] = 0.0
        BC.left.b[:] = 1.0
        BC.left.c[:] = Ts
        T_analytic = lambda x,t: T_analytical_dirichlet(x, t, alfa, T0, Ts)
    else:
        BC.left.a[:] = k
        BC.left.b[:] = 0.0
        BC.left.c[:] = -qs
        T_analytic = lambda x,t: T_analytical_neuman(x, t, alfa, T0, k, qs)

    # Initial condition
    T_init = pf.CellVariable(m, T0, BC) # initial condition
    # physical parameters
    alfa_cell = pf.CellVariable(m, alfa, pf.BoundaryConditions(m))
    alfa_face = pf.harmonicMean(alfa_cell)

    M_diff = pf.diffusionTerm(alfa_face)
    [M_bc, RHS_bc] = pf.boundaryConditionsTerm(BC)

    t=0
    while t<t_sim:
        t +=dt
        [M_trans, RHS_trans] = pf.transientTerm(T_init, dt, 1.0)
        T_val = pf.solvePDE(m, M_bc+M_trans-M_diff, RHS_bc+RHS_trans)
        T_init.update_value(T_val)

    x = m.facecenters.x
    T_face = pf.linearMean(T_val)
    T_num = T_face.xvalue
    T_an = T_analytic(x, t_sim)
    er = np.sum(np.abs(T_num-T_an)/T_an)/Nx
    return er

def conv_numerical_1d() -> float:
    global OUTPUT_DIAGNOSTICS
    L = 1.0  # domain length
    Nx = 50 # number of cells
    meshstruct = pf.createMesh1D(Nx, L)
    BC = pf.BoundaryConditions(meshstruct) # all Neumann boundary condition structure
    BC.left.a[:] = 0 
    BC.left.b[:] = 1 
    BC.left.c[:] = 0 # left boundary
    BC.right.a[:] = 0 
    BC.right.b[:] = 1 
    BC.right.c[:] = 1 # right boundary
    x = meshstruct.cellcenters.x
    ## define the transfer coeffs
    D_val = -1
    D = pf.CellVariable(meshstruct, D_val)
    Dave = pf.harmonicMean(D) # convert a cell variable to face variable
    # alfa = CellVariable(meshstruct, 1)
    u = -10.0
    u_face = pf.FaceVariable(meshstruct, u)
    ## solve
    Mconv =  pf.convectionTerm(u_face)
    # Mconvupwind =  convectionUpwindTerm(u_face)
    Mdiff = pf.diffusionTerm(Dave)
    [Mbc, RHSbc] = pf.boundaryConditionsTerm(BC)
    M = Mconv-Mdiff-Mbc
    # Mupwind = Mconvupwind-Mdiff-Mbc
    RHS = -RHSbc
    c = pf.solvePDE(meshstruct, M, RHS)
    # c_upwind = solvePDE(meshstruct, Mupwind, RHS)
    c_analytical = (1-np.exp(u*x/D_val))/(1-np.exp(u*L/D_val))
    er = np.sum(np.abs(c_analytical-c.value[1:-1]))/Nx
    if OUTPUT_DIAGNOSTICS:
        return er, x, c_analytical, c.value[1:-1]
    else:
        return er


class TestDiffusion(unittest.TestCase):
    def test_1d_dirichlet(self):
        print("\nRunning 1D conduction heat transfer with Dirichlet boundary:")
        left_bc = "Dirichlet"
        er = T_numerical(left_bc)
        eps_T = 0.001
        self.assertLessEqual(er, eps_T)
    def test_1d_neumann(self):
        print("\nRunning 1D conduction heat transfer with Neumann boundary:")
        left_bc = "Neumann"
        er = T_numerical(left_bc)
        eps_T = 0.001
        self.assertLessEqual(er, eps_T)

class TestConvection(unittest.TestCase):
    def test_1d_convection(self):
        print("\nRunning 1D convection:")
        er = conv_numerical_1d()
        eps_c = 0.001
        self.assertLessEqual(er, eps_c)
        
        
if __name__ == '__main__':
    # This part allows for this script to be run directly, without
    # `pytest`. This is useful for further diagnostics and debugging
    print('Running test script without pytest...')
    print()
    
    import matplotlib.pyplot as plt
    OUTPUT_DIAGNOSTICS = True
    
    print("\nRunning 1D convection...")
    er, x, c_an, c_num = conv_numerical_1d()
    plt.figure(1)
    plt.clf()
    plt.plot(x, c_an, label='analytic')
    plt.plot(x, c_num, label='FVM')
    plt.legend()
    
    plt.show()
    
    
    

