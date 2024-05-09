# -*- coding: utf-8 -*-
"""
Test CellVariable copy

"""
import numpy as np
import matplotlib.pyplot as plt

import pyfvtool as pf

# Solving a 1D diffusion equation with a fixed concentration 
# at the left boundary and a closed boundary on the right side


# Calculation parameters
Nx = 20 # number of finite volume cells
Lx = 1.0 # [m] length of the domain 
c_left = 1.0 # left boundary concentration
c_init = 0.0 # initial concentration
D_val = 1e-5 # diffusion coefficient (gas phase)
t_simulation = 7200.0 # [s] simulation time
dt = 60.0 # [s] time step
Nskip = 10 # plot every Nskip-th profile

# Define mesh
mesh = pf.Grid1D(Nx, Lx)

# Create a cell variable with initial concentration
# By default, 'no flux' boundary conditions are applied
c = pf.CellVariable(mesh, c_init)

# Switch the left boundary to Dirichlet: fixed concentration
c.BCs.left.a[:] = 0.0
c.BCs.left.b[:] = 1.0
c.BCs.left.c[:] = c_left
c.apply_BCs()


# Create a copy for a second run
cc = c.copy()


# Assign diffusivity to cells
D_cell = pf.CellVariable(mesh, D_val)
D_face = pf.geometricMean(D_cell) # average value of diffusivity at the interfaces between cells

plt.figure(1)
plt.clf()
# Time loop
t = 0
nplot = 0
while t<t_simulation:
    # Compose discretized terms for matrix equation
    eqnterms = [ pf.transientTerm(c, dt, 1.0),
                -pf.diffusionTerm(D_face)]

    # Solve PDE
    pf.solvePDE(c, eqnterms)
    t+=dt

    if (nplot % Nskip == 0):
        pf.visualizeCells(c)
    nplot+=1
    
    
# Do it once more on the copy

plt.figure(2)
plt.clf()
t = 0
nplot = 0
while t<t_simulation:
    # Compose discretized terms for matrix equation
    eqnterms = [ pf.transientTerm(cc, dt, 1.0),
           -pf.diffusionTerm(D_face)]

    # Solve PDE
    pf.solvePDE(cc, eqnterms)
    t+=dt

    if (nplot % Nskip == 0):
        pf.visualizeCells(cc)
    nplot+=1
    
    
### README code ends here

# end test: identical results for both runs
successful_finish = np.all(c._value == cc._value)


# pytest
def test_success():
    assert successful_finish