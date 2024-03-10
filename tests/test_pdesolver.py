"""
Created on Sat Mar  9 20:32:42 2024

@author: werts-moltech

Solving the Mason-Weaver equation

with the solvePDE2 routine
"""

import matplotlib.pyplot as plt

import pyfvtool as pf


D_coeff = 0.015
sg = 0.2

Nx = 100
Lx = 1.0
dt = 0.01
t_simulation = 2.5
Nskip = 10

msh = pf.Grid1D(Nx, Lx)

BC_c = pf.BoundaryConditions(msh)

c = pf.CellVariable(msh, 1.0, BC_c)

# advection field
u = pf.FaceVariable(msh, (sg,))
# closed boundaries
u.xvalue[0] = 0.0
u.xvalue[-1] = 0.0

# diffusion field
D = pf.FaceVariable(msh, D_coeff)


# prepare plot
plt.figure(1)
plt.clf()
pf.visualizeCells(c)

# time loop
t = 0
nplot = 0
while (t < t_simulation):
    t+=dt
    
    # In the present implementation, also the invariant terms
    # (boundaryConditionsTerm, diffusionTerm and convectionTerm)
    # are re-constructed every cycle. This is done for clarity.
    # Code can be 'optimized' by constructing these terms outside
    # of the loop and store their results. The difference is 
    # probably minimal, since most of the CPU time is in the
    # actual solving of the matrix equation
    
    bcterm = pf.boundaryConditionsTerm(BC_c)

    eqn = [pf.transientTerm(c, dt, 1.0),
           -pf.diffusionTerm(D),
           pf.convectionTerm(u)]

    pf.solvePDE(c,
                bcterm,
                eqn)

    if (nplot % Nskip == 0):
        pf.visualizeCells(c)
    nplot+=1
    
# TO DO: provide a test result for pytest
# e.g. based of D_coeff, sg we can have a value for the amplitude of the 
# steady state exponential 