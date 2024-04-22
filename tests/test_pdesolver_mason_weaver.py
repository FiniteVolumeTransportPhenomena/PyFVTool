"""
Solving the Mason-Weaver equation

see: Midelet, J.; El-Sagheer, A. H.; Brown, T.; Kanaras, A. G.; 
     Werts, M. H. V. "The Sedimentation of Colloidal Nanoparticles in 
     Solution and Its Study Using Quantitative Digital Photography.",
     Part. Part. Syst. Charact. 2017, 34, 1700095. doi:10.1002/ppsc.201700095.

"""

import numpy as np
import matplotlib.pyplot as plt

import pyfvtool as pf

z_max = 1.0
D_coeff = 0.015
sg = 0.2

Nx = 100
Lx = z_max
dt = 0.01
t_simulation = 10.
Nskip = 20

maxdev_ppq = 1000. # max rel deviation in parts per 10^15

msh = pf.Grid1D(Nx, Lx)

# Solution variable (no flux BCs)
c = pf.CellVariable(msh, 1.0)
total_c = [c.domainIntegral()]

# advection field
u = pf.FaceVariable(msh, (sg,))
# closed boundaries: no flow at extremities
u.xvalue[0] = 0.0
u.xvalue[-1] = 0.0

# diffusion field
D = pf.FaceVariable(msh, D_coeff)

# prepare plot
plt.figure(1)
plt.clf()
pf.visualizeCells(c)

# time loop
it = 0

while (it*dt < t_simulation):
    # In the present implementation, also the 'constant' terms
    # (boundaryConditionsTerm, diffusionTerm and convectionTerm)
    # are re-constructed every cycle. This is done for clarity.
    # Code can be 'optimized' by constructing these terms outside
    # of the loop and store their results. The difference in performance is 
    # probably minimal, since most of the CPU time is in the
    # actual solving of the matrix equation
    eqn = [pf.transientTerm(c, dt, 1.0),
           -pf.diffusionTerm(D),
           pf.convectionTerm(u)]

    pf.solvePDE(c, eqn)
    it+=1
    total_c.append(c.domainIntegral())
    if (it % Nskip == 0):
        pf.visualizeCells(c)

plt.xlabel('depth / a.u.')
plt.ylabel('local concentration / a.u.')
plt.title('Finite-volume solution to the Mason-Weaver equation')

plt.figure(2)
plt.clf()
plt.plot(total_c)
plt.ylabel('total amount of c')
plt.xlabel('time step')
plt.ylim(0, 1.2*np.max(total_c))
plt.title('mass conservation')

total_dev = np.array(1e15*(total_c-total_c[0])/total_c[0])
plt.figure(3)
plt.clf()
plt.plot(total_dev)
plt.ylabel('deviation total amount of c [parts per 10^15]')
plt.ylim(-200, 200)
plt.xlabel('time step')
plt.title('deviation from mass conservation')

# amplitude of steady-state solution
z0 = D_coeff/sg
B = z_max/(z0*(1.0-np.exp(-z_max/z0)))

def test_mass_conservation():
    # deviation from mass conservation should be less than 1 part in 1e12
    # in this case
    assert np.max(np.abs(total_dev)) < maxdev_ppq

def test_amplitude():
    # simulation should have reach at least 90% of steady-state value
    # 
    assert np.max(c.innerCellValues) > 0.9*B
