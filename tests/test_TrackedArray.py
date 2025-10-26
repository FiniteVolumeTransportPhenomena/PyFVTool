# -*- coding: utf-8 -*-
"""
Test TrackedArray in BCs and CellVariable
"""

import numpy as np

import pyfvtool as pf



successful_finish = False


# Calculation parameters
Nx = 20 # number of finite volume cells
Lx = 1.0 # [m] length of the domain 
c_init = 4.2
c_left = -8.4

# Define mesh
mesh = pf.Grid1D(Nx, Lx)

# Create a cell variable with initial concentration
# By default, 'no flux' boundary conditions are applied
c = pf.CellVariable(mesh, c_init)

# Switch the left boundary to Dirichlet: fixed concentration
c.BCs.left.a = 0.0
c.BCs.left.b = 1.0
c.BCs.left.c = c_left

assert c.BCs.left.changed==True

c.BCs.left.changed = False

assert c.BCs.left.changed==False


# Switch the right boundary to partial Dirichlet: fixed concentration
c.BCs.right.a[Nx//4:-Nx//4] = 0.0
c.BCs.right.b[Nx//4:-Nx//4] = 1.0
c.BCs.right.c[Nx//4:-Nx//4] = c_left

assert c.BCs.right.changed==True
assert c.BCs_changed == True

c.apply_BCs()

assert c.BCs_changed == False
assert c.BCs.right.changed==False




# 3D
L = 0.01  # a 1 cm domain
Nx = 10   # number of cells
m = pf.Grid3D(Nx, Nx, Nx, L, L, L)  
c = pf.CellVariable(m, 0.0)

# Now switch from Neumann boundary conditions to Dirichlet conditions:
# left boundary: homogeneous Dirichlet left-side 
c.BCs.left.a, c.BCs.left.b, c.BCs.left.c = 0.0, 1.0, 0.0
# right boundary: inhomogeneous Dirchlet right-side
c.BCs.right.a, c.BCs.right.b, c.BCs.right.c = 0.0, 1.0, 1.0

assert c.BCs.left.changed == True
assert c.BCs.right.changed == True

c.BCs.top.a[:] = 0.0
c.BCs.top.b[:] = 1.0
c.BCs.top.c[:] = 2.0

c.BCs.bottom.periodic = True

assert c.BCs.top.changed == True
assert c.BCs.bottom.changed == True

c.BCs.front.a[:] = 0.0
c.BCs.front.b = 1.0
c.BCs.front.c = 2.0
c.BCs.front.periodic = False


assert c.BCs.front.changed == True

c.BCs.back.a = 1.0
c.BCs.back.b = 1.0
c.BCs.back.c = 1.0
c.BCs.front.periodic = True

assert c.BCs.back.changed == True

assert c.BCs_changed == True

c.apply_BCs()

assert c.BCs_changed == False


successful_finish = True

# pytest
def test_success():
    assert successful_finish

