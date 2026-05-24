# Quickstart

This page walks through a minimal working example: a 1D transient diffusion equation.

## The problem

We solve the diffusion equation

$$\frac{\partial c}{\partial t} = D \frac{\partial^2 c}{\partial x^2}$$

on a 1D domain $[0, L]$ with a fixed concentration $c = 1$ at the left boundary
and a no-flux (closed) boundary on the right. The initial concentration is zero everywhere.

## Code

```python
import pyfvtool as pf

# Parameters
Nx = 20          # number of finite volume cells
Lx = 1.0         # [m] domain length
c_left = 1.0     # left boundary concentration
c_init = 0.0     # initial concentration
D_val = 1e-5     # [m²/s] diffusion coefficient
t_end = 7200.0   # [s] simulation time
dt = 60.0        # [s] time step

# 1. Define the mesh
mesh = pf.Grid1D(Nx, Lx)

# 2. Create the cell variable (concentration)
#    By default, no-flux boundary conditions are applied on all boundaries
c = pf.CellVariable(mesh, c_init)

# 3. Set the left boundary to a fixed (Dirichlet) condition
c.BCs.left.a = 0.0
c.BCs.left.b = 1.0
c.BCs.left.c = c_left

# 4. Assign diffusivity and compute face-averaged values
D_cell = pf.CellVariable(mesh, D_val)
D_face = pf.geometricMean(D_cell)

# 5. Time loop
t = 0
while t < t_end:
    eqn = [pf.transientTerm(c, dt, 1.0),
           -pf.diffusionTerm(D_face)]
    pf.solvePDE(c, eqn)
    t += dt

# 6. Visualize
pf.visualizeCells(c)
```

## What each step does

| Step | Function | Purpose |
|------|----------|---------|
| 1 | `Grid1D` | Creates a uniform 1D Cartesian mesh |
| 2 | `CellVariable` | Stores values at cell centres; holds boundary conditions |
| 3 | `BCs` attributes | Specifiy BCs. Here, fixed-value (Dirichlet) on the left. |
| 4 | `geometricMean` | Interpolates cell values to face values |
| 5 | `transientTerm`, `diffusionTerm`, `solvePDE` | Assembles and solves the linear system |
| 6 | `visualizeCells` | Plots the cell-centred values |

## Next steps

- Browse the [Examples](examples/index) for more complete use cases.
- See [Meshes](user_guide/meshes) for 2D and 3D grids, cylindrical, and spherical coordinates.
- See [Boundary conditions](user_guide/boundary_conditions) for Dirichlet, Neumann, Robin, and periodic BCs.

