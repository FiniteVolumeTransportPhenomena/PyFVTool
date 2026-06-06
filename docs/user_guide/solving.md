# Solving PDEs

Once equation terms are assembled, pass them to `solvePDE` together with the
`CellVariable` that holds the unknown:

```python
pf.solvePDE(c, equation_terms)
```

`solvePDE` builds the global sparse matrix, applies boundary conditions, solves
the linear system, and writes the updated values back into `c` in-place.


## Time-stepping

PyFVTool uses **implicit (backward) Euler** time integration by default via
`transientTerm`. This is unconditionally stable, allowing larger time steps
than explicit schemes at the cost of solving a linear system each step.



## Nonlinear PDEs

For nonlinear problems, **Picard iteration** (fixed-point iteration) could be used:
linearize the nonlinear coefficients around the current solution, solve, update,
and repeat until convergence. [FiPy calls this "sweeping"](https://pages.nist.gov/fipy/en/latest/FAQ.html#iterations-timesteps-and-sweeps-oh-my). In other FVM literature, this may be referred to as "nonlinear iterations".

With PyFVTool, Picard iteration has not yet been used, and there are at present no examples of solving nonlinear PDEs (hello, Navier-Stokes!) in PyFVTool.
