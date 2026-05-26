# Solving PDEs

Once equation terms are assembled, pass them to `solvePDE` together with the
`CellVariable` that holds the unknown:

```python
pf.solvePDE(c, equation_terms)
```

`solvePDE` builds the global sparse matrix, applies boundary conditions, solves
the linear system, and writes the updated values back into `c` in-place.

## Nonlinear PDEs

For nonlinear problems, use **Picard iteration** (fixed-point iteration):
linearize the nonlinear coefficients around the current solution, solve, update,
and repeat until convergence:

```python
for _ in range(max_iter):
    D_face = pf.geometricMean(pf.CellVariable(mesh, D_func(c.value)))
    eqn = [-pf.diffusionTerm(D_face), pf.constantSourceTerm(S)]
    residual = pf.solvePDE(c, eqn)
    if residual < tol:
        break
```

## Time-stepping

PyFVTool uses **implicit (backward) Euler** time integration by default via
`transientTerm`. This is unconditionally stable, allowing larger time steps
than explicit schemes at the cost of solving a linear system each step.
