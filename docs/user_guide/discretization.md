# Discretization

PyFVTool discretizes PDE terms into sparse matrix contributions that are assembled
and solved by {func}`pyfvtool.solvePDE`.

## Available terms

### Transient term

$$\alpha \frac{\partial \phi}{\partial t} \approx \alpha \frac{\phi^{n+1} - \phi^n}{\Delta t}$$

```python
pf.transientTerm(c, dt, alpha)
```

### Diffusion term

$$\nabla \cdot (\mathcal{D} \nabla \phi)$$

```python
D_face = pf.geometricMean(D_cell)
pf.diffusionTerm(D_face)
```

### Advection term

$$\nabla \cdot (\mathbf{u} \phi)$$

```python
pf.convectionTerm(u_face)           # central difference
pf.convectionUpwindTerm(u_face)     # first-order upwind
pf.convectionTVDTerm(u_face, c, flux_limiter)  # TVD scheme
```

### Source terms

```python
pf.linearSourceTerm(beta_cell)      # β φ
pf.constantSourceTerm(gamma_cell)   # γ
```

## Averaging methods

Cell-centred values must be interpolated to faces before use in face-based terms:

| Function | Formula |
|----------|---------|
| `linearMean` | arithmetic average |
| `arithmeticMean` | arithmetic average |
| `geometricMean` | geometric average (recommended for diffusivity) |
| `harmonicMean` | harmonic average |
| `upwindMean` | upwind-biased |

## Assembling equations

Combine terms into a list and pass to `solvePDE`:

```python
eqn = [pf.transientTerm(c, dt, 1.0),
       -pf.diffusionTerm(D_face),
       pf.constantSourceTerm(S_cell)]
pf.solvePDE(c, eqn)
```

The sign convention follows the PDE form:
transient + advection + diffusion + source = 0, so diffusion typically
appears with a negative sign.
