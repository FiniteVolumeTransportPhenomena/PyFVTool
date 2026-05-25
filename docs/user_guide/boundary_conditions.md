# Boundary conditions

PyFVTool uses a **unified Robin boundary condition** form for all boundaries:

$$a \,\nabla\phi \cdot \mathbf{e} + b\,\phi = c$$

where $\mathbf{e}$ is the outward unit normal. By choosing $a$, $b$, and $c$ you
recover all common boundary condition types:

| BC type | $a$ | $b$ | $c$ |
|---------|-----|-----|-----|
| Dirichlet (fixed value $\phi_0$) | `0` | `1` | `φ₀` |
| Neumann (fixed flux $q$) | `1` | `0` | `q` |
| No-flux (closed boundary) | `1` | `0` | `0` |
| Robin | any | any | any |

The default boundary condition on all faces is **no-flux** ($a=1$, $b=0$, $c=0$).

## Setting boundary conditions

Boundary conditions are stored on the `CellVariable` object, in its `.BCs` attribute.

```python
c = pf.CellVariable(mesh, initial_value)
```

For a 1D mesh, the available boundaries are `.left` and `.right`:

```python
# Dirichlet: fix concentration to 1.0 at the left boundary
c.BCs.left.a = 0.0
c.BCs.left.b = 1.0
c.BCs.left.c = 1.0

# Neumann: fixed flux of 0.5 at the right boundary
c.BCs.right.a = 1.0
c.BCs.right.b = 0.0
c.BCs.right.c = 0.5
```

For 2D meshes, the available boundaries are `.left`, `.right`, `.bottom`, `.top`.
For 3D meshes, `.back` and `.front` are added.

## Periodic boundary conditions

Periodic BCs can be specified where appropriate.
Consult the examples notebooks for details.
