# Cell variables

A `CellVariable` stores scalar field values at the centres of all cells in a mesh.
It is the central data structure in PyFVTool.

## Creating a cell variable

```python
import pyfvtool as pf

mesh = pf.Grid1D(20, 1.0)

# Uniform initial value
c = pf.CellVariable(mesh, 0.0)

# From a numpy array (must match mesh dimensions)
import numpy as np
c = pf.CellVariable(mesh, np.linspace(0, 1, 20))
```

## Accessing values

```python
c.value        # numpy array of cell-centre values (interior cells)
c.internalCells   # same, explicit
```

## Face variables

Many operations (diffusion coefficients, velocity fields) are defined on cell
**faces** rather than cell centres. Use `FaceVariable` or convert a `CellVariable`
using an averaging method (see [Discretization](discretization)).

```python
D = pf.CellVariable(mesh, 1e-5)
D_face = pf.geometricMean(D)   # FaceVariable
```
