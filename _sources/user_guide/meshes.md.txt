# Meshes

PyFVTool supports structured (regular) meshes in 1D, 2D, and 3D,
in Cartesian, cylindrical, and spherical coordinate systems.

All mesh classes are imported from `pyfvtool` directly:

```python
import pyfvtool as pf
```

## 1D meshes

| Class | Coordinate system |
|-------|------------------|
| `Grid1D` | Cartesian ($x$) |
| `CylindricalGrid1D` | Cylindrical ($r$) |
| `SphericalGrid1D` | Spherical ($r$) |

```python
mesh = pf.Grid1D(Nx, Lx)           # Nx cells over [0, Lx]
mesh = pf.CylindricalGrid1D(Nr, Lr) # radial direction
mesh = pf.SphericalGrid1D(Nr, Lr)
```

## 2D meshes

| Class | Coordinate system |
|-------|------------------|
| `Grid2D` | Cartesian ($x, y$) |
| `CylindricalGrid2D` | Cylindrical ($r, z$) |

```python
mesh = pf.Grid2D(Nx, Ny, Lx, Ly)
mesh = pf.CylindricalGrid2D(Nr, Nz, Lr, Lz)
```

## 3D meshes

| Class | Coordinate system |
|-------|------------------|
| `Grid3D` | Cartesian ($x, y, z$) |
| `CylindricalGrid3D` | Cylindrical ($r, \theta, z$) |

```python
mesh = pf.Grid3D(Nx, Ny, Nz, Lx, Ly, Lz)
```

## Non-uniform grids

You can pass an array of cell face positions instead of a uniform length,
giving full control over cell spacing:

```python
import numpy as np
x_faces = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0])  # non-uniform spacing
mesh = pf.Grid1D(x_faces)
```

## Mesh properties

Once created, a mesh object exposes geometric data needed for discretization.
You generally don't access these directly, but they are available:

```python
mesh.dims        # tuple: (Nx,) or (Nx, Ny) etc.
mesh.cellsize    # cell dimensions
mesh.cellcenters # coordinates of cell centres
mesh.facecenters # coordinates of cell face centres
```
