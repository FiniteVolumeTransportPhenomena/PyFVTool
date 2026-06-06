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




## PyFVTool's inner workings: user (API) coordinate labels vs internal (private) labels

*If you are a normal user of this library, there is no need to read this section. It details the inner workings of PyFVTool. This knowledge is not required to simply use the PyFVTool as a FVM library.*

In its inner computational machinery, PyFVTool always uses an (x, y, z) convention for labeling coordinates, even for cylindrical and spherical grids. This is for historical reasons and efficient coding. Furthermore, three dimensions are always present, even in the case for 1D and 2D grids.

To avoid confusing situations on the user (API) side, PyFVTool uses conventional and logical coordinate labels towards the user, and then translates these to the appropriate internal labels. Thus: 
- for 1D Cartesian grids, there is $x$ as the user coordinate
- for 1D cylindrical grids, there is $r$
- for 2D Cartesian grids, there is $(x, y)$,  as pair of user coodinates
- for 2D cylindrical grids, there is $(r, z)$
- and so on.

The 'internal' (private) labeling uses a preceding underscore to distinguish it from the 'user' labeling. This follows the Python convention that indicates that these variables (properties) are internal to PyFVTool (private) and should not be touched by the external user.

Below, the correspondence between the (conventional) user coordinate labels and the internal (underscored) variable names is given for the different meshes.


### Cell and mesh properties

All mesh objects (subclasses of `MeshStructure`) have composite properties `cellsize`, `cellcenters` and `facecenters` which are defined along each of the coordinates defined by the specific grid of the mesh.

For instance, the mesh `Grid2D` has `Grid2D.cellsize.x`, `Grid2D.cellsize.y` ,  `Grid2D.cellcenters.x` and so on. These are the coordinates labeled according to the 'user' convention. They correspond, in this case, to internal variables `._x` and `._y` .

The correspondence between conventional user coordinate labels and the internal variable names is as given in the table.


|                   |`_x`|`_y`   |`_z` |
|-------------------|----|-------|-----|
|`Grid1D`           |`x` |       |     |
|`CylindricalGrid1D`|`r` |       |     |
|`SphericalGrid1D`  |`r` |       |     |
|`Grid2D`           |`x` |`y`    |     |
|`CylindricalGrid2D`|`r` |`z`    |     |
|`PolarGrid2D`      |`r` |`theta`|     |
|`Grid3D`           |`x` |`y`    |`z`  |
|`CylindricalGrid3D`|`r` |`theta`|`z`  |
|`SphericalGrid3D`  |`r` |`theta`|`phi`|




### FaceVariable

`FaceVariable` objects handle vectorial quantities, defined with respect to the specific mesh coordinate system. Each of the components of the vector is in a separate variable (property) of the object, referred to as xvalue, rvalue and so on. The relation between the conventional user labels of the vector components of the vector and the internal variable names is listed in the table.

|                   |`_xvalue`|`_yvalue`   |`_zvalue` |
|-------------------|---------|------------|----------|
|`Grid1D`           |`xvalue` |            |          |
|`CylindricalGrid1D`|`rvalue` |            |          |
|`SphericalGrid1D`  |`rvalue` |            |          |
|`Grid2D`           |`xvalue` |`yvalue`    |          |
|`CylindricalGrid2D`|`rvalue` |`zvalue`    |          |
|`PolarGrid2D`      |`rvalue` |`thetavalue`|          |
|`Grid3D`           |`xvalue` |`yvalue`    |`zvalue`  |
|`CylindricalGrid3D`|`rvalue` |`thetavalue`|`zvalue`  |
|`SphericalGrid3D`  |`rvalue` |`thetavalue`|`phivalue`|



