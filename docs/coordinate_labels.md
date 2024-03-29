# Labeling of coordinates


## User (API) coordinate labels and internal labels

*If you are a casual user of this library, you do not need to read this section. It details the inner workings of PyFVTool, but this knowledge is not required to simply use the PyFVTool as a library.*

Internally, PyFVTool always uses an (x, y, z) convention for labeling coordinates, even for cylindrical and spherical grids. This is for historical reasons and efficient coding. Also, these three dimensions are always present, even in the case for 1D and 2D grids.

To avoid confusing situations on the user (API) side, PyFVTool uses conventional coordinate labels towards the user, and then translates these to the appropriate internal labels. Thus, for 1D Cartesian grids, there is (x),  for 1D cylindrical grids, there is \(r\), for 2D Cartesian grids, there is (x, y),  for 2D cylindrical grids, there is (r, z), and so on.

The 'internal' labeling uses a preceding underscore to distinguish it from the 'user' labeling. This follows the Python convention that indicates that these variables (properties) are internal to PyFVTool (private) and should not be touched by the external user.

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

`SphericalGrid1D` has not yet been implemented.





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

`SphericalGrid1D` has not yet been implemented.


