# Mesh generation

import numpy as np
from warnings import warn
from typing import overload
from .utilities import int_range


class CellSize:
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        temp = vars(self)
        for item in temp:
            print(item, ':', temp[item])
        return ""

    def __repr__(self):
        temp = vars(self)
        for item in temp:
            print(item, ':', temp[item])
        return ""


class CellLocation:
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        temp = vars(self)
        for item in temp:
            print(item, ':', temp[item])
        return ""

    def __repr__(self):
        temp = vars(self)
        for item in temp:
            print(item, ':', temp[item])
        return ""


class FaceLocation:
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        temp = vars(self)
        for item in temp:
            print(item, ':', temp[item])
        return ""

    def __repr__(self):
        temp = vars(self)
        for item in temp:
            print(item, ':', temp[item])
        return ""


class MeshStructure:
    def __init__(self, dims, cellsize,
                 cellcenters, facecenters, corners, edges):
        self.dims = dims
        self.cellsize = cellsize
        self.cellcenters = cellcenters
        self.facecenters = facecenters
        self.corners = corners
        self.edges = edges

    def visualize(self):
        pass

    def shift_origin(self, x=0.0, y=0.0, z=0.0):
        self.cellcenters.x += x
        self.cellcenters.y += y
        self.cellcenters.z += z
        self.facecenters.x += x
        self.facecenters.y += y
        self.facecenters.z += z

    def __str__(self):
        temp = vars(self)
        for item in temp:
            print(item, ':', temp[item])
        return ""

    def __repr__(self):
        temp = vars(self)
        for item in temp:
            print(item, ':', temp[item])
        return ""


class Grid1D(MeshStructure):
    @overload
    def __init__(self, Nx: int, Lx: float):
        ...
    
    @overload
    def __init__(self, face_locations: np.ndarray):
        ...
        
    @overload
    def __init__(self, dims, cellsize,
                 cellcenters, facecenters, corners, edges):
            ...
    
    def __init__(self, *args):
        """Create a Grid1D mesh object from a list of cell face locations or from
        number of cells and domain length.
    
        Parameters
        ----------
        Nx : int
            Number of cells in the x direction.
        Lx : float
            Length of the domain in the x direction.
        face_locations : ndarray
            Locations of the cell faces in the x direction.
    
        Returns
        -------
        Grid1D
            A 1D mesh object.
    
        Examples
        --------
        >>> import numpy as np
        >>> from pyfvtool import Grid1D
        >>> mesh = Grid1D(10, 10.0)
        >>> print(mesh)
        """
        if (len(args)==6):
            dims, cell_size, cell_location, face_location, corners, edges\
                = args
        else:
            dims, cell_size, cell_location, face_location, corners, edges\
                = _mesh_1d_param(*args)
        super().__init__(dims, cell_size, cell_location,
                         face_location, corners, edges)


    def cell_numbers(self):
        Nx = self.dims[0]
        return int_range(0, Nx+1)

    def __repr__(self):
        print(f"1D Cartesian mesh with {self.dims[0]} cells")
        return ""


class Mesh2D(MeshStructure):
    def __init__(self, dims, cell_size, cell_location, face_location, corners, edges):
        super().__init__(dims, cell_size, cell_location,
                         face_location, corners, edges)

    def cell_numbers(self):
        Nx, Ny = self.dims
        G = int_range(0, (Nx+2)*(Ny+2)-1)
        return G.reshape(Nx+2, Ny+2)

    def __repr__(self):
        print(f"2D Cartesian mesh with {self.dims[0]}x{self.dims[1]} cells")
        return ""


class Mesh3D(MeshStructure):
    def __init__(self, dims, cell_size, cell_location, face_location, corners, edges):
        super().__init__(dims, cell_size, cell_location,
                         face_location, corners, edges)

    def cell_numbers(self):
        Nx, Ny, Nz = self.dims
        G = int_range(0, (Nx+2)*(Ny+2)*(Nz+2)-1)
        return G.reshape(Nx+2, Ny+2, Nz+2)

    def __repr__(self):
        print(
            f"3D Cartesian mesh with Nx={self.dims[0]}xNy={self.dims[1]}xNz={self.dims[1]} cells")
        return ""


class MeshCylindrical1D(Grid1D):
    def __init__(self, dims, cell_size, cell_location, face_location, corners, edges):
        super().__init__(dims, cell_size, cell_location,
                         face_location, corners, edges)

    def __repr__(self):
        print(f"1D Cylindrical (radial) mesh with Nr={self.dims[0]} cells")
        return ""


class MeshSpherical1D(Grid1D):
    def __init__(self, dims, cell_size, cell_location, face_location, corners, edges):
        super().__init__(dims, cell_size, cell_location,
                         face_location, corners, edges)

    def __repr__(self):
        print(f"1D Spherical mesh with Nr={self.dims[0]} cells")
        return ""


class MeshCylindrical2D(Mesh2D):
    def __init__(self, dims, cell_size, cell_location, face_location, corners, edges):
        super().__init__(dims, cell_size, cell_location,
                         face_location, corners, edges)

    def __repr__(self):
        print(
            f"2D Cylindrical mesh with Nr={self.dims[0]}xNz={self.dims[1]} cells")
        return ""


class MeshRadial2D(Mesh2D):
    def __init__(self, dims, cell_size, cell_location, face_location, corners, edges):
        super().__init__(dims, cell_size, cell_location,
                         face_location, corners, edges)

    def __repr__(self):
        print(
            f"2D Radial mesh with Nr={self.dims[0]}xN_theta={self.dims[1]} cells")
        return ""


class MeshCylindrical3D(Mesh3D):
    def __init__(self, dims, cell_size, cell_location, face_location, corners, edges):
        super().__init__(dims, cell_size, cell_location,
                         face_location, corners, edges)

    def __repr__(self):
        print(
            f"3D Cylindrical mesh with Nr={self.dims[0]}xN_theta={self.dims[1]}xNz={self.dims[1]} cells")
        return ""


class MeshSpherical3D(Mesh3D):
    def __init__(self, dims, cell_size, cell_location, face_location, corners, edges):
        super().__init__(dims, cell_size, cell_location,
                         face_location, corners, edges)

    def __repr__(self):
        print(
            f"3D Shperical mesh with Nr={self.dims[0]}xN_theta={self.dims[1]}xN_phi={self.dims[1]} cells")
        return ""


def _facelocation_to_cellsize(facelocation):
    return np.hstack([facelocation[1]-facelocation[0],
                      facelocation[1:]-facelocation[0:-1],
                      facelocation[-1]-facelocation[-2]])


def _mesh_1d_param(*args):
    if len(args) == 1:
        # Use face locations
        facelocationX = args[0]
        Nx = facelocationX.size-1
        cell_size_x = np.hstack([facelocationX[1]-facelocationX[0],
                                 facelocationX[1:]-facelocationX[0:-1],
                                 facelocationX[-1]-facelocationX[-2]])
        cell_size = CellSize(cell_size_x, np.array([0.0]), np.array([0.0]))
        cell_location = CellLocation(
            0.5*(facelocationX[1:]+facelocationX[0:-1]), np.array([0.0]), np.array([0.0]))
        face_location = FaceLocation(
            facelocationX, np.array([0.0]), np.array([0.0]))
    elif len(args) == 2:
        # Use number of cells and domain length
        Nx = args[0]
        Width = args[1]
        dx = Width/Nx
        cell_size = CellSize(
            dx*np.ones(Nx+2), np.array([0.0]), np.array([0.0]))
        cell_location = CellLocation(
            int_range(1, Nx)*dx-dx/2,
            np.array([0.0]),
            np.array([0.0]))
        face_location = FaceLocation(
            int_range(0, Nx)*dx,
            np.array([0.0]),
            np.array([0.0]))
    dims = np.array([Nx], dtype=int)
    cellsize = cell_size
    cellcenters = cell_location
    facecenters = face_location
    corners = np.array([1], dtype=int)
    edges = np.array([1], dtype=int)
    return dims, cellsize, cellcenters, facecenters, corners, edges


def _mesh_2d_param(*args):
    if len(args) == 2:
        # Use face locations
        facelocationX = args[0]
        facelocationY = args[1]
        Nx = facelocationX.size-1
        Ny = facelocationY.size-1
        cell_size = CellSize(_facelocation_to_cellsize(facelocationX),
                             _facelocation_to_cellsize(facelocationY),
                             np.array([0.0]))
        cell_location = CellLocation(
            0.5*(facelocationX[1:]+facelocationX[0:-1]),
            0.5*(facelocationY[1:]+facelocationY[0:-1]),
            np.array([0.0]))
        face_location = FaceLocation(
            facelocationX,
            facelocationY,
            np.array([0.0]))
    elif len(args) == 4:
        # Use number of cells and domain length
        Nx = args[0]
        Ny = args[1]
        Width = args[2]
        Height = args[3]
        dx = Width/Nx
        dy = Height/Ny
        cell_size = CellSize(
            dx*np.ones(Nx+2),
            dy*np.ones(Ny+2),
            np.array([0.0]))
        cell_location = CellLocation(
            int_range(1, Nx)*dx-dx/2,
            int_range(1, Ny)*dy-dy/2,
            np.array([0.0]))
        face_location = FaceLocation(
            int_range(0, Nx)*dx,
            int_range(0, Ny)*dy,
            np.array([0.0]))

    dims = np.array([Nx, Ny], dtype=int)
    cellsize = cell_size
    cellcenters = cell_location
    facecenters = face_location
    G = int_range(1, (Nx+2)*(Ny+2))-1
    corners = G.reshape(Nx+2, Ny+2)[[0, -1, 0, -1], [0, 0, -1, -1]]
    edges = np.array([1], dtype=int)
    return dims, cellsize, cellcenters, facecenters, corners, edges


def _mesh_3d_param(*args):
    if len(args) == 3:
        # Use face locations
        facelocationX = args[0]
        facelocationY = args[1]
        facelocationZ = args[2]
        Nx = facelocationX.size-1
        Ny = facelocationY.size-1
        Nz = facelocationZ.size-1
        cell_size = CellSize(_facelocation_to_cellsize(facelocationX),
                             _facelocation_to_cellsize(facelocationY),
                             _facelocation_to_cellsize(facelocationZ))
        cell_location = CellLocation(
            0.5*(facelocationX[1:]+facelocationX[0:-1]),
            0.5*(facelocationY[1:]+facelocationY[0:-1]),
            0.5*(facelocationZ[1:]+facelocationZ[0:-1]))
        face_location = FaceLocation(
            facelocationX,
            facelocationY,
            facelocationZ)
    elif len(args) == 6:
        # Use number of cells and domain length
        Nx = args[0]
        Ny = args[1]
        Nz = args[2]
        Width = args[3]
        Height = args[4]
        Depth = args[5]
        dx = Width/Nx
        dy = Height/Ny
        dz = Depth/Nz
        cell_size = CellSize(
            dx*np.ones(Nx+2),
            dy*np.ones(Ny+2),
            dz*np.ones(Nz+2))
        cell_location = CellLocation(
            int_range(1, Nx)*dx-dx/2,
            int_range(1, Ny)*dy-dy/2,
            int_range(1, Nz)*dz-dz/2)
        face_location = FaceLocation(
            int_range(0, Nx)*dx,
            int_range(0, Ny)*dy,
            int_range(0, Nz)*dz)
    G = int_range(1, (Nx+2)*(Ny+2)*(Nz+2))-1
    G = G.reshape(Nx+2, Ny+2, Nz+2)
    dims = np.array([Nx, Ny, Nz], dtype=int)
    cellsize = cell_size
    cellcenters = cell_location
    facecenters = face_location
    corners = G[np.ix_((0, -1), (0, -1), (0, -1))].flatten()
    edges = np.hstack([G[0, [0, -1], 1:-1].flatten(),
                       G[-1, [0, -1], 1:-1].flatten(),
                       G[0, 1:-1, [0, -1]].flatten(),
                       G[-1, 1:-1, [0, -1]].flatten(),
                       G[1:-1, 0, [0, -1]].flatten(),
                       G[1:-1, -1, [0, -1]].flatten()])
    return dims, cellsize, cellcenters, facecenters, corners, edges



@overload
def createMesh2D(Nx: int, Ny: int, Lx: float, Ly: float) -> Mesh2D:
    ...


@overload
def createMesh2D(face_locationsX: np.ndarray,
                 face_locationsY: np.ndarray) -> Mesh2D:
    ...


def createMesh2D(*args) -> Mesh2D:
    """Create a Mesh2D object from a list of cell face locations or from
    number of cells and domain length.

    Parameters
    ----------
    Nx : int
        Number of cells in the x direction.
    Ny : int
        Number of cells in the y direction.
    Lx : float
        Length of the domain in the x direction.
    Ly : float
        Length of the domain in the y direction.
    face_locationsX : ndarray
        Locations of the cell faces in the x direction.
    face_locationsY : ndarray
        Locations of the cell faces in the y direction.

    Returns
    -------
    Mesh2D
        A 2D mesh object.

    Examples
    --------
    >>> import numpy as np
    >>> from pyfvtool import createMesh2D
    >>> mesh = createMesh2D(10, 10, 10.0, 10.0)
    >>> print(mesh)
    """
    dims, cellsize, cellcenters, facecenters, corners, edges = _mesh_2d_param(
        *args)
    return Mesh2D(dims, cellsize, cellcenters, facecenters, corners, edges)


@overload
def createMesh3D(Nx: int, Ny: int, Nz: int,
                 Lx: float, Ly: float, Lz: float) -> Mesh3D:
    ...


@overload
def createMesh3D(face_locationsX: np.ndarray,
                 face_locationsY: np.ndarray, face_locationsZ: np.ndarray) -> Mesh3D:
    ...


def createMesh3D(*args) -> Mesh3D:
    """
    Create a Mesh3D object from a list of cell face locations or from
    number of cells and domain length.
    
    Parameters
    ----------
    Nx : int
        Number of cells in the x direction.
        Ny : int
        Number of cells in the y direction.
        Nz : int
        Number of cells in the z direction.
        Lx : float
        Length of the domain in the x direction.
        Ly : float
        Length of the domain in the y direction.
        Lz : float
        Length of the domain in the z direction.
        face_locationsX : ndarray
        Locations of the cell faces in the x direction.
        face_locationsY : ndarray
        Locations of the cell faces in the y direction.
        face_locationsZ : ndarray
        Locations of the cell faces in the z direction.
        
        Returns
        -------
        Mesh3D
        A 3D mesh object.
            
        Examples
        --------
        >>> import numpy as np
        >>> from pyfvtool import createMesh3D
        >>> mesh = createMesh3D(10, 10, 10, 10.0, 10.0, 10.0)
        >>> print(mesh)
        """
    dims, cellsize, cellcenters, facecenters, corners, edges = _mesh_3d_param(
        *args)
    return Mesh3D(dims, cellsize, cellcenters, facecenters, corners, edges)


@overload
def createMeshCylindrical1D(Nx: int, Lx: float) -> MeshCylindrical1D:
    ...


@overload
def createMeshCylindrical1D(face_locations: np.ndarray) -> MeshCylindrical1D:
    ...


def createMeshCylindrical1D(*args) -> MeshCylindrical1D:
    """Create a MeshCylindrical1D object from a list of cell face locations or from
    number of cells and domain length.
    
    Parameters
    ----------
    Nx : int
        Number of cells in the x direction.
    Lx : float
        Length of the domain in the x direction.
    face_locations : ndarray
        Locations of the cell faces in the x direction.
        
    Returns
    -------
    MeshCylindrical1D
        A 1D cylindrical mesh object.
    
    Examples
    --------
    >>> import numpy as np
    >>> from pyfvtool import createMeshCylindrical1D
    >>> mesh = createMeshCylindrical1D(10, 10.0)
    >>> print(mesh)
    """
    dims, cellsize, cellcenters, facecenters, corners, edges = _mesh_1d_param(
        *args)
    return MeshCylindrical1D(dims, cellsize, cellcenters, facecenters, corners, edges)


@overload
def createMeshCylindrical2D(Nx: int, Ny: int,
                            Lx: float, Ly: float) -> MeshCylindrical2D:
    ...


@overload
def createMeshCylindrical2D(face_locationsX: np.ndarray,
                            face_locationsY: np.ndarray) -> MeshCylindrical2D:
    ...


def createMeshCylindrical2D(*args) -> MeshCylindrical2D:
    """Create a MeshCylindrical2D object from a list of cell face locations or from
    number of cells and domain length.
    
    Parameters
    ----------
    Nx : int
        Number of cells in the x direction.
    Ny : int
        Number of cells in the y direction.
    Lx : float
        Length of the domain in the x direction.
    Ly : float
        Length of the domain in the y direction.
    face_locationsX : ndarray
        Locations of the cell faces in the x direction.
    face_locationsY : ndarray
        Locations of the cell faces in the y direction.
    
    Returns
    -------
    MeshCylindrical2D
        A 2D cylindrical mesh object.
        
    Examples
    --------
    >>> import numpy as np
    >>> from pyfvtool import createMeshCylindrical2D
    >>> mesh = createMeshCylindrical2D(10, 10, 10.0, 10.0)
    >>> print(mesh)
    """
    dims, cellsize, cellcenters, facecenters, corners, edges = _mesh_2d_param(
        *args)
    return MeshCylindrical2D(dims, cellsize, cellcenters, facecenters, corners, edges)


@overload
def createMeshCylindrical3D(Nx: int, Ny: int,
                            Nz: int, Lx: float,
                            Ly: float, Lz: float) -> MeshCylindrical3D:
    ...


@overload
def createMeshCylindrical3D(face_locationsX: np.ndarray,
                            face_locationsY: np.ndarray,
                            face_locationsZ: np.ndarray) -> MeshCylindrical3D:
    ...


def createMeshCylindrical3D(*args) -> MeshCylindrical3D:
    """
    Create a MeshCylindrical3D object from a list of cell face locations or from
    number of cells and domain length.

    Parameters
    ----------
    Nx : int
        Number of cells in the x direction.
    Ny : int
        Number of cells in the y direction.
    Nz : int
        Number of cells in the z direction.
    Lx : float
        Length of the domain in the x direction.
    Ly : float
        Length of the domain in the y direction.
    Lz : float
        Length of the domain in the z direction.
    face_locationsX : ndarray
        Locations of the cell faces in the x direction.
    face_locationsY : ndarray
        Locations of the cell faces in the y direction.
    face_locationsZ : ndarray
        Locations of the cell faces in the z direction.
    
    Returns
    -------
    MeshCylindrical3D
        A 3D cylindrical mesh object.

    Examples
    --------
    >>> import numpy as np
    >>> from pyfvtool import createMeshCylindrical3D
    >>> mesh = createMeshCylindrical3D(10, 10, 10, 10.0, 10.0, 10.0)
    >>> print(mesh)

    Notes
    -----
    The mesh is created in cylindrical coordinates.
    """
    if len(args) == 3:
        theta_max = args[1][-1]
    else:
        theta_max = args[4]
    if theta_max > 2*np.pi:
        warn("Recreate the mesh with an upper bound of 2*pi for theta or there will be unknown consequences!")
    dims, cellsize, cellcenters, facecenters, corners, edges = _mesh_3d_param(
        *args)
    return MeshCylindrical3D(dims, cellsize, cellcenters, facecenters, corners, edges)


@overload
def createMeshSpherical1D(Nx: int, Lx: float) -> MeshSpherical1D:
    ...

@overload
def createMeshSpherical1D(face_locations: np.ndarray) -> MeshSpherical1D:
    ...


def createMeshSpherical1D(*args) -> MeshSpherical1D:
    """
    Create a MeshSpherical1D object from a list of cell face locations or from
    number of cells and domain length.

    Parameters
    ----------
    Nx : int
        Number of cells in the x direction.
    Lx : float
        Length of the domain in the x direction.
    face_locations : ndarray
        Locations of the cell faces in the x direction.
    
    Returns
    -------
    MeshSpherical1D
        A 1D spherical mesh object.

    Examples
    --------
    >>> import numpy as np
    >>> from pyfvtool import createMeshSpherical1D
    >>> mesh = createMeshSpherical1D(10, 10.0)
    >>> print(mesh)

    Notes
    -----
    The mesh is created in spherical coordinates.
    """
    dims, cellsize, cellcenters, facecenters, corners, edges = _mesh_1d_param(
        *args)
    return MeshSpherical1D(dims, cellsize, cellcenters, facecenters, corners, edges)


@overload
def createMeshRadial2D(Nx: int, Ny: int, Lx: float, Ly: float) -> MeshRadial2D:
    ...


@overload
def createMeshRadial2D(face_locationsX: np.ndarray,
                       face_locationsY: np.ndarray) -> MeshRadial2D:
    ...


def createMeshRadial2D(*args) -> MeshRadial2D:
    """
    Create a MeshRadial2D object from a list of cell face locations or from
    number of cells and domain length.
    
    Parameters
    ----------
    Nx : int
        Number of cells in the x direction.
    Ny : int
        Number of cells in the y direction.
    Lx : float
        Length of the domain in the x direction.
    Ly : float
        Length of the domain in the y direction.
    face_locationsX : ndarray
        Locations of the cell faces in the x direction.
    face_locationsY : ndarray
        Locations of the cell faces in the y direction.
    
    Returns
    -------
    MeshRadial2D
        A 2D radial mesh object.
    
    Examples
    --------
    >>> import numpy as np
    >>> from pyfvtool import createMeshRadial2D
    >>> mesh = createMeshRadial2D(10, 10, 10.0, 10.0)
    >>> print(mesh)

    Notes
    -----
    The mesh is created in radial (cylindrical) coordinates.
    """
    if len(args) == 2:
        theta_max = args[1][-1]
    else:
        theta_max = args[3]
    if theta_max > 2*np.pi:
        warn("Recreate the mesh with an upper bound of 2*pi for \theta or there will be unknown consequences!")
    dims, cellsize, cellcenters, facecenters, corners, edges = _mesh_2d_param(
        *args)
    return MeshRadial2D(dims, cellsize, cellcenters, facecenters, corners, edges)


@overload
def createMeshSpherical3D(Nx: int, Ny: int, Nz: int,
                          Lx: float, Ly: float, Lz: float) -> MeshSpherical3D:
    ...

@overload
def createMeshSpherical3D(face_locationsX: np.ndarray, face_locationsY: np.ndarray,
                          face_locationsZ: np.ndarray) -> MeshSpherical3D:
    ...


def createMeshSpherical3D(*args) -> MeshSpherical3D:
    """
    Create a MeshSpherical3D object from a list of cell face locations or from
    number of cells and domain length.

    Parameters
    ----------
    Nx : int
        Number of cells in the x direction.
    Ny : int
        Number of cells in the y direction.
    Nz : int
        Number of cells in the z direction.
    Lx : float
        Length of the domain in the x direction.
    Ly : float
        Length of the domain in the y direction.
    Lz : float
        Length of the domain in the z direction.
    face_locationsX : ndarray
        Locations of the cell faces in the x direction.
    face_locationsY : ndarray
        Locations of the cell faces in the y direction.
    face_locationsZ : ndarray
        Locations of the cell faces in the z direction.
    
    Returns
    -------
    MeshSpherical3D
        A 3D spherical mesh object.
    
    Examples
    --------
    >>> import numpy as np
    >>> from pyfvtool import createMeshSpherical3D
    >>> mesh = createMeshSpherical3D(10, 10, 10, 10.0, 10.0, 10.0)
    >>> print(mesh)

    Notes
    -----
    The mesh is created in spherical coordinates.
    """
    if args[4] > 2*np.pi:
        warn("Recreate the mesh with an upper bound of 2*pi for \\theta"\
             " or there will be unknown consequences!")
    if args[5] > 2*np.pi:
        warn("Recreate the mesh with an upper bound of 2*pi for \\phi"\
             " or there will be unknown consequences!")
    dims, cellsize, cellcenters, facecenters, corners, edges =\
        _mesh_3d_param(*args)
    return MeshSpherical3D(dims, cellsize, cellcenters,
                           facecenters, corners, edges)
