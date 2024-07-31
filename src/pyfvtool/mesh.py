# Mesh generation

import numpy as np
from warnings import warn
from typing import overload
from .utilities import int_range


#%%
#   General data structures for describing meshes


class CellProp:
    def __init__(self, _x: np.ndarray, _y: np.ndarray, _z: np.ndarray,
                 coordlabels: dict):
        self._x = _x
        self._y = _y
        self._z = _z
        self.coordlabels = coordlabels

    def __str__(self):
        temp = vars(self)
        result = ""
        for item in temp:
            result += f"{item}: {temp[item]}\n"
        return result
    
    def __repr__(self):
        return str(self)
    
    
    # The following coordinate-labeling properties should probably be read-only.
    # For this reason, the 'setters' have been commented out, but are kept for
    # future reference.

    @property
    def x(self):
        if 'x' in self.coordlabels:
            if self.coordlabels['x']=='_x':
                return self._x
            else:
                raise AttributeError("Unexpectedly, user label 'x' does not correspond to '_x'")
        else:
            raise AttributeError("This mesh has no coordinate labeled 'x'.")
            
    # @x.setter
    # def x(self, value):
    #     if 'x' in self.coordlabels:
    #         if self.coordlabels['x']=='_x':
    #             self._x = value
    #         else:
    #             raise AttributeError("Unexpectedly, user label 'x' does not correspond to '_x'")
    #     else:
    #         raise AttributeError("This mesh has no coordinate labeled 'x'.")

    @property
    def r(self):
        if 'r' in self.coordlabels:
            if self.coordlabels['r']=='_x':
                return self._x
            else:
                raise AttributeError("Unexpectedly, user label 'r' does not correspond to '_x'")
        else:
            raise AttributeError("This mesh has no coordinate labeled 'r'.")
            
    # @r.setter
    # def r(self, value):
    #     if 'r' in self.coordlabels:
    #         if self.coordlabels['r']=='_x':
    #             self._x = value
    #         else:
    #             raise AttributeError("Unexpectedly, user label 'r' does not correspond to '_x'")
    #     else:
    #         raise AttributeError("This mesh has no coordinate labeled 'r'.")
    
    @property
    def z(self):
        if 'z' in self.coordlabels:
            if self.coordlabels['z']=='_y':
                return self._y
            elif self.coordlabels['z']=='_z':
                return self._z
            else:
                raise AttributeError(f"Unexpected label correspondence: 'z' -> '{self.coordlabels['z']}'")
        else:
            raise AttributeError("This mesh has no coordinate labeled 'z'.")
            
    # @z.setter
    # def z(self, value):
    #     if 'z' in self.coordlabels:
    #         if self.coordlabels['z']=='_y':
    #             self._y = value
    #         if self.coordlabels['z']=='_z':
    #             self._z = value
    #         else:
    #             raise AttributeError(f"Unexpected label correspondence: 'z' -> '{self.coordlabels['z']}'")
    #     else:
    #         raise AttributeError("This mesh has no coordinate labeled 'z'.")

    @property
    def y(self):
        if 'y' in self.coordlabels:
            if self.coordlabels['y']=='_y':
                return self._y
            else:
                raise AttributeError(f"Unexpected label correspondence: 'y' -> '{self.coordlabels['y']}'")
        else:
            raise AttributeError("This mesh has no coordinate labeled 'y'.")
            
    # @y.setter
    # def y(self, value):
    #     if 'y' in self.coordlabels:
    #         if self.coordlabels['y']=='_y':
    #             self._y = value
    #         else:
    #             raise AttributeError(f"Unexpected label correspondence: 'y' -> '{self.coordlabels['y']}'")
    #     else:
    #         raise AttributeError("This mesh has no coordinate labeled 'y'.")

    @property
    def theta(self):
        if 'theta' in self.coordlabels:
            if self.coordlabels['theta']=='_y':
                return self._y
            else:
                raise AttributeError(f"Unexpected label correspondence: 'theta' -> '{self.coordlabels['y']}'")
        else:
            raise AttributeError("This mesh has no coordinate labeled 'theta'.")
            
    # @theta.setter
    # def theta(self, value):
    #     if 'theta' in self.coordlabels:
    #         if self.coordlabels['theta']=='_y':
    #             self._y = value
    #         else:
    #             raise AttributeError(f"Unexpected label correspondence: 'theta' -> '{self.coordlabels['y']}'")
    #     else:
    #         raise AttributeError("This mesh has no coordinate labeled 'theta'.")

    @property
    def phi(self):
        if 'phi' in self.coordlabels:
            if self.coordlabels['phi']=='_z':
                return self._z
            else:
                raise AttributeError(f"Unexpected label correspondence: 'phi' -> '{self.coordlabels['y']}'")
        else:
            raise AttributeError("This mesh has no coordinate labeled 'phi'.")
            
    # @phi.setter
    # def phi(self, value):
    #     if 'phi' in self.coordlabels:
    #         if self.coordlabels['phi']=='_z':
    #             self._z = value
    #         else:
    #             raise AttributeError(f"Unexpected label correspondence: 'phi' -> '{self.coordlabels['y']}'")
    #     else:
    #         raise AttributeError("This mesh has no coordinate labeled 'phi'.")




class CellSize(CellProp):
    pass


class CellLocation(CellProp):
    pass


class FaceLocation(CellProp):
    pass



class MeshStructure:
    def __init__(self, dims, cellsize,
                 cellcenters, facecenters, corners, edges):
        self.dims = dims
        self.cellsize = cellsize
        self.cellcenters = cellcenters
        self.facecenters = facecenters
        self.corners = corners
        self.edges = edges

    def __str__(self):
        temp = vars(self)
        result = ""
        for item in temp:
            result += f"{item}: {temp[item]}\n"
        return result

    def _facelocation_to_cellsize(self, facelocation):
        return np.hstack([facelocation[1]-facelocation[0],
                          facelocation[1:]-facelocation[0:-1],
                          facelocation[-1]-facelocation[-2]])

    def _getCellVolumes(self):
        """Get the volumes of all finite volume cells in the mesh
        
        Returns
        -------
        np.ndarray
            containing all cell volumes, arranged according to gridcells
        
        TODO: move each of these to the respective grid classes?
        """
        if (type(self) is Grid1D):
            c = self.cellsize._x[1:-1]
        elif (type(self) is CylindricalGrid1D):
            c = 2.0*np.pi*self.cellsize._x[1:-1]*self.cellcenters._x
        elif (type(self) is SphericalGrid1D):
            c = 4.0*np.pi*self.cellsize._x[1:-1]*self.cellcenters._x**2
        elif (type(self) is Grid2D):
            c = self.cellsize._x[1:-1][:, np.newaxis]\
                *self.cellsize._y[1:-1][np.newaxis, :]
        elif (type(self) is CylindricalGrid2D):
            c = 2.0*np.pi*self.cellcenters._x[:, np.newaxis]\
                *self.cellsize._x[1:-1][:, np.newaxis]\
                *self.cellsize._y[1:-1][np.newaxis, :]
        elif (type(self) is PolarGrid2D):
            c = self.cellcenters._x\
                *self.cellsize._x[1:-1][:, np.newaxis]\
                *self.cellsize._y[1:-1][np.newaxis, :]
        elif (type(self) is Grid3D):
            c = self.cellsize._x[1:-1][:,np.newaxis,np.newaxis]\
                *self.cellsize._y[1:-1][np.newaxis,:,np.newaxis]\
                *self.cellsize._z[1:-1][np.newaxis,np.newaxis,:]
        elif (type(self) is CylindricalGrid3D):
            c = self.cellcenters._x\
                *self.cellsize._x[1:-1][:,np.newaxis,np.newaxis]\
                *self.cellsize._y[1:-1][np.newaxis,:,np.newaxis]\
                *self.cellsize._z[np.newaxis,np.newaxis,:]
        elif (type(self) is SphericalGrid3D):
            c = self.cellcenters._x**2\
                *self.cellsize._x[1:-1][:,np.newaxis,np.newaxis]\
                *self.cellsize._y[1:-1][np.newaxis,:,np.newaxis]\
                *self.cellsize._z[np.newaxis,np.newaxis,:]
            warn("SphericalGrid3D: cell volumes might not be correct!") # TODO: Check these volumes
        return c
    
    # read-only property cellvolume
    @property
    def cellvolume(self):
        return self._getCellVolumes()
    


#%%
#   1D Grids


class Grid1D(MeshStructure):
    """Mesh based on a 1D Cartesian grid (x)
    =====================================
    
    This class can be instantiated in different ways: from a list of cell face
    locations or from the number of cells and domain length.
    
    Instantiation Options:
    ----------------------
    - Grid1D(Nx, Lx)
    - Grid1D(face_locationsX)
    
    
    Parameters
    ----------
    Grid1D(Nx, Lx)
        Nx : int
            Number of cells in the x direction.
        Lx : float
            Length of the domain in the x direction.
    
    Grid1D(face_locationsX)
        face_locationsX : ndarray
            Locations of the cell faces in the x direction.
    
    Examples
    --------
    >>> import numpy as np
    >>> from pyfvtool import Grid1D
    >>> mesh = Grid1D(10, 10.0)
    >>> print(mesh)
    """

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
        if (len(args)==6):
            dims, cell_size, cell_location, face_location, corners, edges\
                = args
        else:
            dims, cell_size, cell_location, face_location, corners, edges\
                = self._mesh_1d_param(*args)
        super().__init__(dims, cell_size, cell_location,
                         face_location, corners, edges)

    def _mesh_1d_param(self, *args, coordlabels={'x':'_x'}):
        if len(args) == 1:
            # Use face locations
            facelocationX = args[0]
            Nx = facelocationX.size-1
            cell_size_x = np.hstack([facelocationX[1]-facelocationX[0],
                                     facelocationX[1:]-facelocationX[0:-1],
                                     facelocationX[-1]-facelocationX[-2]])
            cell_size = CellSize(cell_size_x, np.array([0.0]), np.array([0.0]),
                                 coordlabels)
            cell_location = CellLocation(
                0.5*(facelocationX[1:]+facelocationX[0:-1]), 
                np.array([0.0]), 
                np.array([0.0]),
                coordlabels)
            face_location = FaceLocation(
                facelocationX, np.array([0.0]), np.array([0.0]),
                coordlabels)
        elif len(args) == 2:
            # Use number of cells and domain length
            Nx = args[0]
            Width = args[1]
            dx = Width/Nx
            cell_size = CellSize(
                dx*np.ones(Nx+2), np.array([0.0]), np.array([0.0]),
                coordlabels)
            cell_location = CellLocation(
                int_range(1, Nx)*dx-dx/2,
                np.array([0.0]),
                np.array([0.0]),
                coordlabels)
            face_location = FaceLocation(
                int_range(0, Nx)*dx,
                np.array([0.0]),
                np.array([0.0]),
                coordlabels)
        dims = np.array([Nx], dtype=int)
        cellsize = cell_size
        cellcenters = cell_location
        facecenters = face_location
        corners = np.array([1], dtype=int)
        edges = np.array([1], dtype=int)
        return dims, cellsize, cellcenters, facecenters, corners, edges

    def __repr__(self):
        print(f"1D Cartesian mesh with {self.dims[0]} cells")
        return ""

    def cell_numbers(self):
        Nx = self.dims[0]
        return int_range(0, Nx+1)



class CylindricalGrid1D(Grid1D):
    """Mesh based on a 1D cylindrical grid (r)
    =======================================
    
    This class can be instantiated in different ways: from a list of cell face
    locations or from the number of cells and domain length.
    
    Instantiation Options:
    ----------------------
    - CylindricalGrid1D(Nr, Lr)
    - CylindricalGrid1D(face_locationsR)
    
    
    Parameters
    ----------
    CylindricalGrid1D(Nr, Lr)
        Nr : int
            Number of cells in the r direction.
        Lr : float
            Length of the domain in the r direction.
    
    CylindricalGrid1D(face_locationsR)
        face_locationsR : ndarray
            Locations of the cell faces in the r direction.
    
    Examples
    --------
    >>> import numpy as np
    >>> from pyfvtool import CylindricalGrid1D
    >>> mesh = CylindricalGrid1D(10, 10.0)
    >>> print(mesh)
    """

    @overload
    def __init__(self, Nr: int, Lr: float):
        ...
    
    @overload
    def __init__(self, face_locationsR: np.ndarray):
        ...
            
    @overload
    def __init__(self, dims, cellsize,
                       cellcenters, facecenters, corners, edges):
        ...

    def __init__(self, *args):
        if (len(args)==6):
            dims, cell_size, cell_location, face_location, corners, edges\
                = args
        else:
            dims, cell_size, cell_location, face_location, corners, edges\
                = self._mesh_1d_param(*args, coordlabels={'r':'_x'})
        super().__init__(dims, cell_size, cell_location,
                         face_location, corners, edges)

    def __repr__(self):
        print(f"1D Cylindrical (radial) mesh with Nr={self.dims[0]} cells")
        return ""



class SphericalGrid1D(Grid1D):
    """Mesh based on a 1D spherical grid (r)
    =====================================
    
    This class can be instantiated in different ways: from a list of cell face
    locations or from the number of cells and domain length.
    
    Instantiation Options:
    ----------------------
    - SphericalGrid1D(Nr, Lr)
    - SphericalGrid1D(face_locationsR)
    
    
    Parameters
    ----------
    SphericalGrid1D(Nr, Lr)
        Nr : int
            Number of cells in the r direction.
        Lr : float
            Length of the domain in the r direction.
    
    SphericalGrid1D(face_locationsR)
        face_locationsR : ndarray
            Locations of the cell faces in the r direction.
    
    Examples
    --------
    >>> import numpy as np
    >>> from pyfvtool import SphericalGrid1D
    >>> mesh = SphericalGrid1D(10, 10.0)
    >>> print(mesh)
    """

    @overload
    def __init__(self, Nr: int, Lr: float):
        ...
    
    @overload
    def __init__(self, face_locationsR: np.ndarray):
        ...
    
    @overload
    def __init__(self, dims, cellsize,
                 cellcenters, facecenters, corners, edges):
        ...

    def __init__(self, *args):
        if (len(args)==6):
            dims, cell_size, cell_location, face_location, corners, edges\
                = args
        else:
            dims, cell_size, cell_location, face_location, corners, edges\
                = self._mesh_1d_param(*args, coordlabels={'r':'_x'})
        super().__init__(dims, cell_size, cell_location,
                         face_location, corners, edges)

    def __repr__(self):
        return f"1D Spherical mesh with Nr={self.dims[0]} cells"



#%% 
#   2D Grids

class Grid2D(MeshStructure):
    """Mesh based on a 2D Cartesian grid (x, y)
    ========================================
    
    This class can be instantiated in different ways: from a list of cell face
    locations or from the number of cells and domain length.
    
    Instantiation Options:
    ----------------------
    - Grid2D(Nx, Ny, Lx, Ly)
    - Grid2D(face_locationsX, face_locationsY)
    
    
    Parameters
    ----------
    Grid2D(Nx, Ny, Lx, Ly)
        Nx : int
            Number of cells in the x direction.
        Ny : int
            Number of cells in the y direction.
        Lx : float
            Length of the domain in the x direction.
        Ly : float
            Length of the domain in the y direction.
    
    Grid2D(face_locationsX, face_locationsY)
        face_locationsX : ndarray
            Locations of the cell faces in the x direction.
        face_locationsY : ndarray
            Locations of the cell faces in the y direction.
    
    
    Examples
    --------
    >>> import numpy as np
    >>> from pyfvtool import Grid2D
    >>> mesh = Grid2D(10, 10, 10.0, 10.0)
    >>> print(mesh)
    """


    @overload
    def __init__(self, Nx: int, Ny: int, Lx: float, Ly: float):
        ...
    
    
    @overload
    def __init__(self, face_locationsX: np.ndarray,
                       face_locationsY: np.ndarray):
        ...

    @overload
    def __init__(self, dims, cellsize,
                       cellcenters, facecenters, corners, edges):
        ...

    def __init__(self, *args):

        if (len(args)==6):
            dims, cell_size, cell_location, face_location, corners, edges\
                = args
        else:
            dims, cell_size, cell_location, face_location, corners, edges\
                = self._mesh_2d_param(*args)
        super().__init__(dims, cell_size, cell_location,
                         face_location, corners, edges)

    def _mesh_2d_param(self, *args, coordlabels={'x':'_x',
                                                 'y':'_y'}):
        if len(args) == 2:
            # Use face locations
            facelocationX = args[0]
            facelocationY = args[1]
            Nx = facelocationX.size-1
            Ny = facelocationY.size-1
            cell_size = CellSize(self._facelocation_to_cellsize(facelocationX),
                                 self._facelocation_to_cellsize(facelocationY),
                                 np.array([0.0]),
                                 coordlabels)
            cell_location = CellLocation(
                0.5*(facelocationX[1:]+facelocationX[0:-1]),
                0.5*(facelocationY[1:]+facelocationY[0:-1]),
                np.array([0.0]),
                coordlabels)
            face_location = FaceLocation(
                facelocationX,
                facelocationY,
                np.array([0.0]),
                coordlabels)
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
                np.array([0.0]),
                coordlabels)
            cell_location = CellLocation(
                int_range(1, Nx)*dx-dx/2,
                int_range(1, Ny)*dy-dy/2,
                np.array([0.0]),
                coordlabels)
            face_location = FaceLocation(
                int_range(0, Nx)*dx,
                int_range(0, Ny)*dy,
                np.array([0.0]),
                coordlabels)
    
        dims = np.array([Nx, Ny], dtype=int)
        cellsize = cell_size
        cellcenters = cell_location
        facecenters = face_location
        G = int_range(1, (Nx+2)*(Ny+2))-1
        corners = G.reshape(Nx+2, Ny+2)[[0, -1, 0, -1], [0, 0, -1, -1]]
        edges = np.array([1], dtype=int)
        return dims, cellsize, cellcenters, facecenters, corners, edges

    def __repr__(self):
        return f"2D Cartesian mesh with {self.dims[0]}x{self.dims[1]} cells"
    
    def cell_numbers(self):
        Nx, Ny = self.dims
        G = int_range(0, (Nx+2)*(Ny+2)-1)
        return G.reshape(Nx+2, Ny+2)



class CylindricalGrid2D(Grid2D):
    """Mesh based on a 2D cylindrical grid (r, z)
    ==========================================
    
    This class can be instantiated in different ways: from a list of cell face
    locations or from the number of cells and domain length.
    
    Instantiation Options:
    ----------------------
    - CylindricalGrid2D(Nr, Nz, Lr, Lz)
    - CylindricalGrid2D(face_locationsR, face_locationsZ)
    
    Parameters
    ----------
    CylindricalGrid2D(Nr, Nz, Lr, Lz)
        Nr : int
            Number of cells in the r direction.
        Nz : int
            Number of cells in the z direction.
        Lr : float
            Length of the domain in the r direction.
        Lz : float
            Length of the domain in the z direction.
    
    CylindricalGrid2D(face_locationsR, face_locationsZ)
        face_locationsR : ndarray
            Locations of the cell faces in the r direction.
        face_locationsZ : ndarray
            Locations of the cell faces in the z direction.
    
    Returns
    -------
    CylindricalGrid2D
        A 2D cylindrical mesh object.
    
    Examples
    --------
    >>> import numpy as np
    >>> from pyfvtool import CylindricalGrid2D
    >>> mesh = CylindricalGrid2D(10, 10, 10.0, 10.0)
    >>> print(mesh)
    """

    @overload
    def __init__(self, Nr: int, Nz: int,
                       Lr: float, Lz: float):
        ...
        
    @overload
    def __init__(self, face_locationsR: np.ndarray,
                       face_locationsZ: np.ndarray):
        ...

    @overload
    def __init__(self, dims, cellsize,
                 cellcenters, facecenters, corners, edges):
        ...

    def __init__(self, *args):

        if (len(args)==6):
            dims, cell_size, cell_location, face_location, corners, edges\
                = args
        else:
            dims, cell_size, cell_location, face_location, corners, edges\
                = self._mesh_2d_param(*args, coordlabels={'r':'_x',
                                                          'z':'_y'})
        super().__init__(dims, cell_size, cell_location,
                         face_location, corners, edges)

    def __repr__(self):
        print(
            f"2D Cylindrical mesh with Nr={self.dims[0]}xNz={self.dims[1]} cells")
        return ""



class PolarGrid2D(Grid2D):
    """Mesh based on a 2D polar grid (r, theta)
    ========================================
    
    This class can be instantiated in different ways: from a list of cell face
    locations or from the number of cells and domain length.
    
    Instantiation Options:
    ----------------------
    - PolarGrid2D(Nr, Ntheta, Lr, Ltheta)
    - PolarGrid2D(face_locationsR, face_locationsTheta)
    
    Parameters:
    -----------
    PolarGrid2D(Nr, Ntheta, Lr, Ltheta):
        Nr : int
            Number of cells in the r direction.
        Ntheta : int
            Number of cells in the theta direction.
        Lr : float
            Length of the domain in the r direction.
        Ltheta : float
            Length of the domain in the theta direction.
    
    PolarGrid2D(face_locationsR, face_locationsTheta):
        face_locationsR : ndarray
            Locations of the cell faces in the r direction.
        face_locationsTheta : ndarray
            Locations of the cell faces in the theta direction.
    
    Examples:
    ---------
    >>> import numpy as np
    >>> from pyfvtool import PolarGrid2D
    >>> mesh = PolarGrid2D(10, 10, 10.0, 10.0)
    >>> print(mesh)
    """
    @overload
    def __init__(self, Nr: int, Ntheta: int, Lr: float, Ltheta: float):
        ...
    
    
    @overload
    def __init__(self,  face_locationsR: np.ndarray,
                        face_locationsTheta: np.ndarray):
        ...

    @overload
    def __init__(self, dims, cellsize,
                       cellcenters, facecenters, corners, edges):
        ...


    def __init__(self, *args):
        if (len(args)==6):
            dims, cell_size, cell_location,\
                  face_location, corners, edges= args
        else:
            if len(args) == 2:
                theta_max = args[1][-1]
            else:
                theta_max = args[3]
            if (theta_max > 2*np.pi):
                warn("Recreate the mesh with an upper bound of 2*pi for \\theta or there will be unknown consequences!")
            dims, cell_size, cell_location, face_location, corners, edges\
                = self._mesh_2d_param(*args, coordlabels={'r'    :'_x',
                                                          'theta':'_y'})
        super().__init__(dims, cell_size, cell_location,
                         face_location, corners, edges)

    def __repr__(self):
        return f"2D Polar mesh with N_r={self.dims[0]}xN_theta={self.dims[1]} cells"



#%%
#   3D Grids


class Grid3D(MeshStructure):
    """Mesh based on a 3D Cartesian grid (x, y, z)
    ===========================================
        
    This class can be instantiated in different ways: from a list of cell face
    locations or from the number of cells and domain length. There are multiple
    overloaded '__init__' methods available to provide flexibility in
    instantiation.
    
    
    Instantiation Options:
    ----------------------
    - Grid3D(Nx, Ny, Nz, Lx, Ly, Lz)
    - Grid3D(face_locationsX, face_locationsY, face_locationsZ)
    
    
    Parameters
    ----------
    Grid3D(Nx, Ny, Nz, Lx, Ly, Lz)
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
            
    Grid3D(face_locationsX, face_locationsY, face_locationsZ)
        face_locationsX : ndarray
            Locations of the cell faces in the x direction.
        face_locationsY : ndarray
            Locations of the cell faces in the y direction.
        face_locationsZ : ndarray
            Locations of the cell faces in the z direction.
        
        
    Examples
    --------
    >>> import numpy as np
    >>> from pyfvtool import Grid3D
    >>> mesh = Grid3D(10, 10, 10, 10.0, 10.0, 10.0)
    >>> print(mesh)    
        
    """
    @overload
    def __init__(self, Nx: int, Ny: int, Nz: int,
                       Lx: float, Ly: float, Lz: float):
        ...
        
    @overload
    def _init__(self, face_locationsX: np.ndarray,
                      face_locationsY: np.ndarray,
                      face_locationsZ: np.ndarray):
        ...

    @overload
    def __init__(self, dims, cellsize,
                       cellcenters, facecenters, corners, edges):
        ...

    def __init__(self, *args):
        direct_init = False # Flag to indicate if this is a 'direct' __init__
                            # not requiring any parsing of arguments.
                            # These 'direct' instantiantions are used
                            # internally.
        if len(args)==6:
            # Resolve ambiguous @overload situation for 3D meshes
            # not very elegant, but it works
            if isinstance(args[0], np.ndarray):    
                direct_init = True
        if direct_init:
            dims, cell_size, cell_location, face_location, corners, edges\
                = args
        else:
            dims, cell_size, cell_location, face_location, corners, edges\
                = self._mesh_3d_param(*args)
        super().__init__(dims, cell_size, cell_location,
                         face_location, corners, edges)


    def _mesh_3d_param(self, *args, coordlabels={'x':'_x',
                                                 'y':'_y',
                                                 'z':'_z'}):
        if len(args) == 3:
            # Use face locations
            facelocationX = args[0]
            facelocationY = args[1]
            facelocationZ = args[2]
            Nx = facelocationX.size-1
            Ny = facelocationY.size-1
            Nz = facelocationZ.size-1
            cell_size = CellSize(self._facelocation_to_cellsize(facelocationX),
                                 self._facelocation_to_cellsize(facelocationY),
                                 self._facelocation_to_cellsize(facelocationZ),
                                 coordlabels)
            cell_location = CellLocation(
                0.5*(facelocationX[1:]+facelocationX[0:-1]),
                0.5*(facelocationY[1:]+facelocationY[0:-1]),
                0.5*(facelocationZ[1:]+facelocationZ[0:-1]),
                coordlabels)
            face_location = FaceLocation(
                facelocationX,
                facelocationY,
                facelocationZ,
                coordlabels)
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
                dz*np.ones(Nz+2),
                coordlabels)
            cell_location = CellLocation(
                int_range(1, Nx)*dx-dx/2,
                int_range(1, Ny)*dy-dy/2,
                int_range(1, Nz)*dz-dz/2,
                coordlabels)
            face_location = FaceLocation(
                int_range(0, Nx)*dx,
                int_range(0, Ny)*dy,
                int_range(0, Nz)*dz,
                coordlabels)
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

    def cell_numbers(self):
        Nx, Ny, Nz = self.dims
        G = int_range(0, (Nx+2)*(Ny+2)*(Nz+2)-1)
        return G.reshape(Nx+2, Ny+2, Nz+2)

    def __repr__(self):
        return  f"3D Cartesian mesh with "\
            f"Nx={self.dims[0]}xNy={self.dims[1]}xNz={self.dims[1]} cells"




class CylindricalGrid3D(Grid3D):
    """Mesh based on a 3D cylindrical grid (r, theta, z)
    =================================================
        
    This class can be instantiated in different ways: from a list of cell face
    locations or from the number of cells and domain length.
    
    
    Instantiation Options:
    ----------------------
    - CylindricalGrid3D(Nr, Ntheta, Nz, Lr, Ltheta, Lz)
    - CylindricalGrid3D(face_locationsR, face_locationsTheta, face_locationsZ)
        
    Parameters
    ----------
    CylindricalGrid3D(Nr, Ntheta, Nz, Lr, Ltheta, Lz)
        Nr : int
            Number of cells in the r direction.
        Ntheta : int
            Number of cells in the theta direction.
        Nz : int
            Number of cells in the z direction.
        Lr : float
            Length of the domain in the r direction.
        Ltheta : float
            Length of the domain in the theta direction.
        Lz : float
            Length of the domain in the z direction.
            
    
    CylindricalGrid3D(face_locationsR, face_locationsTheta, face_locationsZ)
        face_locationsR : ndarray
            Locations of the cell faces in the r direction.
        face_locationsTheta : ndarray
            Locations of the cell faces in the theta direction.
        face_locationsZ : ndarray
            Locations of the cell faces in the z direction.
        
        
    Examples
    --------
    >>> import numpy as np
    >>> from pyfvtool import CylindricalGrid3D
    >>> mesh = CylindricalGrid3D(10, 10, 10, 10.0, 10.0, 10.0)
    >>> print(mesh)
    """

    @overload
    def __init__(self, Nr: int, Ntheta: int, Nz: int,
                       Lr: float, Ltheta: float, Lz: float):
        ...
    
    @overload
    def __init__(self, face_locationsR: np.ndarray,
                       face_locationsTheta: np.ndarray,
                       face_locationsZ: np.ndarray):
        ...

    @overload
    def __init__(self, dims, cellsize,
                       cellcenters, facecenters, corners, edges):
        ...


    def __init__(self, *args):
        """

        """
        direct_init = False # Flag to indicate if this is a 'direct' __init__
                            # not requiring any parsing of arguments.
                            # These 'direct' instantiantions are used
                            # internally.
        if len(args)==6:
            # Resolve ambiguous @overload situation for 3D meshes
            # not very elegant, but it works
            if isinstance(args[0], np.ndarray):    
                direct_init = True
                
        if direct_init:
            dims, cell_size, cell_location, face_location, corners, edges\
                = args
        else:
            if len(args) == 3:
                theta_max = args[1][-1]
            else:
                theta_max = args[4]
            if theta_max > 2*np.pi:
                warn("Recreate the mesh with an upper bound of 2*pi for theta or there will be unknown consequences!")

            dims, cell_size, cell_location, face_location, corners, edges\
                = self._mesh_3d_param(*args, coordlabels={'r'    :'_x',
                                                          'theta':'_y',
                                                          'z'    :'_z'})

        super().__init__(dims, cell_size, cell_location,
                         face_location, corners, edges)

    def __repr__(self):
        return f"3D Cylindrical mesh with Nr={self.dims[0]}x"\
            f"N_theta={self.dims[1]}xNz={self.dims[1]} cells"



class SphericalGrid3D(Grid3D):
    """Mesh based on a 3D spherical grid (r, theta, phi)
    =================================================
    
    Create a SphericalGrid3D object from a list of cell face locations or from
    the number of cells and domain length.
    
    
    Instantiation Options:
    ----------------------
    - SphericalGrid3D(Nr, Ntheta, Nphi, Lr, Ltheta, Lphi)
    - SphericalGrid3D(face_locationsR, face_locationsTheta, face_locationsPhi)
        
    Parameters
    ----------
    SphericalGrid3D(Nr, Ntheta, Nphi, Lr, Ltheta, Lphi)
        Nr : int
            Number of cells in the r direction.
        Ntheta : int
            Number of cells in the theta direction.
        Nphi : int
            Number of cells in the phi direction.
        Lr : float
            Length of the domain in the r direction.
        Ltheta : float
            Length of the domain in the theta direction.
        Lphi : float
            Length of the domain in the phi direction.
            
    
    SphericalGrid3D(face_locationsR, face_locationsTheta, face_locationsPhi)
        face_locationsR : ndarray
            Locations of the cell faces in the r direction.
        face_locationsTheta : ndarray
            Locations of the cell faces in the theta direction.
        face_locationsPhi : ndarray
            Locations of the cell faces in the phi direction.
        
        
    Examples
    --------
    >>> import numpy as np
    >>> from pyfvtool import SphericalGrid3D
    >>> mesh = SphericalGrid3D(10, 10, 10, 10.0, 10.0, 10.0)
    >>> print(mesh)
    """

    
    @overload
    def __init__(self, Nr: int, Ntheta: int, Nphi: int,
                       Lr: float, Ltheta: float, Lphi: float):
        ...
    
    @overload
    def __init__(self, face_locationsR: np.ndarray,
                       face_locationsTheta: np.ndarray,
                       face_locationsPhi: np.ndarray):
        ...

    @overload
    def __init__(self, dims, cellsize,
                       cellcenters, facecenters, corners, edges):
        ...


    def __init__(self, *args):
        direct_init = False # Flag to indicate if this is a 'direct' __init__
                            # not requiring any parsing of arguments.
                            # These 'direct' instantiations are used
                            # internally.
        if len(args)==6:
            # Resolve ambiguous @overload situation for 3D meshes
            # not very elegant, but it works
            if isinstance(args[0], np.ndarray):    
                direct_init = True
                
        if direct_init:
            dims, cell_size, cell_location, face_location, corners, edges\
                = args
        else:
            if len(args) == 3:
                theta_max = args[1][-1]
                phi_max = args[2][-1]
            elif len(args) == 6:
                theta_max = args[4]
                phi_max = args[5]
            
            if theta_max > np.pi:
                warn("Recreate the mesh with an upper bound of pi for \\theta"\
                    " or there will be unknown consequences!")
            if phi_max > 2*np.pi:
                warn("Recreate the mesh with an upper bound of 2*pi for \\phi"\
                    " or there will be unknown consequences! Do not forget to use a periodic boundary condition for \\phi!")
            dims, cell_size, cell_location, face_location, corners, edges\
                = self._mesh_3d_param(*args, coordlabels={'r'    :'_x',
                                                        'theta':'_y',
                                                        'phi'  :'_z'})

        super().__init__(dims, cell_size, cell_location,
                         face_location, corners, edges)

    def __repr__(self):
        return f"3D Shperical mesh with Nr={self.dims[0]}x"\
            "N_theta={self.dims[1]}xN_phi={self.dims[1]} cells"
