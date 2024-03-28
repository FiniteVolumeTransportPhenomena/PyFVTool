# Mesh generation

import numpy as np
from warnings import warn
from typing import overload
from .utilities import int_range


#%%
#   General data structures for describing meshes

class CellSize:
    def __init__(self, _x: np.ndarray, _y: np.ndarray, _z: np.ndarray):
        self.x = _x
        self.y = _y
        self.z = _z

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
    def __init__(self, _x: np.ndarray, _y: np.ndarray, _z: np.ndarray):
        self.x = _x
        self.y = _y
        self.z = _z

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
    def __init__(self, _x: np.ndarray, _y: np.ndarray, _z: np.ndarray):
        self.x = _x
        self.y = _y
        self.z = _z

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

    def _facelocation_to_cellsize(self, facelocation):
        return np.hstack([facelocation[1]-facelocation[0],
                          facelocation[1:]-facelocation[0:-1],
                          facelocation[-1]-facelocation[-2]])
    
    def visualize(self):
        pass


    def shift_origin(self, _x=0.0, _y=0.0, _z=0.0):
        self.cellcenters.x += _x
        self.cellcenters.y += _y
        self.cellcenters.z += _z
        self.facecenters.x += _x
        self.facecenters.y += _y
        self.facecenters.z += _z


    def cellVolumes(self):
        """Get the volumes of all finite volume cells in the mesh
        
        Returns
        -------
        np.ndarray
            containing all cell volumes, arranged according to gridcells
            
        TODO: these could perhaps be calculated statically, when initializing
        the mesh.
        
        """
        if (type(self) is Grid1D):
            c = self.cellsize.x[1:-1]
        elif (type(self) is CylindricalGrid1D):
            c = 2.0*np.pi*self.cellsize.x[1:-1]*self.cellcenters.x
        elif (type(self) is Grid2D):
            c = self.cellsize.x[1:-1][:, np.newaxis]\
                *self.cellsize.y[1:-1][np.newaxis, :]
        elif (type(self) is CylindricalGrid2D):
            c = 2.0*np.pi*self.cellcenters.x[:, np.newaxis]\
                *self.cellsize.x[1:-1][:, np.newaxis]\
                *self.cellsize.y[1:-1][np.newaxis, :]
        elif (type(self) is PolarGrid2D):
            c = self.cellcenters.x\
                *self.cellsize.x[1:-1][:, np.newaxis]\
                *self.cellsize.y[1:-1][np.newaxis, :]
        elif (type(self) is Grid3D):
            c = self.cellsize.x[1:-1][:,np.newaxis,np.newaxis]\
                *self.cellsize.y[1:-1][np.newaxis,:,np.newaxis]\
                *self.cellsize.z[1:-1][np.newaxis,np.newaxis,:]
        elif (type(self) is CylindricalGrid3D):
            c = self.cellcenters.x\
                *self.cellsize.x[1:-1][:,np.newaxis,np.newaxis]\
                *self.cellsize.y[1:-1][np.newaxis,:,np.newaxis]\
                *self.cellsize.z[np.newaxis,np.newaxis,:]
        return c




#%%
#   1D Grids


class Grid1D(MeshStructure):
    """Mesh based on a 1D Cartesian grid (x)
    
    """
    
    @overload
    def __init__(self, Nx: int, Lx: float):
        """
        TODO: docstring (multiple dispatch)
        These @overload docstrings do NOT show up in help(pf.Grid1D). Therefore,
        put all versions in the main __init__ docstring 

        Parameters
        ----------
        Nx : int
            DESCRIPTION.
        Lx : float
            DESCRIPTION.

        Returns
        -------
        None.

        """
        ...
    
    @overload
    def __init__(self, face_locations: np.ndarray):
        """
        TODO: docstring (multiple dispatch)
        These @overload docstrings do NOT show up in help(pf.Grid1D). Therefore,
        put all versions in the main __init__ docstring 


        Parameters
        ----------
        face_locations : np.ndarray
            DESCRIPTION.

        Returns
        -------
        None.

        """
        ...
        
    @overload
    def __init__(self, dims, cellsize,
                 cellcenters, facecenters, corners, edges):
        """
        TODO: docstring (multiple dispatch)
        These @overload docstrings do NOT show up in help(pf.Grid1D). Therefore,
        put all versions in the main __init__ docstring 


        Parameters
        ----------
        dims : TYPE
            DESCRIPTION.
        cellsize : TYPE
            DESCRIPTION.
        cellcenters : TYPE
            DESCRIPTION.
        facecenters : TYPE
            DESCRIPTION.
        corners : TYPE
            DESCRIPTION.
        edges : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
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
                = self._mesh_1d_param(*args)
        super().__init__(dims, cell_size, cell_location,
                         face_location, corners, edges)

    def _mesh_1d_param(self, *args):
        # In the future, when implementing specific coordinate labels (e.g. (r,z) for 2D
        # cylindrical), we may create subclasses for CellSize, CellLocation,
        # FaceLocation, and pass the suitable subclasses as the first three positional
        # arguments to this _mesh_2d_param method, before *args. This will
        # allow this method to apply the suitable subclass handling the
        # coordinate labels. Of course, we should start with renaming the existing
        # 'internal' x,y,z to _x,_y,_z (same for xvalues, yvalues, zvalues)

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

    def __repr__(self):
        print(f"1D Cartesian mesh with {self.dims[0]} cells")
        return ""

    def cell_numbers(self):
        Nx = self.dims[0]
        return int_range(0, Nx+1)



class CylindricalGrid1D(Grid1D):
    """Mesh based on a 1D cylindrical grid (r)
    
    The volume elements are cylindrical shells around a central axis
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
        """Create a CylindricalGrid1D object from a list of cell face locations or from
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
        CylindricalGrid1D
            A 1D cylindrical mesh object.
        
        Examples
        --------
        >>> import numpy as np
        >>> from pyfvtool import CylindricalGrid1D
        >>> mesh = CylindricalGrid1D(10, 10.0)
        >>> print(mesh)
        """
        if (len(args)==6):
            dims, cell_size, cell_location, face_location, corners, edges\
                = args
        else:
            dims, cell_size, cell_location, face_location, corners, edges\
                = self._mesh_1d_param(*args)
        super().__init__(dims, cell_size, cell_location,
                         face_location, corners, edges)

    def __repr__(self):
        print(f"1D Cylindrical (radial) mesh with Nr={self.dims[0]} cells")
        return ""



class SphericalGrid1D(Grid1D):
    """Mesh based on a 1D spherical grid (r)
    
    The volume elements are concentric spherical shells
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
        """
        Create a SphericalGrid1D object from a list of cell face locations or from
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
        SphericalGrid1D
            A 1D spherical mesh object.
    
        Examples
        --------
        >>> import numpy as np
        >>> from pyfvtool import SphericalGrid1D
        >>> mesh = SphericalGrid1D(10, 10.0)
        >>> print(mesh)
    
        Notes
        -----
        The mesh is created in spherical coordinates.
        """
        if (len(args)==6):
            dims, cell_size, cell_location, face_location, corners, edges\
                = args
        else:
            dims, cell_size, cell_location, face_location, corners, edges\
                = self._mesh_1d_param(*args)
        super().__init__(dims, cell_size, cell_location,
                         face_location, corners, edges)

    def __repr__(self):
        print(f"1D Spherical mesh with Nr={self.dims[0]} cells")
        return ""



#%% 
#   2D Grids

class Grid2D(MeshStructure):
    """Mesh based on a 2D Cartesian grid (x, y)
    
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
        """Create a Grid2D object from a list of cell face locations or from
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
        Grid2D
            A 2D mesh object.
    
        Examples
        --------
        >>> import numpy as np
        >>> from pyfvtool import Grid2D
        >>> mesh = Grid2D(10, 10, 10.0, 10.0)
        >>> print(mesh)
        """
        if (len(args)==6):
            dims, cell_size, cell_location, face_location, corners, edges\
                = args
        else:
            dims, cell_size, cell_location, face_location, corners, edges\
                = self._mesh_2d_param(*args)
        super().__init__(dims, cell_size, cell_location,
                         face_location, corners, edges)

    def _mesh_2d_param(self, *args):
        # In the future, when implementing specific coordinate labels (e.g. (r,z) for 2D
        # cylindrical), we may create subclasses for CellSize, CellLocation,
        # FaceLocation, and pass the suitable subclasses as the first three positional
        # arguments to this _mesh_2d_param method, before *args. This will
        # allow this method to apply the suitable subclass handling the
        # coordinate labels. Of course, we should start with renaming the existing
        # 'internal' x,y,z to _x,_y,_z (same for xvalues, yvalues, zvalues)

        if len(args) == 2:
            # Use face locations
            facelocationX = args[0]
            facelocationY = args[1]
            Nx = facelocationX.size-1
            Ny = facelocationY.size-1
            cell_size = CellSize(self._facelocation_to_cellsize(facelocationX),
                                 self._facelocation_to_cellsize(facelocationY),
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

    def __repr__(self):
        print(f"2D Cartesian mesh with {self.dims[0]}x{self.dims[1]} cells")
        return ""
    
    def cell_numbers(self):
        Nx, Ny = self.dims
        G = int_range(0, (Nx+2)*(Ny+2)-1)
        return G.reshape(Nx+2, Ny+2)


class CylindricalGrid2D(Grid2D):
    """Mesh based on a 2D cylindrical grid (r, z)

    """
    @overload
    def __init__(self, Nx: int, Ny: int,
                 Lx: float, Ly: float):
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
        """Create a CylindricalGrid2D object from a list of cell face locations or from
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
        CylindricalGrid2D
            A 2D cylindrical mesh object.
            
        Examples
        --------
        >>> import numpy as np
        >>> from pyfvtool import CylindricalGrid2D
        >>> mesh = CylindricalGrid2D(10, 10, 10.0, 10.0)
        >>> print(mesh)
        """
        if (len(args)==6):
            dims, cell_size, cell_location, face_location, corners, edges\
                = args
        else:
            dims, cell_size, cell_location, face_location, corners, edges\
                = self._mesh_2d_param(*args)
        super().__init__(dims, cell_size, cell_location,
                         face_location, corners, edges)


    def __repr__(self):
        print(
            f"2D Cylindrical mesh with Nr={self.dims[0]}xNz={self.dims[1]} cells")
        return ""



class PolarGrid2D(Grid2D):
    """Mesh based on a 2D polar grid (r, theta)

    """
    @overload
    def __init__(self, Nx: int, Ny: int, Lx: float, Ly: float):
        ...
    
    
    @overload
    def __init__(self,  face_locationsX: np.ndarray,
                        face_locationsY: np.ndarray):
        ...

    @overload
    def __init__(self, dims, cellsize,
                 cellcenters, facecenters, corners, edges):
        ...


    def __init__(self, *args):
        """
        Create a PolarGrid2D object from a list of cell face locations or from
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
        PolarGrid2D
            A 2D radial mesh object.
        
        Examples
        --------
        >>> import numpy as np
        >>> from pyfvtool import PolarGrid2D
        >>> mesh = PolarGrid2D(10, 10, 10.0, 10.0)
        >>> print(mesh)
    
        Notes
        -----
        The mesh is created in radial (cylindrical) coordinates.
        """
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
                = self._mesh_2d_param(*args)
        super().__init__(dims, cell_size, cell_location,
                         face_location, corners, edges)

    def __repr__(self):
        print(
            f"2D Polar mesh with Nr={self.dims[0]}xN_theta={self.dims[1]} cells")
        return ""



#%%
#   3D Grids


class Grid3D(MeshStructure):
    """Mesh based on a 3D Cartesian grid (x, y, z)"""
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
        """
        Create a Grid3D object from a list of cell face locations or from
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
        Grid3D
        A 3D mesh object.
            
        Examples
        --------
        >>> import numpy as np
        >>> from pyfvtool import Grid3D
        >>> mesh = Grid3D(10, 10, 10, 10.0, 10.0, 10.0)
        >>> print(mesh)
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
            dims, cell_size, cell_location, face_location, corners, edges\
                = self._mesh_3d_param(*args)
        super().__init__(dims, cell_size, cell_location,
                         face_location, corners, edges)


    def _mesh_3d_param(self, *args):
        # In the future, when implementing specific coordinate labels (e.g. (r,z) for 2D
        # cylindrical), we may create subclasses for CellSize, CellLocation,
        # FaceLocation, and pass the suitable subclasses as the first three positional
        # arguments to this _mesh_3d_param method, before *args. This will
        # allow this method to apply the suitable subclass handling the
        # coordinate labels. Of course, we should start with renaming the existing
        # 'internal' x,y,z to _x,_y,_z (same for xvalues, yvalues, zvalues)
    
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
                                 self._facelocation_to_cellsize(facelocationZ))
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

    def cell_numbers(self):
        Nx, Ny, Nz = self.dims
        G = int_range(0, (Nx+2)*(Ny+2)*(Nz+2)-1)
        return G.reshape(Nx+2, Ny+2, Nz+2)

    def __repr__(self):
        print(
            f"3D Cartesian mesh with Nx={self.dims[0]}xNy={self.dims[1]}xNz={self.dims[1]} cells")
        return ""


class CylindricalGrid3D(Grid3D):
    """Mesh based on a 3D cylindrical grid (r, theta, z)"""
    @overload
    def __init__(self, Nx: int, Ny: int, Nz: int,
                       Lx: float, Ly: float, Lz: float):
        ...
    
    @overload
    def __init__(self, face_locationsX: np.ndarray,
                       face_locationsY: np.ndarray,
                       face_locationsZ: np.ndarray):
        ...

    @overload
    def __init__(self, dims, cellsize,
                       cellcenters, facecenters, corners, edges):
        ...


    def __init__(self, *args):
        """
        Create a CylindricalGrid3D object from a list of cell face locations or from
        number of cells and domain length.

        TO DO: docstring (and coordinate labels) -> r, theta, z
    
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
        CylindricalGrid3D
            A 3D cylindrical mesh object.
    
        Examples
        --------
        >>> import numpy as np
        >>> from pyfvtool import CylindricalGrid3D
        >>> mesh = CylindricalGrid3D(10, 10, 10, 10.0, 10.0, 10.0)
        >>> print(mesh)
    
        Notes
        -----
        The mesh is created in cylindrical coordinates.
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
                = self._mesh_3d_param(*args)

        super().__init__(dims, cell_size, cell_location,
                         face_location, corners, edges)

    def __repr__(self):
        print(
            f"3D Cylindrical mesh with Nr={self.dims[0]}xN_theta={self.dims[1]}xNz={self.dims[1]} cells")
        return ""


class SphericalGrid3D(Grid3D):
    """Mesh based on a 3D spherical grid (r, theta, phi)"""
    
    @overload
    def __init__(self, Nx: int, Ny: int, Nz: int,
                       Lx: float, Ly: float, Lz: float):
        ...
    
    @overload
    def __init__(self, face_locationsX: np.ndarray,
                       face_locationsY: np.ndarray,
                       face_locationsZ: np.ndarray):
        ...

    @overload
    def __init__(self, dims, cellsize,
                       cellcenters, facecenters, corners, edges):
        ...


    def __init__(self, *args):
        """
        Create a SphericalGrid3D object from a list of cell face locations or from
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
        SphericalGrid3D
            A 3D spherical mesh object.
        
        Examples
        --------
        >>> import numpy as np
        >>> from pyfvtool import SphericalGrid3D
        >>> mesh = SphericalGrid3D(10, 10, 10, 10.0, 10.0, 10.0)
        >>> print(mesh)
    
        Notes
        -----
        The mesh is created in spherical coordinates.
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
            if args[4] > 2*np.pi:
                warn("Recreate the mesh with an upper bound of 2*pi for \\theta"\
                     " or there will be unknown consequences!")
            if args[5] > 2*np.pi:
                warn("Recreate the mesh with an upper bound of 2*pi for \\phi"\
                     " or there will be unknown consequences!")
            dims, cell_size, cell_location, face_location, corners, edges\
                = self._mesh_3d_param(*args)

        super().__init__(dims, cell_size, cell_location,
                         face_location, corners, edges)

    def __repr__(self):
        print(
            f"3D Shperical mesh with Nr={self.dims[0]}xN_theta={self.dims[1]}xN_phi={self.dims[1]} cells")
        return ""

