import numpy as np
from typing import overload

from .mesh import MeshStructure
from .mesh import Mesh1D, Mesh2D, Mesh3D
from .mesh import MeshCylindrical1D, MeshCylindrical2D
from .mesh import MeshRadial2D, MeshCylindrical3D



class FaceVariable:
    """
    Face variable class
    

    Create a FaceVariable for the given mesh with the given value.
    Examples:
    >>> import pyfvtool as pf
    >>> m = pf.createMesh1D(10, 1.0)
    >>> f = pf.FaceVariable(m, 1.0)

    """
    @overload
    def __init__(self, mesh: MeshStructure, faceval : float):
        ...
     
    @overload
    def __init__(self, mesh: MeshStructure, faceval : np.ndarray):
        ...
        
    @overload
    def __init__(self, 
                 mesh: MeshStructure,
                 xvalue: np.ndarray,
                 yvalue: np.ndarray,
                 zvalue: np.ndarray):
        ...
 
    
    def __init__(self,
                 mesh: MeshStructure,
                 *args):
        if len(args)==3:             
            xvalue = args[0]
            yvalue = args[1]
            zvalue = args[2]
        elif len(args)==1:
            faceval = args[0]
            if issubclass(type(mesh), Mesh1D):
                Nx = mesh.dims
                if np.isscalar(faceval):
                    xvalue = faceval*np.ones(Nx+1)
                    yvalue = np.array([])
                    zvalue = np.array([])
                else:
                    xvalue = faceval[0]*np.ones(Nx+1)
                    yvalue = np.array([])
                    zvalue = np.array([])
            elif issubclass(type(mesh), Mesh2D):
                Nx, Ny = mesh.dims
                if np.isscalar(faceval):
                    xvalue = faceval*np.ones((Nx+1, Ny))
                    yvalue = faceval*np.ones((Nx, Ny+1))
                    zvalue = np.array([])
                else:
                    xvalue = faceval[0]*np.ones((Nx+1, Ny))
                    yvalue = faceval[1]*np.ones((Nx, Ny+1))
                    zvalue = np.array([])
            elif issubclass(type(mesh), Mesh3D):
                Nx, Ny, Nz = mesh.dims
                if np.isscalar(faceval):
                    xvalue = faceval*np.ones((Nx+1, Ny, Nz))
                    yvalue = faceval*np.ones((Nx, Ny+1, Nz))
                    zvalue = faceval*np.ones((Nx, Ny, Nz+1))
                else:
                    xvalue = faceval[0]*np.ones((Nx+1, Ny, Nz))
                    yvalue = faceval[1]*np.ones((Nx, Ny+1, Nz))
                    zvalue = faceval[2]*np.ones((Nx, Ny, Nz+1))
        else:
            raise TypeError('Unexpected number of arguments')
            
        self.domain = mesh
        self.xvalue = xvalue
        self.yvalue = yvalue
        self.zvalue = zvalue


    def __add__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain,
                                self.xvalue+other.xvalue,
                                self.yvalue+other.yvalue,
                                self.zvalue+other.zvalue)
        else:
            return FaceVariable(self.domain,
                                self.xvalue+other,
                                self.yvalue+other,
                                self.zvalue+other)

    def __radd__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain,
                                self.xvalue+other.xvalue,
                                self.yvalue+other.yvalue,
                                self.zvalue+other.zvalue)
        else:
            return FaceVariable(self.domain,
                                self.xvalue+other,
                                self.yvalue+other,
                                self.zvalue+other)

    def __rsub__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, other.xvalue-self.xvalue,
                                other.yvalue-self.yvalue,
                                other.zvalue-self.zvalue)
        else:
            return FaceVariable(self.domain,
                                other-self.xvalue,
                                other-self.yvalue,
                                other-self.zvalue)

    def __sub__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, self.xvalue-other.xvalue,
                                self.yvalue-other.yvalue,
                                self.zvalue-other.zvalue)
        else:
            return FaceVariable(self.domain, self.xvalue-other,
                                self.yvalue-other,
                                self.zvalue-other)

    def __mul__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, self.xvalue*other.xvalue,
                                self.yvalue*other.yvalue,
                                self.zvalue*other.zvalue)
        else:
            return FaceVariable(self.domain, self.xvalue*other,
                                self.yvalue*other,
                                self.zvalue*other)

    def __rmul__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, self.xvalue*other.xvalue,
                                self.yvalue*other.yvalue,
                                self.zvalue*other.zvalue)
        else:
            return FaceVariable(self.domain, self.xvalue*other,
                                self.yvalue*other,
                                self.zvalue*other)

    def __truediv__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, self.xvalue/other.xvalue,
                                self.yvalue/other.yvalue,
                                self.zvalue/other.zvalue)
        else:
            return FaceVariable(self.domain, self.xvalue/other,
                                self.yvalue/other,
                                self.zvalue/other)

    def __rtruediv__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, other.xvalue/self.xvalue,
                                other.yvalue/self.yvalue,
                                other.zvalue/self.zvalue)
        else:
            return FaceVariable(self.domain, other/self.xvalue,
                                other/self.yvalue,
                                other/self.zvalue)

    def __neg__(self):
        return FaceVariable(self.domain, -self.xvalue,
                            -self.yvalue,
                            -self.zvalue)

    def __pow__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, self.xvalue**other.xvalue,
                                self.yvalue**other.yvalue,
                                self.zvalue**other.zvalue)
        else:
            return FaceVariable(self.domain, self.xvalue**other,
                                self.yvalue**other,
                                self.zvalue**other)

    def __rpow__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, other.xvalue**self.xvalue,
                                other.yvalue**self.yvalue,
                                other.zvalue**self.zvalue)
        else:
            return FaceVariable(self.domain, other**self.xvalue,
                                other**self.yvalue,
                                other**self.zvalue)

    def __gt__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, self.xvalue > other.xvalue,
                                self.yvalue > other.yvalue,
                                self.zvalue > other.zvalue)
        else:
            return FaceVariable(self.domain, self.xvalue > other,
                                self.yvalue > other,
                                self.zvalue > other)

    def __ge__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, self.xvalue >= other.xvalue,
                                self.yvalue >= other.yvalue,
                                self.zvalue >= other.zvalue)
        else:
            return FaceVariable(self.domain, self.xvalue >= other,
                                self.yvalue >= other,
                                self.zvalue >= other)

    def __lt__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, self.xvalue < other.xvalue,
                                self.yvalue < other.yvalue,
                                self.zvalue < other.zvalue)
        else:
            return FaceVariable(self.domain, self.xvalue < other,
                                self.yvalue < other,
                                self.zvalue < other)

    def __le__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, self.xvalue <= other.xvalue,
                                self.yvalue <= other.yvalue,
                                self.zvalue <= other.zvalue)
        else:
            return FaceVariable(self.domain, self.xvalue <= other,
                                self.yvalue <= other,
                                self.zvalue <= other)

    def __and__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, np.logical_and(self.xvalue, other.xvalue),
                                np.logical_and(self.yvalue, other.yvalue),
                                np.logical_and(self.zvalue, other.zvalue))
        else:
            return FaceVariable(self.domain, np.logical_and(self.xvalue, other),
                                np.logical_and(self.yvalue, other),
                                np.logical_and(self.zvalue, other))

    def __or__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, np.logical_or(self.xvalue, other.xvalue),
                                np.logical_or(self.yvalue, other.yvalue),
                                np.logical_or(self.zvalue, other.zvalue))
        else:
            return FaceVariable(self.domain, np.logical_or(self.xvalue, other),
                                np.logical_or(self.yvalue, other),
                                np.logical_or(self.zvalue, other))

    def __abs__(self):
        return FaceVariable(self.domain, np.abs(self.xvalue),
                            np.abs(self.yvalue),
                            np.abs(self.zvalue))


def createFaceVariable(mesh, faceval):

    if issubclass(type(mesh), Mesh1D):
        Nx = mesh.dims
        if np.isscalar(faceval):
            return FaceVariable(mesh, faceval*np.ones(Nx+1), np.array([]), np.array([]))
        else:
            return FaceVariable(mesh, faceval[0]*np.ones(Nx+1), np.array([]), np.array([]))
    elif issubclass(type(mesh), Mesh2D):
        Nx, Ny = mesh.dims
        if np.isscalar(faceval):
            return FaceVariable(mesh, faceval*np.ones((Nx+1, Ny)),
                            faceval*np.ones((Nx, Ny+1)), np.array([]))
        else:
            return FaceVariable(mesh, faceval[0]*np.ones((Nx+1, Ny)),
                            faceval[1]*np.ones((Nx, Ny+1)), np.array([]))
    elif issubclass(type(mesh), Mesh3D):
        Nx, Ny, Nz = mesh.dims
        if np.isscalar(faceval):
            return FaceVariable(mesh, faceval*np.ones((Nx+1, Ny, Nz)),
                            faceval*np.ones((Nx, Ny+1, Nz)),
                            faceval*np.ones((Nx, Ny, Nz+1)))
        else:
            return FaceVariable(mesh, faceval[0]*np.ones((Nx+1, Ny, Nz)),
                            faceval[1]*np.ones((Nx, Ny+1, Nz)),
                            faceval[2]*np.ones((Nx, Ny, Nz+1)))

        
def faceLocations(m: MeshStructure):
    """
    this function returns the location of the cell faces as face variables. 
    
    It can later be used for the calculation of face variables as a function of location
    
    Incompletely tested
    
    Parameters
    ----------
    m : {MeshStructure object}
        Domain of the problem

    Returns
    -------
    X : {FaceVariable object}
        Edge x-positions        
    Y : {FaceVariable object}
        Edge y-positions        
    Z : {FaceVariable object}
        Edge z-positions        

    See Also
    --------
    cellLocations

    Notes
    -----

    Examples
    --------
    >>>     
    
    """    
    
    N = m.dims
    
    if (type(m) is Mesh1D)\
     or (type(m) is MeshCylindrical1D):
        X = createFaceVariable(m, 0)
        X.xvalue = m.facecenters.x
        return X
        
    elif (type(m) is Mesh2D)\
       or (type(m) is MeshCylindrical2D)\
       or (type(m) is MeshRadial2D):
        X = createFaceVariable(m, 0)
        Y = createFaceVariable(m, 0)
        X.xvalue = np.tile(m.facecenters.x[:, np.newaxis], (1, N[1]))
        X.yvalue = np.tile(m.cellcenters.y[:, np.newaxis].T, (N[0]+1, 1))
        Y.xvalue = np.tile(m.cellcenters.x[:, np.newaxis], (1, N[1]+1))
        Y.yvalue = np.tile(m.facecenters.y[:, np.newaxis].T, (N[0], 1))
        return X, Y
        
    elif (type(m) is Mesh3D)\
       or (type(m) is MeshCylindrical3D):
        X = createFaceVariable(m, 0)
        Y = createFaceVariable(m, 0)
        Z = createFaceVariable(m, 0)
        z = np.zeros((1,1,N[2]))
        z[0, 0, :] = m.cellcenters.z
        
        X.xvalue = np.tile(m.facecenters.x[:, np.newaxis, np.newaxis], (1, N[1], N[2]))
        X.yvalue = np.tile((m.cellcenters.y[:, np.newaxis].T)[:, :, np.newaxis], (N[0]+1, 1, N[2]))
        X.zvalue = np.tile(z, (N[0]+1, N[1], 1))
        
        Y.xvalue = np.tile(m.cellcenters.x[:, np.newaxis, np.newaxis], (1, N[1]+1, N[2]))
        Y.yvalue = np.tile((m.facecenters.y[:, np.newaxis].T)[:, :, np.newaxis], (N[0], 1, N[2]))
        Y.zvalue = np.tile(z, (N[0], N[1]+1, 1))

        z = np.zeros((1,1,N[2]+1))
        z[0, 0, :] = m.cellcenters.z
        Z.xvalue = np.tile(m.cellcenters.x[:, np.newaxis, np.newaxis], (1, N[1], N[2]+1))
        Z.yvalue = np.tile((m.facecenters.y[:, np.newaxis].T)[:, :, np.newaxis], (N[0], 1, N[2]+1))
        Z.zvalue = np.tile(z, (N[0], N[1], 1))
        return X, Y, Z
    raise TypeError('mesh type not implemented')
    return None
        
        
def faceeval(f, *args):
    """
    Evaluate a function f on a FaceVariable.
    Examples:
    >>> import pyfvtool as pf
    >>> m = pf.createMesh1D(10, 1.0)
    >>> f = pf.createFaceVariable(m, 1.0)
    >>> g = pf.faceeval(lambda x: x**2, f)
    """
    if len(args)==1:
        return FaceVariable(args[0].domain,
                            f(args[0].xvalue),
                            f(args[0].yvalue),
                            f(args[0].zvalue))
    elif len(args)==2:
        return FaceVariable(args[0].domain,
                            f(args[0].xvalue, args[1].xvalue),
                            f(args[0].yvalue, args[1].yvalue),
                            f(args[0].zvalue, args[1].zvalue))
    elif len(args)==3:
        return FaceVariable(args[0].domain,
                            f(args[0].xvalue, args[1].xvalue, args[2].xvalue),
                            f(args[0].yvalue, args[1].yvalue, args[2].yvalue),
                            f(args[0].zvalue, args[1].zvalue, args[2].zvalue))
    elif len(args)==4:
        return FaceVariable(args[0].domain,
                            f(args[0].xvalue, args[1].xvalue, args[2].xvalue, args[3].xvalue),
                            f(args[0].yvalue, args[1].yvalue, args[2].yvalue, args[3].yvalue),
                            f(args[0].zvalue, args[1].zvalue, args[2].zvalue, args[3].zvalue))
    elif len(args)==5:
        return FaceVariable(args[0].domain,
                            f(args[0].xvalue, args[1].xvalue, args[2].xvalue, args[3].xvalue, args[4].xvalue),
                            f(args[0].yvalue, args[1].yvalue, args[2].yvalue, args[3].yvalue, args[4].yvalue),
                            f(args[0].zvalue, args[1].zvalue, args[2].zvalue, args[3].zvalue, args[4].zvalue))
    elif len(args)==6:
        return FaceVariable(args[0].domain,
                            f(args[0].xvalue, args[1].xvalue, args[2].xvalue, args[3].xvalue, args[4].xvalue, args[5].xvalue),
                            f(args[0].yvalue, args[1].yvalue, args[2].yvalue, args[3].yvalue, args[4].yvalue, args[5].yvalue),
                            f(args[0].zvalue, args[1].zvalue, args[2].zvalue, args[3].zvalue, args[4].zvalue, args[5].zvalue))
    elif len(args)==7:
        return FaceVariable(args[0].domain,
                            f(args[0].xvalue, args[1].xvalue, args[2].xvalue, args[3].xvalue, args[4].xvalue, args[5].xvalue, args[6].xvalue),
                            f(args[0].yvalue, args[1].yvalue, args[2].yvalue, args[3].yvalue, args[4].yvalue, args[5].yvalue, args[6].yvalue),
                            f(args[0].zvalue, args[1].zvalue, args[2].zvalue, args[3].zvalue, args[4].zvalue, args[5].zvalue, args[6].zvalue))
    elif len(args)==8:
        return FaceVariable(args[0].domain,
                            f(args[0].xvalue, args[1].xvalue, args[2].xvalue, args[3].xvalue, args[4].xvalue, args[5].xvalue, args[6].xvalue, args[7].xvalue),
                            f(args[0].yvalue, args[1].yvalue, args[2].yvalue, args[3].yvalue, args[4].yvalue, args[5].yvalue, args[6].yvalue, args[7].yvalue),
                            f(args[0].zvalue, args[1].zvalue, args[2].zvalue, args[3].zvalue, args[4].zvalue, args[5].zvalue, args[6].zvalue, args[7].zvalue))
