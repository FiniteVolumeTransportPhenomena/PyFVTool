import numpy as np
from typing import overload

from .mesh import MeshStructure
from .mesh import Grid1D, Grid2D, Grid3D
from .mesh import CylindricalGrid1D, CylindricalGrid2D
from .mesh import PolarGrid2D, CylindricalGrid3D



class FaceVariable:
    """
    Face variable class
    

    Create a FaceVariable for the given mesh with the given value.
    Examples:
    >>> import pyfvtool as pf
    >>> m = pf.Grid1D(10, 1.0)
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
                 _xvalue: np.ndarray,
                 _yvalue: np.ndarray,
                 _zvalue: np.ndarray):
        ...
 
    
    def __init__(self,
                 mesh: MeshStructure,
                 *args):
        if len(args)==3:             
            _xvalue = args[0]
            _yvalue = args[1]
            _zvalue = args[2]
        elif len(args)==1:
            faceval = args[0]
            if issubclass(type(mesh), Grid1D):
                Nx = mesh.dims
                if np.isscalar(faceval):
                    _xvalue = faceval*np.ones(Nx+1)
                    _yvalue = np.array([])
                    _zvalue = np.array([])
                else:
                    _xvalue = faceval[0]*np.ones(Nx+1)
                    _yvalue = np.array([])
                    _zvalue = np.array([])
            elif issubclass(type(mesh), Grid2D):
                Nx, Ny = mesh.dims
                if np.isscalar(faceval):
                    _xvalue = faceval*np.ones((Nx+1, Ny))
                    _yvalue = faceval*np.ones((Nx, Ny+1))
                    _zvalue = np.array([])
                else:
                    _xvalue = faceval[0]*np.ones((Nx+1, Ny))
                    _yvalue = faceval[1]*np.ones((Nx, Ny+1))
                    _zvalue = np.array([])
            elif issubclass(type(mesh), Grid3D):
                Nx, Ny, Nz = mesh.dims
                if np.isscalar(faceval):
                    _xvalue = faceval*np.ones((Nx+1, Ny, Nz))
                    _yvalue = faceval*np.ones((Nx, Ny+1, Nz))
                    _zvalue = faceval*np.ones((Nx, Ny, Nz+1))
                else:
                    _xvalue = faceval[0]*np.ones((Nx+1, Ny, Nz))
                    _yvalue = faceval[1]*np.ones((Nx, Ny+1, Nz))
                    _zvalue = faceval[2]*np.ones((Nx, Ny, Nz+1))
        else:
            raise TypeError('Unexpected number of arguments')
            
        self.domain = mesh
        self._xvalue = _xvalue
        self._yvalue = _yvalue
        self._zvalue = _zvalue


    @property
    def xvalue(self):
        # TO DO: only if cartesian grid
        return self._xvalue
    
    @xvalue.setter
    def xvalue(self, value):
        # TO DO: only if cartesian grid
        self._xvalue = value
    
    @property
    def yvalue(self):
        # TO DO: only if cartesian grid
        return self._yvalue
    
    @yvalue.setter
    def yvalue(self, value):
        # TO DO: only if cartesian grid
        self._yvalue = value
        
    @property
    def zvalue(self):
        # TO DO: only if cartesian or cylindrical grid
        # in the case of 2D cylindrical return self._yvalue !!
        return self._zvalue
    
    @zvalue.setter
    def zvalue(self, value):
        # TO DO: only if cartesian or cylindrical grid
        # in the case of 2D cylindrical self._yvalue = value !!
        self._zvalue = value
        
    





    def __add__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain,
                                self._xvalue+other._xvalue,
                                self._yvalue+other._yvalue,
                                self._zvalue+other._zvalue)
        else:
            return FaceVariable(self.domain,
                                self._xvalue+other,
                                self._yvalue+other,
                                self._zvalue+other)

    def __radd__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain,
                                self._xvalue+other._xvalue,
                                self._yvalue+other._yvalue,
                                self._zvalue+other._zvalue)
        else:
            return FaceVariable(self.domain,
                                self._xvalue+other,
                                self._yvalue+other,
                                self._zvalue+other)

    def __rsub__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, other._xvalue-self._xvalue,
                                other._yvalue-self._yvalue,
                                other._zvalue-self._zvalue)
        else:
            return FaceVariable(self.domain,
                                other-self._xvalue,
                                other-self._yvalue,
                                other-self._zvalue)

    def __sub__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, self._xvalue-other._xvalue,
                                self._yvalue-other._yvalue,
                                self._zvalue-other._zvalue)
        else:
            return FaceVariable(self.domain, self._xvalue-other,
                                self._yvalue-other,
                                self._zvalue-other)

    def __mul__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, self._xvalue*other._xvalue,
                                self._yvalue*other._yvalue,
                                self._zvalue*other._zvalue)
        else:
            return FaceVariable(self.domain, self._xvalue*other,
                                self._yvalue*other,
                                self._zvalue*other)

    def __rmul__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, self._xvalue*other._xvalue,
                                self._yvalue*other._yvalue,
                                self._zvalue*other._zvalue)
        else:
            return FaceVariable(self.domain, self._xvalue*other,
                                self._yvalue*other,
                                self._zvalue*other)

    def __truediv__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, self._xvalue/other._xvalue,
                                self._yvalue/other._yvalue,
                                self._zvalue/other._zvalue)
        else:
            return FaceVariable(self.domain, self._xvalue/other,
                                self._yvalue/other,
                                self._zvalue/other)

    def __rtruediv__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, other._xvalue/self._xvalue,
                                other._yvalue/self._yvalue,
                                other._zvalue/self._zvalue)
        else:
            return FaceVariable(self.domain, other/self._xvalue,
                                other/self._yvalue,
                                other/self._zvalue)

    def __neg__(self):
        return FaceVariable(self.domain, -self._xvalue,
                            -self._yvalue,
                            -self._zvalue)

    def __pow__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, self._xvalue**other._xvalue,
                                self._yvalue**other._yvalue,
                                self._zvalue**other._zvalue)
        else:
            return FaceVariable(self.domain, self._xvalue**other,
                                self._yvalue**other,
                                self._zvalue**other)

    def __rpow__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, other._xvalue**self._xvalue,
                                other._yvalue**self._yvalue,
                                other._zvalue**self._zvalue)
        else:
            return FaceVariable(self.domain, other**self._xvalue,
                                other**self._yvalue,
                                other**self._zvalue)

    def __gt__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, self._xvalue > other._xvalue,
                                self._yvalue > other._yvalue,
                                self._zvalue > other._zvalue)
        else:
            return FaceVariable(self.domain, self._xvalue > other,
                                self._yvalue > other,
                                self._zvalue > other)

    def __ge__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, self._xvalue >= other._xvalue,
                                self._yvalue >= other._yvalue,
                                self._zvalue >= other._zvalue)
        else:
            return FaceVariable(self.domain, self._xvalue >= other,
                                self._yvalue >= other,
                                self._zvalue >= other)

    def __lt__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, self._xvalue < other._xvalue,
                                self._yvalue < other._yvalue,
                                self._zvalue < other._zvalue)
        else:
            return FaceVariable(self.domain, self._xvalue < other,
                                self._yvalue < other,
                                self._zvalue < other)

    def __le__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, self._xvalue <= other._xvalue,
                                self._yvalue <= other._yvalue,
                                self._zvalue <= other._zvalue)
        else:
            return FaceVariable(self.domain, self._xvalue <= other,
                                self._yvalue <= other,
                                self._zvalue <= other)

    def __and__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, np.logical_and(self._xvalue, other._xvalue),
                                np.logical_and(self._yvalue, other._yvalue),
                                np.logical_and(self._zvalue, other._zvalue))
        else:
            return FaceVariable(self.domain, np.logical_and(self._xvalue, other),
                                np.logical_and(self._yvalue, other),
                                np.logical_and(self._zvalue, other))

    def __or__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, np.logical_or(self._xvalue, other._xvalue),
                                np.logical_or(self._yvalue, other._yvalue),
                                np.logical_or(self._zvalue, other._zvalue))
        else:
            return FaceVariable(self.domain, np.logical_or(self._xvalue, other),
                                np.logical_or(self._yvalue, other),
                                np.logical_or(self._zvalue, other))

    def __abs__(self):
        return FaceVariable(self.domain, np.abs(self._xvalue),
                            np.abs(self._yvalue),
                            np.abs(self._zvalue))


        
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
    
    if (type(m) is Grid1D)\
     or (type(m) is CylindricalGrid1D):
        X = FaceVariable(m, 0)
        X._xvalue = m.facecenters.x
        return X
        
    elif (type(m) is Grid2D)\
       or (type(m) is CylindricalGrid2D)\
       or (type(m) is PolarGrid2D):
        X = FaceVariable(m, 0)
        Y = FaceVariable(m, 0)
        X._xvalue = np.tile(m.facecenters.x[:, np.newaxis], (1, N[1]))
        X._yvalue = np.tile(m.cellcenters.y[:, np.newaxis].T, (N[0]+1, 1))
        Y._xvalue = np.tile(m.cellcenters.x[:, np.newaxis], (1, N[1]+1))
        Y._yvalue = np.tile(m.facecenters.y[:, np.newaxis].T, (N[0], 1))
        return X, Y
        
    elif (type(m) is Grid3D)\
       or (type(m) is CylindricalGrid3D):
        X = FaceVariable(m, 0)
        Y = FaceVariable(m, 0)
        Z = FaceVariable(m, 0)
        z = np.zeros((1,1,N[2]))
        z[0, 0, :] = m.cellcenters.z
        
        X._xvalue = np.tile(m.facecenters.x[:, np.newaxis, np.newaxis], (1, N[1], N[2]))
        X._yvalue = np.tile((m.cellcenters.y[:, np.newaxis].T)[:, :, np.newaxis], (N[0]+1, 1, N[2]))
        X._zvalue = np.tile(z, (N[0]+1, N[1], 1))
        
        Y._xvalue = np.tile(m.cellcenters.x[:, np.newaxis, np.newaxis], (1, N[1]+1, N[2]))
        Y._yvalue = np.tile((m.facecenters.y[:, np.newaxis].T)[:, :, np.newaxis], (N[0], 1, N[2]))
        Y._zvalue = np.tile(z, (N[0], N[1]+1, 1))

        z = np.zeros((1,1,N[2]+1))
        z[0, 0, :] = m.cellcenters.z
        Z._xvalue = np.tile(m.cellcenters.x[:, np.newaxis, np.newaxis], (1, N[1], N[2]+1))
        Z._yvalue = np.tile((m.facecenters.y[:, np.newaxis].T)[:, :, np.newaxis], (N[0], 1, N[2]+1))
        Z._zvalue = np.tile(z, (N[0], N[1], 1))
        return X, Y, Z
    raise TypeError('mesh type not implemented')
    return None
        
        
def faceeval(f, *args):
    """
    Evaluate a function f on a FaceVariable.
    Examples:
    >>> import pyfvtool as pf
    >>> m = pf.Grid1D(10, 1.0)
    >>> f = pf.FaceVariable(m, 1.0)
    >>> g = pf.faceeval(lambda x: x**2, f)
    """
    if len(args)==1:
        return FaceVariable(args[0].domain,
                            f(args[0]._xvalue),
                            f(args[0]._yvalue),
                            f(args[0]._zvalue))
    elif len(args)==2:
        return FaceVariable(args[0].domain,
                            f(args[0]._xvalue, args[1]._xvalue),
                            f(args[0]._yvalue, args[1]._yvalue),
                            f(args[0]._zvalue, args[1]._zvalue))
    elif len(args)==3:
        return FaceVariable(args[0].domain,
                            f(args[0]._xvalue, args[1]._xvalue, args[2]._xvalue),
                            f(args[0]._yvalue, args[1]._yvalue, args[2]._yvalue),
                            f(args[0]._zvalue, args[1]._zvalue, args[2]._zvalue))
    elif len(args)==4:
        return FaceVariable(args[0].domain,
                            f(args[0]._xvalue, args[1]._xvalue, args[2]._xvalue, args[3]._xvalue),
                            f(args[0]._yvalue, args[1]._yvalue, args[2]._yvalue, args[3]._yvalue),
                            f(args[0]._zvalue, args[1]._zvalue, args[2]._zvalue, args[3]._zvalue))
    elif len(args)==5:
        return FaceVariable(args[0].domain,
                            f(args[0]._xvalue, args[1]._xvalue, args[2]._xvalue, args[3]._xvalue, args[4]._xvalue),
                            f(args[0]._yvalue, args[1]._yvalue, args[2]._yvalue, args[3]._yvalue, args[4]._yvalue),
                            f(args[0]._zvalue, args[1]._zvalue, args[2]._zvalue, args[3]._zvalue, args[4]._zvalue))
    elif len(args)==6:
        return FaceVariable(args[0].domain,
                            f(args[0]._xvalue, args[1]._xvalue, args[2]._xvalue, args[3]._xvalue, args[4]._xvalue, args[5]._xvalue),
                            f(args[0]._yvalue, args[1]._yvalue, args[2]._yvalue, args[3]._yvalue, args[4]._yvalue, args[5]._yvalue),
                            f(args[0]._zvalue, args[1]._zvalue, args[2]._zvalue, args[3]._zvalue, args[4]._zvalue, args[5]._zvalue))
    elif len(args)==7:
        return FaceVariable(args[0].domain,
                            f(args[0]._xvalue, args[1]._xvalue, args[2]._xvalue, args[3]._xvalue, args[4]._xvalue, args[5]._xvalue, args[6]._xvalue),
                            f(args[0]._yvalue, args[1]._yvalue, args[2]._yvalue, args[3]._yvalue, args[4]._yvalue, args[5]._yvalue, args[6]._yvalue),
                            f(args[0]._zvalue, args[1]._zvalue, args[2]._zvalue, args[3]._zvalue, args[4]._zvalue, args[5]._zvalue, args[6]._zvalue))
    elif len(args)==8:
        return FaceVariable(args[0].domain,
                            f(args[0]._xvalue, args[1]._xvalue, args[2]._xvalue, args[3]._xvalue, args[4]._xvalue, args[5]._xvalue, args[6]._xvalue, args[7]._xvalue),
                            f(args[0]._yvalue, args[1]._yvalue, args[2]._yvalue, args[3]._yvalue, args[4]._yvalue, args[5]._yvalue, args[6]._yvalue, args[7]._yvalue),
                            f(args[0]._zvalue, args[1]._zvalue, args[2]._zvalue, args[3]._zvalue, args[4]._zvalue, args[5]._zvalue, args[6]._zvalue, args[7]._zvalue))
