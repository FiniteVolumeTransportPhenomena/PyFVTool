# CellVariable class definition and operator overloading

from copy import deepcopy
from typing import overload


import numpy as np

from .mesh import MeshStructure
from .mesh import Grid1D, Grid2D, Grid3D
from .mesh import CylindricalGrid1D, CylindricalGrid2D
from .mesh import SphericalGrid1D, PolarGrid2D, CylindricalGrid3D, SphericalGrid3D
from .boundary import BoundaryConditionsBase, BoundaryConditions
from .boundary import cellValuesWithBoundaries, boundaryConditionsTerm
from .utilities import TrackedArray



class CellVariable:

    @overload
    def __init__(self, mesh_struct: MeshStructure, cell_value: np.ndarray,
                 BC: BoundaryConditionsBase):
        ...

    @overload
    def __init__(self, mesh_struct: MeshStructure, cell_value: np.ndarray):
        ...

    @overload
    def __init__(self, mesh_struct: MeshStructure, cell_value: float,
                 BC: BoundaryConditionsBase):
        ...

    @overload
    def __init__(self, mesh_struct: MeshStructure, cell_value: float):
        ...

    def __init__(self, mesh_struct: MeshStructure, cell_value, *arg,
                 BCsTerm_precalc = True):
        """
        Create a cell variable of class CellVariable

        Parameters
        ----------
        mesh_struct : MeshStructure
            Mesh describing the calculation domain.
        cell_value : float or numpy.ndarray
            Initialization value(s) of the CellVariable
        BC: BoundaryConditions
            Boundary conditions to be applied to the cell variable.
            *Required if CellVariable represents a solution variable.* This
            requirement also applies if default 'no-flux' boundary conditions are
            desired, in which case the BoundaryCondition should be created without
            further parameters (see .boundary.BoundaryConditions)
        BCsTerm_precalc : Boolean, optional
            Pre-calculate the matrix equation terms for the boundary conditions.
            The default is True. It can be switched to False, if these matrix
            equation terms are not needed, avoiding unneeded computations.
            

        Raises
        ------
        ValueError
            The shape of cell_value does not correspond to the mesh shape.

        Returns
        -------
        CellVariable
            An initialized instance of CellVariable.

        """
        self.BCsTerm_precalc = BCsTerm_precalc
        self.domain = mesh_struct
        self._value = None
        # After initialization, make sure that `_value` is a TrackedArray.
        # Also, when directly re-assigning `_value` use TrackedArray. Since
        # `_value` is only accessed by PyFVTool code internally, and never 
        # by the normal user, this note is only of importance to those who 
        # directly change the PyFVTool code base.

        if np.isscalar(cell_value):
            phi_val = cell_value*np.ones(mesh_struct.dims)
        elif cell_value.size == 1:
            phi_val = cell_value*np.ones(mesh_struct.dims)
        elif np.all(np.array(cell_value.shape)==mesh_struct.dims):
            phi_val = cell_value
        elif np.all(np.array(cell_value.shape)==mesh_struct.dims+2):
            # Values for ghost cells already included,
            # simply fill
            self._value = TrackedArray(cell_value)
        else:
            raise ValueError(f"The cell size {cell_value.shape} is not valid "\
                             f"for a mesh of size {mesh_struct.dims}.")
        if len(arg)==1:
            self.BCs = arg[0]
        elif len(arg)==0:
            self.BCs = BoundaryConditions(self.domain)
        else:
            raise Exception('Incorrect number of arguments')

        if self._value is None:
            # initialize self._value incl. ghost cells
            self._value = TrackedArray(cellValuesWithBoundaries(phi_val, 
                                                                self.BCs))
        if self.BCsTerm_precalc:
            self._BCsTerm  = boundaryConditionsTerm(self.BCs)
        self.value.modified = False

    @property
    def value(self):
        if issubclass(type(self.domain), Grid1D):
            return self._value[1:-1]
        elif issubclass(type(self.domain), Grid2D):
            return self._value[1:-1, 1:-1]
        elif issubclass(type(self.domain), Grid3D):
            return self._value[1:-1, 1:-1, 1:-1]
        
    @value.setter
    def value(self, values):
        if issubclass(type(self.domain), Grid1D):
            self._value[1:-1] = values
        elif issubclass(type(self.domain), Grid2D):
            self._value[1:-1, 1:-1] = values
        elif issubclass(type(self.domain), Grid3D):
            self._value[1:-1, 1:-1, 1:-1] = values
    
    # read-only property cellvolume
    @property
    def cellvolume(self):
        return self.domain.cellvolume
        
    # read-only property cellcenters
    @property
    def cellcenters(self):
        return self.domain.cellcenters
        
    
    def __add__(self, other):
        if type(other) is CellVariable:
            return CellVariable(self.domain, 
                                self.value + other.value,
                                deepcopy(self.BCs))
        else:
            return CellVariable(self.domain, 
                                self.value + other,
                                deepcopy(self.BCs))

    def __radd__(self, other):
        if type(other) is CellVariable:
            return CellVariable(self.domain, 
                                self.value + other.value,
                                deepcopy(self.BCs))
        else:
            return CellVariable(self.domain, 
                                self.value + other,
                                deepcopy(self.BCs))

    def __rsub__(self, other):
        if type(other) is CellVariable:
            return CellVariable(self.domain, 
                                other.value - self.value,
                                deepcopy(self.BCs))
        else:
            return CellVariable(self.domain, 
                                other - self.value,
                                deepcopy(self.BCs))
    
    def __sub__(self, other):
        if type(other) is CellVariable:
            return CellVariable(self.domain,
                                self.value - other.value,
                                deepcopy(self.BCs))
        else:
            return CellVariable(self.domain, 
                                self.value - other,
                                deepcopy(self.BCs))

    def __mul__(self, other):
        if type(other) is CellVariable:
            return CellVariable(self.domain, 
                                self.value * other.value,
                                deepcopy(self.BCs))
        else:
            return CellVariable(self.domain, 
                                self.value * other,
                                deepcopy(self.BCs))

    def __rmul__(self, other):
        if type(other) is CellVariable:
            return CellVariable(self.domain, 
                                self.value * other.value,
                                deepcopy(self.BCs))
        else:
            return CellVariable(self.domain, 
                                self.value * other,
                                deepcopy(self.BCs))

    def __truediv__(self, other):
        if type(other) is CellVariable:
            return CellVariable(self.domain, 
                                self.value / other.value,
                                deepcopy(self.BCs))
        else:
            return CellVariable(self.domain, 
                                self.value / other,
                                deepcopy(self.BCs))

    def __rtruediv__(self, other):
        if type(other) is CellVariable:
            return CellVariable(self.domain, 
                                other.value / self.value,
                                deepcopy(self.BCs))
        else:
            return CellVariable(self.domain, 
                                other / self.value,
                                deepcopy(self.BCs))
    
    def __neg__(self):
        return CellVariable(self.domain, -self.value,
                            deepcopy(self.BCs))
    
    def __pow__(self, other):
        if type(other) is CellVariable:
            return CellVariable(self.domain, 
                                self.value**other.value,
                                deepcopy(self.BCs))
        else:
            return CellVariable(self.domain, 
                                self.value**other,
                                deepcopy(self.BCs))
    
    def __rpow__(self, other):
        if type(other) is CellVariable:
            return CellVariable(self.domain, 
                                other.value**self.value,
                                deepcopy(self.BCs))
        else:
            return CellVariable(self.domain, 
                                other**self.value,
                                deepcopy(self.BCs))
    
    def __gt__(self, other):
        if type(other) is CellVariable:
            return CellVariable(self.domain, 
                                self.value>other.value,
                                deepcopy(self.BCs))
        else:
            return CellVariable(self.domain, 
                                self.value>other,
                                deepcopy(self.BCs))

    def __ge__(self, other):
        if type(other) is CellVariable:
            return CellVariable(self.domain, 
                                self.value>=other.value,
                                deepcopy(self.BCs))
        else:
            return CellVariable(self.domain, 
                                self.value>=other,
                                deepcopy(self.BCs))

    def __lt__(self, other):
        if type(other) is CellVariable:
            return CellVariable(self.domain, 
                                self.value<other.value,
                                deepcopy(self.BCs))
        else:
            return CellVariable(self.domain, 
                                self.value<other,
                                deepcopy(self.BCs))
    
    def __le__(self, other):
        if type(other) is CellVariable:
            return CellVariable(self.domain,
                                self.value<=other.value,
                                deepcopy(self.BCs))
        else:
            return CellVariable(self.domain, 
                                self.value<=other,
                                deepcopy(self.BCs))

    def __and__(self, other):
        if type(other) is CellVariable:
            return CellVariable(self.domain, 
                                np.logical_and(self.value, 
                                               other.value),
                                deepcopy(self.BCs))
        else:
            return CellVariable(self.domain, 
                                np.logical_and(self.value, other),
                                deepcopy(self.BCs))
    
    def __or__(self, other):
        if type(other) is CellVariable:
            return CellVariable(self.domain, 
                                np.logical_or(self.value, 
                                              other.value),
                                deepcopy(self.BCs))
        else:
            return CellVariable(self.domain, 
                                np.logical_or(self.value, other),
                                deepcopy(self.BCs))
    
    def __abs__(self):
        return CellVariable(self.domain,
                            np.abs(self.value),
                            deepcopy(self.BCs))


    def apply_BCs(self):
        """(Re)initialize ghost cells according to the boundary conditions and
        the internal (inner) cell values.
        
        It is necessary to explicitly call this method in certain special cases, 
        in particular when the CellVariable is used prior to using `solvePDE()`. 
        Or when `solvePDE()` is not used at all, typically when working with
        the 'expert-level' function `solveMatrixPDE()`.
        
        In general, superfluous calls to apply_BCs() will not hurt.
        
        Returns
        -------
        None.

        The `modified` attribute of the `CellVariableBCs` is reset, as well as
        the `modified` attribute of the `CellVariable.value`

        """
        self._value = TrackedArray(cellValuesWithBoundaries(self.value,
                                                            self.BCs))
        if self.BCsTerm_precalc:
            self._BCsTerm = boundaryConditionsTerm(self.BCs)
 
        self.BCs.modified = False
        self.value.modified = False
        
        
    def update_value(self, new_cell):
        np.copyto(self._value, new_cell._value)
        self._value.modified = True
  
    
    def copy(self):
        """
        Create a copy of the CellVariable
        
        
        Returns
        -------
        CellVariable
            Copy of the CellVariable.
        """
        return CellVariable(self.domain, np.copy(self._value),
                            deepcopy(self.BCs))
    
    def plotprofile(self):
        """
        Create a 'profile' of a cell variable for plotting, export, etc. 
        
        'Plot profiles' are sets of arrays that contain the axes' coordinates,
        cell values, including the values at the boundaries.
        
        For 2D and 3D visualization, it is perhaps better to use only the 
        actual cell values (e.g. for a false color map à la plt.pcolormesh),
        and not include the values at the boundaries.
        

        1D meshes
        =========
        This generates a pair of vectors containing the abscissa and ordinates
        for plotting the values of the cell variable over the entire calculation
        domain. It includes the values at the outer faces of the domain, by 
        taking into account the values of the ghost cells.
        
        Returns
        -------
        x : np.ndarray
            x (or r) coordinates.
        phi0 : np.ndarray
            Value of the CellVariables at those points.


        2D meshes
        =========
        This generates a set of vectors containing the (x, y) or (r, z)  
        coordinates and the values of the cell variable at those coordinates
        for plotting the values of the cell variable over the entire calculation
        domain. It includes the values at the outer faces of the domain, by 
        taking into account the values of the ghost cells.

        Returns
        -------
        x : np.ndarray
            x (or r) coordinates.
        y : np.ndarray
            y (or z) coordinates.
        phi0 : np.ndarray
            Value of the CellVariables at those coordinates.
        
        
        3D meshes
        =========
        This generates a set of vectors containing the (x, y, z) or (r, theta, z)  
        coordinates and the values of the cell variable at those coordinates
        for plotting the values of the cell variable over the entire calculation
        domain. It includes the values at the outer faces of the domain, by 
        taking into account the values of the ghost cells.

        Returns
        -------
        x : np.ndarray
            x (or r) coordinates.
        y : np.ndarray
            y (or theta) coordinates.
        z : np.ndarray
            z coordinates.
        phi0 : np.ndarray
            Value of the CellVariables at those coordinates.
        """
        #
        # TODO:
        #    Add a keyword that specifies how the outer FaceValues will be estimated
        #    Currently, this is just the average of the last inner cell and the boundary
        #    (ghost) cell.
        #    In certain cases it may be visually desirable to use an extrapolation of
        #    the last inner cell values.
        #
        if isinstance(self.domain, Grid1D):
            x = np.hstack([self.domain.facecenters._x[0],
                           self.domain.cellcenters._x,
                           self.domain.facecenters._x[-1]])
            phi0 = np.hstack([0.5*(self._value[0]+self._value[1]),
                              self._value[1:-1],
                              0.5*(self._value[-2]+self._value[-1])])
            # The size of the ghost cell is always equal to the size of the 
            # first (or last) cell within the domain. The value at the
            # boundary can therefore be obtained by direct averaging with a
            # weight factor of 0.5.
            return (x, phi0)
        elif isinstance(self.domain, Grid2D):
            x = np.hstack([self.domain.facecenters._x[0],
                           self.domain.cellcenters._x,
                           self.domain.facecenters._x[-1]])
            y = np.hstack([self.domain.facecenters._y[0],
                           self.domain.cellcenters._y,
                           self.domain.facecenters._y[-1]])
            phi0 = np.copy(self._value)
            phi0[:, 0] = 0.5*(phi0[:, 0]+phi0[:, 1])
            phi0[0, :] = 0.5*(phi0[0, :]+phi0[1, :])
            phi0[:, -1] = 0.5*(phi0[:, -1]+phi0[:, -2])
            phi0[-1, :] = 0.5*(phi0[-1, :]+phi0[-2, :])
            phi0[0, 0] = phi0[0, 1]
            phi0[0, -1] = phi0[0, -2]
            phi0[-1, 0] = phi0[-1, 1]
            phi0[-1, -1] = phi0[-1, -2]
            return (x, y, phi0)
        elif isinstance(self.domain, Grid3D):
            x = np.hstack([self.domain.facecenters._x[0],
                           self.domain.cellcenters._x,
                           self.domain.facecenters._x[-1]])[:, np.newaxis, np.newaxis]
            y = np.hstack([self.domain.facecenters._y[0],
                           self.domain.cellcenters._y,
                           self.domain.facecenters._y[-1]])[np.newaxis, :, np.newaxis]
            z = np.hstack([self.domain.facecenters._z[0],
                           self.domain.cellcenters._z,
                           self.domain.facecenters._z[-1]])[np.newaxis, np.newaxis, :]
            phi0 = np.copy(self._value)
            phi0[:,0,:]=0.5*(phi0[:,0,:]+phi0[:,1,:])
            phi0[:,-1,:]=0.5*(phi0[:,-2,:]+phi0[:,-1,:])
            phi0[:,:,0]=0.5*(phi0[:,:,0]+phi0[:,:,0])
            phi0[:,:,-1]=0.5*(phi0[:,:,-2]+phi0[:,:,-1])
            phi0[0,:,:]=0.5*(phi0[1,:,:]+phi0[2,:,:])
            phi0[-1,:,:]=0.5*(phi0[-2,:,:]+phi0[-1,:,:])
            return (x, y, z, phi0)
        else:
            raise NotImplementedError("plotprofile() not implemented for mesh type '{0:s}'".\
                            format(self.domain.__class__.__name__))
    
    
    def domainIntegral(self) -> float:
        """
        Calculate the finite-volume integral of a CellVariable over entire domain
        
        The finite-volume integral over the entire mesh domain gives the total
        amount of `CellVariable` present in the system. Calculation of this
        integral is useful for checking conservation of the quantity concerned
        (in case of 'no-flux' BCs), or for monitoring its evolution due to 
        exchanges via the boundaries (other BCs) or to the presence of source
        terms.
    
        May later become a built-in method of the CellVariable class, but for now 
        this implementation as a function is chosen for consistency with FVTool. 
    
    
        Returns
        -------
        float
            Total finite-volume integral over entire domain.
    
        """
        v = self.cellvolume
        c = self.value
        return (v*c).flatten().sum()



def cellLocations(m: MeshStructure):
    """
    this function returns the location of the cell centers as cell variables. 
    
    It can later be used in defining properties that are variable in space.
    
    Incompletely tested, and there may be other, more direct, ways to 
    calculate properties that vary in space, e.g. using cellcenters directly?
    
    Parameters
    ----------
    m : {MeshStructure object}
        Domain of the problem

    Returns
    -------
    X : {CellVariable object}
        Node x-positions        
    Y : {CellVariable object}
        Node y-positions        
    Z : {CellVariable object}
        Node z-positions        

    See Also
    --------
    faceLocations

    Notes
    -----

    Examples
    --------
    >>>     
    
    """    
    N = m.dims
   
    
    if (type(m) is Grid1D)\
     or (type(m) is CylindricalGrid1D)\
     or (type(m) is SphericalGrid1D):
        X = CellVariable(m, m.cellcenters._x)
        return X
    elif (type(m) is Grid2D)\
       or (type(m) is CylindricalGrid2D)\
       or (type(m) is PolarGrid2D): 
        X = CellVariable(m, np.tile(m.cellcenters._x[:, np.newaxis], (1, N[1])))
        Y = CellVariable(m, np.tile(m.cellcenters._y[:, np.newaxis].T, (N[0], 1)))
        return X, Y  
    elif (type(m) is Grid3D)\
       or (type(m) is CylindricalGrid3D)\
       or (type(m) is SphericalGrid3D): 
        X = CellVariable(m, np.tile(m.cellcenters._x[:, np.newaxis, np.newaxis], (1, N[1], N[2])))
        Y = CellVariable(m, np.tile((m.cellcenters._y[:, np.newaxis].T)[:,:,np.newaxis], (N[0], 1, N[2])))
        z = np.zeros((1,1,N[2]))
        z[0, 0, :] = m.cellcenters._z
        Z = CellVariable(m, np.tile(z, (N[0], N[1], 1)))
        return X, Y, Z
    raise TypeError('mesh type not implemented')
    return None 



def funceval(f, *args):
    if len(args)==1:
        return CellVariable(args[0].domain, 
                            f(args[0].value),
                            deepcopy(args[0].BCs))
    elif len(args)==2:
        return CellVariable(args[0].domain, 
                            f(args[0].value, 
                              args[1].value),
                            deepcopy(args[0].BCs))
    elif len(args)==3:
        return CellVariable(args[0].domain, 
                            f(args[0].value, 
                              args[1].value, 
                              args[2].value),
                            deepcopy(args[0].BCs))
    elif len(args)==4:
        return CellVariable(args[0].domain, 
                            f(args[0].value, 
                              args[1].value, 
                              args[2].value, 
                              args[3].value),
                            deepcopy(args[0].BCs))
    elif len(args)==5:
        return CellVariable(args[0].domain, 
                            f(args[0].value, 
                              args[1].value, 
                              args[2].value, 
                              args[3].value, 
                              args[4].value),
                            deepcopy(args[0].BCs))
    elif len(args)==6:
        return CellVariable(args[0].domain, 
                            f(args[0].value, 
                              args[1].value, 
                              args[2].value, 
                              args[3].value, 
                              args[4].value, 
                              args[5].value),
                            deepcopy(args[0].BCs))
    elif len(args)==7:
        return CellVariable(args[0].domain, 
                            f(args[0].value, 
                              args[1].value, 
                              args[2].value, 
                              args[3].value, 
                              args[4].value, 
                              args[5].value, 
                              args[6].value),
                            deepcopy(args[0].BCs))
    elif len(args)==8:
        return CellVariable(args[0].domain, 
                            f(args[0].value, 
                              args[1].value, 
                              args[2].value, 
                              args[3].value, 
                              args[4].value, 
                              args[5].value, 
                              args[6].value, 
                              args[7].value),
                            deepcopy(args[0].BCs))
    


def celleval(f, *args):
    return funceval(f, *args)



