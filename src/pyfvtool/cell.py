# CellVariable class definition and operator overloading

import numpy as np
from typing import overload

from .mesh import MeshStructure
from .mesh import Grid1D, Grid2D, Grid3D
from .mesh import CylindricalGrid1D, CylindricalGrid2D
from .mesh import PolarGrid2D, CylindricalGrid3D
from .boundary import BoundaryConditionsBase, BoundaryConditions
from .boundary import cellValuesWithBoundaries

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

    def __init__(self, mesh_struct: MeshStructure, cell_value, *arg):
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
            

        Raises
        ------
        ValueError
            The shape of cell_value does not correspond to the mesh shape.

        Returns
        -------
        CellVariable
            An initialized instance of CellVariable.

        """
                
        self.domain = mesh_struct
        self.value = None

        if np.isscalar(cell_value):
            phi_val = cell_value*np.ones(mesh_struct.dims)
        elif cell_value.size == 1:
            phi_val = cell_value*np.ones(mesh_struct.dims)
        elif np.all(np.array(cell_value.shape)==mesh_struct.dims):
            phi_val = cell_value
        elif np.all(np.array(cell_value.shape)==mesh_struct.dims+2):
            # Values for boundary cells already included,
            # simply fill
            self.value = cell_value
        else:
            raise ValueError(f"The cell size {cell_value.shape} is not valid "\
                             f"for a mesh of size {mesh_struct.dims}.")
                
        if self.value is None:
            if len(arg)==1:
                self.value = cellValuesWithBoundaries(phi_val, arg[0])
            elif len(arg)==0:
                self.value = cellValuesWithBoundaries(phi_val, 
                                 BoundaryConditions(mesh_struct))
            else:
                raise Exception('Incorrect number of arguments')

    @property
    def internalCellValues(self):
        if issubclass(type(self.domain), Grid1D):
            return self.value[1:-1]
        elif issubclass(type(self.domain), Grid2D):
            return self.value[1:-1, 1:-1]
        elif issubclass(type(self.domain), Grid3D):
            return self.value[1:-1, 1:-1, 1:-1]
        
    @internalCellValues.setter
    def internalCellValues(self, values):
        if issubclass(type(self.domain), Grid1D):
            self.value[1:-1] = values
        elif issubclass(type(self.domain), Grid2D):
            self.value[1:-1, 1:-1] = values
        elif issubclass(type(self.domain), Grid3D):
            self.value[1:-1, 1:-1, 1:-1] = values
    
    def __add__(self, other):
        if type(other) is CellVariable:
            return CellVariable(self.domain, self.value+other.value)
        else:
            return CellVariable(self.domain, self.value+other)

    def __radd__(self, other):
        if type(other) is CellVariable:
            return CellVariable(self.domain, self.value+other.value)
        else:
            return CellVariable(self.domain, self.value+other)

    def __rsub__(self, other):
        if type(other) is CellVariable:
            return CellVariable(self.domain, other.value-self.value)
        else:
            return CellVariable(self.domain, other-self.value)
    
    def __sub__(self, other):
        if type(other) is CellVariable:
            return CellVariable(self.domain, self.value-other.value)
        else:
            return CellVariable(self.domain, self.value-other)

    def __mul__(self, other):
        if type(other) is CellVariable:
            return CellVariable(self.domain, self.value*other.value)
        else:
            return CellVariable(self.domain, self.value*other)

    def __rmul__(self, other):
        if type(other) is CellVariable:
            return CellVariable(self.domain, self.value*other.value)
        else:
            return CellVariable(self.domain, self.value*other)

    def __truediv__(self, other):
        if type(other) is CellVariable:
            return CellVariable(self.domain, self.value/other.value)
        else:
            return CellVariable(self.domain, self.value/other)

    def __rtruediv__(self, other):
        if type(other) is CellVariable:
            return CellVariable(self.domain, other.value/self.value)
        else:
            return CellVariable(self.domain, other/self.value)
    
    def __neg__(self):
        return CellVariable(self.domain, -self.value)
    
    def __pow__(self, other):
        if type(other) is CellVariable:
            return CellVariable(self.domain, self.value**other.value)
        else:
            return CellVariable(self.domain, self.value**other)
    
    def __rpow__(self, other):
        if type(other) is CellVariable:
            return CellVariable(self.domain, other.value**self.value)
        else:
            return CellVariable(self.domain, other**self.value)
    
    def __gt__(self, other):
        if type(other) is CellVariable:
            return CellVariable(self.domain, self.value>other.value)
        else:
            return CellVariable(self.domain, self.value>other)

    def __ge__(self, other):
        if type(other) is CellVariable:
            return CellVariable(self.domain, self.value>=other.value)
        else:
            return CellVariable(self.domain, self.value>=other)

    def __lt__(self, other):
        if type(other) is CellVariable:
            return CellVariable(self.domain, self.value<other.value)
        else:
            return CellVariable(self.domain, self.value<other)
    
    def __le__(self, other):
        if type(other) is CellVariable:
            return CellVariable(self.domain, self.value<=other.value)
        else:
            return CellVariable(self.domain, self.value<=other)

    def __and__(self, other):
        if type(other) is CellVariable:
            return CellVariable(self.domain, np.logical_and(self.value, other.value))
        else:
            return CellVariable(self.domain, np.logical_and(self.value, other))
    
    def __or__(self, other):
        if type(other) is CellVariable:
            return CellVariable(self.domain, np.logical_or(self.value, other.value))
        else:
            return CellVariable(self.domain, np.logical_or(self.value, other))
    
    def __abs__(self):
        return CellVariable(self.domain, np.abs(self.value))


    def update_bc_cells(self, BC: BoundaryConditionsBase):
        phi_temp = CellVariable(self.domain, self.internalCellValues, BC)
        self.update_value(phi_temp)

    def update_value(self, new_cell):
        np.copyto(self.value, new_cell.value)

    def bc_to_ghost(self):
        """
        assign the boundary values to the ghost cells
        """
        if issubclass(type(self.domain), Grid1D):
            self.value[0] = 0.5*(self.value[1]+self.value[0])
            self.value[-1] = 0.5*(self.value[-2]+self.value[-1])
        elif issubclass(type(self.domain), Grid2D):
            self.value[0, 1:-1] = 0.5*(self.value[1, 1:-1]+self.value[0, 1:-1])
            self.value[-1, 1:-1] = 0.5*(self.value[-2, 1:-1]+self.value[-1, 1:-1])
            self.value[1:-1, 0] = 0.5*(self.value[1:-1, 1]+self.value[1:-1, 0])
            self.value[1:-1, -1] = 0.5*(self.value[1:-1, -2]+self.value[1:-1, -1])
        elif issubclass(type(self.domain), Grid3D):
            self.value[0, 1:-1, 1:-1] = 0.5*(self.value[1, 1:-1, 1:-1]+self.value[0, 1:-1, 1:-1])
            self.value[-1, 1:-1, 1:-1] = 0.5*(self.value[-2, 1:-1, 1:-1]+self.value[-1, 1:-1, 1:-1])
            self.value[1:-1, 0, 1:-1] = 0.5*(self.value[1:-1, 1, 1:-1]+self.value[1:-1, 0, 1:-1])
            self.value[1:-1, -1, 1:-1] = 0.5*(self.value[1:-1, -2, 1:-1]+self.value[1:-1, -1, 1:-1])
            self.value[1:-1, 1:-1, 0] = 0.5*(self.value[1:-1, 1:-1, 1]+self.value[1:-1, 1:-1, 0])
            self.value[1:-1, 1:-1, -1] = 0.5*(self.value[1:-1, 1:-1, -2]+self.value[1:-1, 1:-1, -1])
    
    def copy(self):
        """
        Create a copy of the CellVariable
        
        
        Returns
        -------
        CellVariable
            Copy of the CellVariable.
        """
        return CellVariable(self.domain, np.copy(self.value))
    
    def plotprofile(self):
        """
        Create a profile of a cell variable for plotting, export, etc. 
        
        It generates sets of arrays that contain the axes' coordinates,
        cell values, including the values at the boundaries.
        
        For 2D and 3D visualization, it is perhaps best to only use the 
        internalCellValues for plotting (e.g. for a false color map à la 
        plt.pcolormesh)
        

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
            x = np.hstack([self.domain.facecenters.x[0],
                           self.domain.cellcenters.x,
                           self.domain.facecenters.x[-1]])
            phi0 = np.hstack([0.5*(self.value[0]+self.value[1]),
                              self.value[1:-1],
                              0.5*(self.value[-2]+self.value[-1])])
            # The size of the ghost cell is always equal to the size of the 
            # first (or last) cell within the domain. The value at the
            # boundary can therefore be obtained by direct averaging with a
            # weight factor of 0.5.
            return (x, phi0)
        elif isinstance(self.domain, Grid2D):
            x = np.hstack([self.domain.facecenters.x[0],
                           self.domain.cellcenters.x,
                           self.domain.facecenters.x[-1]])
            y = np.hstack([self.domain.facecenters.y[0],
                           self.domain.cellcenters.y,
                           self.domain.facecenters.y[-1]])
            phi0 = np.copy(self.value)
            phi0[:, 0] = 0.5*(phi0[:, 0]+phi0[:, 1])
            phi0[0, :] = 0.5*(phi0[0, :]+phi0[1, :])
            phi0[:, -1] = 0.5*(phi0[:, -1]+phi0[:, -2])
            phi0[-1, :] = 0.5*(phi0[-1, :]+phi0[-2, :])
            phi0[0, 0] = phi0[0, 1]
            phi0[0, -1] = phi0[0, -2]
            phi0[-1, 0] = phi0[-1, 1]
            phi0[-1, -1] = phi0[-1, -2]
            return (x, y, phi0)
        else:
            raise NotImplementedError("plotprofile() not implemented for mesh type '{0:s}'".\
                            format(self.domain.__class__.__name__))




def cellVolume(m: MeshStructure):
    BC = BoundaryConditions(m)
    if (type(m) is Grid1D):
        c=m.cellsize.x[1:-1]
    elif (type(m) is CylindricalGrid1D):
        c=2.0*np.pi*m.cellsize.x[1:-1]*m.cellcenters.x
    elif (type(m) is Grid2D):
        c=m.cellsize.x[1:-1][:, np.newaxis]*m.cellsize.y[1:-1][np.newaxis, :]
    elif (type(m) is CylindricalGrid2D):
        c=2.0*np.pi*m.cellcenters.x[:, np.newaxis]*m.cellsize.x[1:-1][:, np.newaxis]*m.cellsize.y[1:-1][np.newaxis, :]
    elif (type(m) is PolarGrid2D):
        c=m.cellcenters.x*m.cellsize.x[1:-1][:, np.newaxis]*m.cellsize.y[1:-1][np.newaxis, :]
    elif (type(m) is Grid3D):
        c=m.cellsize.x[1:-1][:,np.newaxis,np.newaxis]*m.cellsize.y[1:-1][np.newaxis,:,np.newaxis]*m.cellsize.z[1:-1][np.newaxis,np.newaxis,:]
    elif (type(m) is CylindricalGrid3D):
        c=m.cellcenters.x*m.cellsize.x[1:-1][:,np.newaxis,np.newaxis]*m.cellsize.y[1:-1][np.newaxis,:,np.newaxis]*m.cellsize.z[np.newaxis,np.newaxis,:]
    return CellVariable(m, c, BC)



def cellLocations(m: MeshStructure):
    """
    this function returns the location of the cell centers as cell variables. 
    
    It can later be used in defining properties that are variable in space.
    
    Incompletely tested
    
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
     or (type(m) is CylindricalGrid1D):
        X = CellVariable(m, m.cellcenters.x)
        return X
    elif (type(m) is Grid2D)\
       or (type(m) is CylindricalGrid2D)\
       or (type(m) is PolarGrid2D): 
        X = CellVariable(m, np.tile(m.cellcenters.x[:, np.newaxis], (1, N[1])))
        Y = CellVariable(m, np.tile(m.cellcenters.y[:, np.newaxis].T, (N[0], 1)))
        return X, Y  
    elif (type(m) is Grid3D)\
       or (type(m) is CylindricalGrid3D): 
        X = CellVariable(m, np.tile(m.cellcenters.x[:, np.newaxis, np.newaxis], (1, N[1], N[2])))
        Y = CellVariable(m, np.tile((m.cellcenters.y[:, np.newaxis].T)[:,:,np.newaxis], (N[0], 1, N[2])))
        z = np.zeros((1,1,N[2]))
        z[0, 0, :] = m.cellcenters.z
        Z = CellVariable(m, np.tile(z, (N[0], N[1], 1)))
        return X, Y, Z
    raise TypeError('mesh type not implemented')
    return None 


def funceval(f, *args):
    if len(args)==1:
        return CellVariable(args[0].domain, 
                            f(args[0].value))
    elif len(args)==2:
        return CellVariable(args[0].domain, 
                            f(args[0].value, args[1].value))
    elif len(args)==3:
        return CellVariable(args[0].domain, 
                            f(args[0].value, args[1].value, args[2].value))
    elif len(args)==4:
        return CellVariable(args[0].domain, 
                            f(args[0].value, args[1].value, args[2].value, args[3].value))
    elif len(args)==5:
        return CellVariable(args[0].domain, 
                            f(args[0].value, args[1].value, args[2].value, args[3].value, args[4].value))
    elif len(args)==6:
        return CellVariable(args[0].domain, 
                            f(args[0].value, args[1].value, args[2].value, args[3].value, args[4].value, args[5].value))
    elif len(args)==7:
        return CellVariable(args[0].domain, 
                            f(args[0].value, args[1].value, args[2].value, args[3].value, args[4].value, args[5].value, args[6].value))
    elif len(args)==8:
        return CellVariable(args[0].domain, 
                            f(args[0].value, args[1].value, args[2].value, args[3].value, args[4].value, args[5].value, args[6].value, args[7].value))
    


def celleval(f, *args):
    return funceval(f, *args)



def domainInt(phi: CellVariable) -> float:
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
    An alias `domainIntegrate`has been added that sounds better than
    `domainInt`.

    Parameters
    ----------
    phi : CellVariable
        Variable whose finite-volume integral will calculated.

    Returns
    -------
    float
        Total finite-volume integral over entire domain.

    """
    v = cellVolume(phi.domain).internalCellValues
    c = phi.internalCellValues
    return (v*c).flatten().sum()


def domainIntegrate(phi: CellVariable) -> float:
    """
    Alias for `domainInt()`
    
    See `domainInt()`
    """
    return domainInt(phi)




# TODO:
# get_CellVariable_profile3D can become a method of CellVariable
#    (shared with 1D and 2D versions)
# TODO:
#    Add a keyword that specifies how the outer FaceValues will be estimated
#    Currently, this is just the average of the last inner cell and the boundary
#    (ghost) cell.
#    In certain cases it may be visually desirable to use an extrapolation of
#    the last inner cell values.
# Perhaps for 2D and 3D visualization it is perhaps best to only use the 
# innervalues (e.g. for a false color map à la plt.pcolormesh)

def get_CellVariable_profile3D(phi: CellVariable):
    """
    Create a profile of a cell variable for plotting, export, etc. (3D).
    
    This generates a set of vectors containing the (x, y, z) or (r, theta, z)  
    coordinates and the values of the cell variable at those coordinates
    for plotting the values of the cell variable over the entire calculation
    domain. It includes the values at the outer faces of the domain, by 
    taking into account the values of the ghost cells.
    
    This function may later become a method of the CellVariable class, but
    is a function now for simplicity and consistency with other CellVariable
    utility functions (e.g. `domainIntegrate`).

    Parameters
    ----------
    phi : CellVariable


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
    
    x = np.hstack([phi.domain.facecenters.x[0],
                   phi.domain.cellcenters.x,
                   phi.domain.facecenters.x[-1]])[:, np.newaxis, np.newaxis]
    y = np.hstack([phi.domain.facecenters.y[0],
                   phi.domain.cellcenters.y,
                   phi.domain.facecenters.y[-1]])[np.newaxis, :, np.newaxis]
    z = np.hstack([phi.domain.facecenters.z[0],
                   phi.domain.cellcenters.z,
                   phi.domain.facecenters.z[-1]])[np.newaxis, np.newaxis, :]
    phi0 = np.copy(phi.value)
    phi0[:,0,:]=0.5*(phi0[:,0,:]+phi0[:,1,:])
    phi0[:,-1,:]=0.5*(phi0[:,-2,:]+phi0[:,-1,:])
    phi0[:,:,0]=0.5*(phi0[:,:,0]+phi0[:,:,0])
    phi0[:,:,-1]=0.5*(phi0[:,:,-2]+phi0[:,:,-1])
    phi0[0,:,:]=0.5*(phi0[1,:,:]+phi0[2,:,:])
    phi0[-1,:,:]=0.5*(phi0[-2,:,:]+phi0[-1,:,:])
    return (x, y, z, phi0)

def BC2GhostCells(phi0):
    """
    assign the boundary values to the ghost cells and returns the new cell variable
    """
    phi = phi0.copy()
    if issubclass(type(phi.domain), Grid1D):
        phi.value[0] = 0.5*(phi.value[1]+phi.value[0])
        phi.value[-1] = 0.5*(phi.value[-2]+phi.value[-1])
    elif issubclass(type(phi.domain), Grid2D):
        phi.value[0, 1:-1] = 0.5*(phi.value[1, 1:-1]+phi.value[0, 1:-1])
        phi.value[-1, 1:-1] = 0.5*(phi.value[-2, 1:-1]+phi.value[-1, 1:-1])
        phi.value[1:-1, 0] = 0.5*(phi.value[1:-1, 1]+phi.value[1:-1, 0])
        phi.value[1:-1, -1] = 0.5*(phi.value[1:-1, -2]+phi.value[1:-1, -1])
    elif issubclass(type(phi.domain), Grid3D):
        phi.value[0, 1:-1, 1:-1] = 0.5*(phi.value[1, 1:-1, 1:-1]+phi.value[0, 1:-1, 1:-1])
        phi.value[-1, 1:-1, 1:-1] = 0.5*(phi.value[-2, 1:-1, 1:-1]+phi.value[-1, 1:-1, 1:-1])
        phi.value[1:-1, 0, 1:-1] = 0.5*(phi.value[1:-1, 1, 1:-1]+phi.value[1:-1, 0, 1:-1])
        phi.value[1:-1, -1, 1:-1] = 0.5*(phi.value[1:-1, -2, 1:-1]+phi.value[1:-1, -1, 1:-1])
        phi.value[1:-1, 1:-1, 0] = 0.5*(phi.value[1:-1, 1:-1, 1]+phi.value[1:-1, 1:-1, 0])
        phi.value[1:-1, 1:-1, -1] = 0.5*(phi.value[1:-1, 1:-1, -2]+phi.value[1:-1, 1:-1, -1])
    return phi