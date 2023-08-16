# CellValue class definition and operator overloading

import numpy as np
from typing import overload
from .mesh import *
from .boundary import *

class CellVariable:
    def __init__(self, mesh_struct: MeshStructure, cell_value: np.ndarray):
        self.domain = mesh_struct
        if np.all(np.array(cell_value.shape)==mesh_struct.dims+2):
            self.value = cell_value
        else:
              raise Exception("The cell value is not valid. Check the size of the input array.")

        self.value = cell_value

    def internalCells(self):
        if issubclass(type(self.domain), Mesh1D):
            return self.value[1:-1]
        elif issubclass(type(self.domain), Mesh2D):
            return self.value[1:-1, 1:-1]
        elif issubclass(type(self.domain), Mesh3D):
            return self.value[1:-1, 1:-1, 1:-1]
    
    def update_bc_cells(self, BC: BoundaryCondition):
        phi_temp = createCellVariable(self.domain, self.internalCells(), BC)
        self.update_value(phi_temp)

    def update_value(self, new_cell):
        self.value[:] = new_cell.value
    
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

@overload
def createCellVariable(mesh_struct: MeshStructure, cell_value: np.ndarray, BC: BoundaryCondition) -> CellVariable:
    ...

@overload
def createCellVariable(mesh_struct: MeshStructure, cell_value: np.ndarray) -> CellVariable:
    ...

@overload
def createCellVariable(mesh_struct: MeshStructure, cell_value: float, BC: BoundaryCondition) -> CellVariable:
    ...

@overload
def createCellVariable(mesh_struct: MeshStructure, cell_value: float) -> CellVariable:
    ...

def createCellVariable(mesh_struct: MeshStructure, cell_value, *arg) -> CellVariable:
    """
    Create a cell variable of class CellVariable

    Parameters
    ----------
    mesh_struct : MeshStructure
        Mesh describing the calculation domain.
    cell_value : float or numpy.ndarray
        Initialization value(s) of the CellVariable
    BC: BoundaryCondition
        Boundary conditions to be applied to the cell variable.
        *Required if CellVariable represents a solution variable.* This
        requirement also applies if default 'no-flux' boundary conditions are
        desired, in which case the BoundaryCondition should be created without
        further parameters (see boundary.createBC)
        

    Raises
    ------
    ValueError
        The shape of cell_value does not correspond to the mesh shape.

    Returns
    -------
    CellVariable
        An initialized instance of CellVariable.

    """
    

    if np.isscalar(cell_value):
        phi_val = cell_value*np.ones(mesh_struct.dims)
    elif cell_value.size == 1:
        phi_val = cell_value*np.ones(mesh_struct.dims)
    elif np.all(np.array(cell_value.shape)==mesh_struct.dims):
        phi_val = cell_value
    else:
        raise ValueError(f"The cell size {cell_value.shape} is not valid for a mesh of size {mesh_struct.dims}.")
    
    if len(arg)==1:
        return CellVariable(mesh_struct, cellBoundary(phi_val, arg[0]))
    else:
        return CellVariable(mesh_struct, cellBoundary(phi_val, createBC(mesh_struct)))

def cellVolume(m: MeshStructure):
    BC = createBC(m)
    if (type(m) is Mesh1D):
        c=m.cellsize.x[1:-1]
    elif (type(m) is MeshCylindrical1D):
        c=2.0*np.pi*m.cellsize.x[1:-1]*m.cellcenters.x
    elif (type(m) is Mesh2D):
        c=m.cellsize.x[1:-1][:, np.newaxis]*m.cellsize.y[1:-1][np.newaxis, :]
    elif (type(m) is MeshCylindrical2D):
        c=2.0*np.pi*m.cellcenters.x[:, np.newaxis]*m.cellsize.x[1:-1][:, np.newaxis]*m.cellsize.y[1:-1][np.newaxis, :]
    elif (type(m) is MeshRadial2D):
        c=m.cellcenters.x*m.cellsize.x[1:-1][:, np.newaxis]*m.cellsize.y[1:-1][np.newaxis, :]
    elif (type(m) is Mesh3D):
        c=m.cellsize.x[1:-1][:,np.newaxis,np.newaxis]*m.cellsize.y[1:-1][np.newaxis,:,np.newaxis]*m.cellsize.z[1:-1][np.newaxis,np.newaxis,:]
    elif (type(m) is MeshCylindrical3D):
        c=m.cellcenters.x*m.cellsize.x[1:-1][:,np.newaxis,np.newaxis]*m.cellsize.y[1:-1][np.newaxis,:,np.newaxis]*m.cellsize.z[np.newaxis,np.newaxis,:]
    return createCellVariable(m, c, BC)

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
    v = cellVolume(phi.domain).internalCells()
    c = phi.internalCells()
    return (v*c).flatten().sum()

def domainIntegrate(phi: CellVariable) -> float:
    """
    Alias for `domainInt()`
    
    See `domainInt()`
    """
    return domainInt(phi)