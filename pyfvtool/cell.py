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
    """Create a cell variable of class CellVariable
    createCellVariable(mesh_struct: MeshStructure, cell_value: np.ndarray, BC: BoundaryCondition) -> CellVariable
    """
    if np.isscalar(cell_value):
        phi_val = cell_value*np.ones(mesh_struct.dims)
    elif cell_value.size == 1:
        phi_val = cell_value*np.ones(mesh_struct.dims)
    elif np.all(np.array(cell_value.shape)==mesh_struct.dims):
        phi_val = cell_value
    else:
        raise Exception(f"The cell size {cell_value.shape} is not valid for a mesh of size {mesh_struct.dims}.")
    
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
        c=m.cellcenters.x*m.cellsize.x[1:-1][:,np.newaxis,np.newaxis]*m.cellsize.y[1:-1][np.newaxis,:,np.newaxis]*z[np.newaxis,np.newaxis,:]
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
