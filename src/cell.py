# CellValue class definition and operator overloading

import numpy as np
from mesh import *
from boundary import *

class CellVariable:
    def __init__(self, mesh_struct: MeshStructure, cell_value: np.ndarray):
        self.domain = mesh_struct
        if np.all(np.array(cell_value.shape)==mesh_struct.dims+2):
            self.value = cell_value
        else:
              raise Exception("The cell value is not valid. Check the size of the input array.")

        self.value = cell_value

    def update_bc_cells(self, BC: BoundaryCondition):
        pass

    def update_value(self, new_cell):
        self.value[:] = new_cell.value
    def __add__(self, other):
        pass
    def __radd__(self, other):
        pass
    def __sub__(self, other):
        pass
    def __rsub__(self, other):
        pass
    def __mul__(self, other):
        pass
    def __rmul__(self, other):
        pass
    def __truediv__(self, other):
        pass
    def __rtruediv__(self, other):
        pass


def createCellVariable(mesh_struct: MeshStructure, cell_value: np.ndarray, BC: BoundaryCondition) -> CellVariable:
    """Create a cell variable of class CellVariable
    createCellVariable(mesh_struct: MeshStructure, cell_value: np.ndarray, BC: BoundaryCondition) -> CellVariable
    """

    if cell_value.size == 1:
        phi_val = cell_value*np.ones(mesh_struct.dims)
    elif np.all(np.array(cell_value.shape)==mesh_struct.dims):
        phi_val = cell_value
    else: 
        raise Exception("The cell size {cell_value.shape} is not valid for a mesh of size {mesh_struct.dims}.")
    return CellVariable(mesh_struct, cellBoundary(phi_val, BC))

