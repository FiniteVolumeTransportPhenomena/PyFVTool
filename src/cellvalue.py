"""
CellValue class definition
"""
import numpy as np
from mesh import *
from boundary import *

class CellVariable:
    def __init__(self, mesh_struct: MeshStructure, cell_value: np.ndarray, BC: BoundaryCondition):
        self.domain = mesh_struct
        self.value = cell_value
        self.bc = BC