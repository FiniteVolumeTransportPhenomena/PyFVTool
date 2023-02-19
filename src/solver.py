import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.linalg import spsolve
from mesh import *
from utilities import *
from cell import *
from face import *



def solvePDE(m: MeshStructure, M:csr_array, RHS: np.ndarray) -> CellVariable:
    phi = spsolve(M, RHS)
    return CellVariable(m, np.reshape(phi, m.dims+2))