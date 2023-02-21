import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.linalg import spsolve
from pyfvtool.mesh import *
from pyfvtool.utilities import *
from pyfvtool.cell import *
from pyfvtool.face import *



def solvePDE(m: MeshStructure, M:csr_array, RHS: np.ndarray) -> CellVariable:
    phi = spsolve(M, RHS)
    return CellVariable(m, np.reshape(phi, m.dims+2))