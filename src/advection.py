# advection terms
import numpy as np
from scipy.sparse import csr_array
from mesh import *
from utilities import *
from cell import *
from face import *

def convectionTerm1D(u: FaceVariable):
    # u is a face variable
    # extract data from the mesh structure
    Nx = u.domain.dims[0]
    G = u.domain.cell_numbers()
    #DX = u.domain.cellsize.x
    DXe = u.domain.cellsize.x[2:]
    DXw = u.domain.cellsize.x[0:-2]
    DXp = u.domain.cellsize.x[1:-1]

    # reassign the east, west for code readability
    ue = u.xvalue[1:Nx+1]/(DXp+DXe)
    uw = u.xvalue[0:Nx]/(DXp+DXw)

    # calculate the coefficients for the internal cells
    AE = ue
    AW = -uw
    APx = (ue*DXe-uw*DXw)/DXp

    # build the sparse matrix based on the numbering system
    iix = np.tile(G[1:Nx+1].ravel(), 3)
    jjx = np.hstack([G[0:Nx], G[1:Nx+1], G[2:Nx+2]])
    sx = np.hstack([AW, APx, AE])

    # build the sparse matrix
    kx = 3*Nx
    return csr_array((sx[0:kx], (iix[0:kx], jjx[0:kx])), 
                     shape=(Nx+2, Nx+2))

def convectionUpwindTerm1D(u: FaceVariable, *args):
    # u is a face variable
    # extract data from the mesh structure
    Nx = u.domain.dims[0]
    G = u.domain.cell_numbers()
    DXp = u.domain.cellsize.x[1:-1]

    # find the velocity direction for the upwind scheme
    ue_min = np.minimum(u.xvalue[1:Nx+1], 0.0)
    ue_max = np.maximum(u.xvalue[1:Nx+1], 0.0)
    uw_min = np.minimum(u.xvalue[0:Nx], 0.0)
    uw_max = np.maximum(u.xvalue[0:Nx], 0.0)

    # calculate the coefficients for the internal cells
    AE = ue_min/DXp
    AW = -uw_max/DXp
    APx = (ue_max-uw_min)/DXp

    # correct for the cells next to the boundary
    # Left boundary:
    APx[0] = APx[0]-uw_max[0]/(2.0*DXp[0])
    AW[0] = AW[0]/2.0
    # Right boundary:
    AE[-1] = AE[-1]/2.0
    APx[-1] = APx[-1] + ue_min[-1]/(2.0*DXp[-1])

    # build the sparse matrix based on the numbering system
    iix = np.tile(G[2:Nx+1], 3)
    jjx = np.hstack([G[0:Nx], G[1:Nx+1], G[2:Nx+2]])
    sx = np.hstack([AW, APx, AE])

    # build the sparse matrix
    kx = 3*Nx
    return csr_array((sx[0:kx], (iix[0:kx], jjx[0:kx])), 
                     shape=(Nx+2, Nx+2))