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
    if len(args) > 0:
        u_upwind = args[0]
    else:
        u_upwind = u
    # extract data from the mesh structure
    Nx = u.domain.dims[0]
    G = u.domain.cell_numbers()
    DXp = u.domain.cellsize.x[1:-1]

    ue_min = u.xvalue[1:Nx+1]
    ue_max = u.xvalue[1:Nx+1]
    uw_min = u.xvalue[0:Nx]
    uw_max = u.xvalue[0:Nx]

    # find the velocity direction for the upwind scheme
    ue_min[u_upwind.xvalue[1:Nx+1] > 0.0] = 0.0
    ue_max[u_upwind.xvalue[1:Nx+1] < 0.0] = 0.0
    uw_min[u_upwind.xvalue[0:Nx] > 0.0] = 0.0
    uw_max[u_upwind.xvalue[0:Nx] < 0.0] = 0.0

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
    iix = np.tile(G[1:Nx+1], 3)
    jjx = np.hstack([G[0:Nx], G[1:Nx+1], G[2:Nx+2]])
    sx = np.hstack([AW, APx, AE])

    # build the sparse matrix
    kx = 3*Nx
    return csr_array((sx[0:kx], (iix[0:kx], jjx[0:kx])),
                     shape=(Nx+2, Nx+2))


def fsign(phi_in, eps1=1e-16):
    return (np.abs(phi_in) >= eps1)*phi_in+eps1*(phi_in == 0.0)+eps1*(np.abs(phi_in) < eps1)*np.sign(phi_in)


def convectionTvdRHS1D(u: FaceVariable, phi: CellVariable,
                       FL, u_upwind: FaceVariable):
    # u is a face variable
    # phi is a cell variable
    # a function to avoid division by zero
    eps1 = 1.0e-20

    # extract data from the mesh structure
    Nx = u.domain.dims[0]
    DXp = u.domain.cellsize.x[1:-1]
    dx = 0.5*(u.domain.cellsize.x[0:-1]+u.domain.cellsize.x[1:])
    RHS = np.zeros(Nx+2)
    psi_p = np.zeros(Nx+1)
    psi_m = np.zeros(Nx+1)

    # calculate the upstream to downstream gradient ratios for u>0 (+ ratio)
    dphi_p = (phi.value[1:Nx+2]-phi.value[0:Nx+1])/dx
    rp = dphi_p[0:-1]/fsign(dphi_p[1:])
    psi_p[1:Nx+1] = 0.5*FL(rp)*(phi.value[2:Nx+2]-phi.value[1:Nx+1])
    psi_p[0] = 0.0  # left boundary will be handled explicitly

    # calculate the upstream to downstream gradient ratios for u<0 (- ratio)
    rm = dphi_p[1:]/fsign(dphi_p[0:-1])
    psi_m[0:Nx] = 0.5*FL(rm)*(phi.value[0:Nx]-phi.value[1:Nx+1])
    psi_m[Nx] = 0.0  # right boundary will be handled explicitly

    # find the velocity direction for the upwind scheme
    ue_min = u.xvalue[1:Nx+1]
    ue_max = u.xvalue[1:Nx+1]
    uw_min = u.xvalue[0:Nx]
    uw_max = u.xvalue[0:Nx]

    ue_min[u_upwind.xvalue[1:Nx+1] > 0.0] = 0.0
    ue_max[u_upwind.xvalue[1:Nx+1] < 0.0] = 0.0
    uw_min[u_upwind.xvalue[0:Nx] > 0.0] = 0.0
    uw_max[u_upwind.xvalue[0:Nx] < 0.0] = 0.0

    # calculate the TVD correction term
    RHS[1:Nx+1] = -(1.0/DXp)*((ue_max*psi_p[1:Nx+1]+ue_min*psi_m[1:Nx+1])-
                                  (uw_max*psi_p[0:Nx]+uw_min*psi_m[0:Nx]))
    return RHS
