# advection terms
import numpy as np
from scipy.sparse import csr_array
from mesh import *
from utilities import *
from cell import *
from face import *

# ------------------- Utility functions ---------------------


def fsign(phi_in, eps1=1e-16):
    return (np.abs(phi_in) >= eps1)*phi_in+eps1*(phi_in == 0.0)+eps1*(np.abs(phi_in) < eps1)*np.sign(phi_in)

# ------------------- 1D functions ---------------------


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


def convectionTvdRHS1D(u: FaceVariable, phi: CellVariable,
                       FL, *args):
    # u is a face variable
    # phi is a cell variable
    # a function to avoid division by zero
    if len(args) > 0:
        u_upwind = args[0]
    else:
        u_upwind = u
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
    RHS[1:Nx+1] = -(1.0/DXp)*((ue_max*psi_p[1:Nx+1]+ue_min*psi_m[1:Nx+1]) -
                              (uw_max*psi_p[0:Nx]+uw_min*psi_m[0:Nx]))
    return RHS

# ------------------- 1D Cylindrical functions ---------------------


def convectionTermCylindrical1D(u: FaceVariable):
    # u is a face variable
    # extract data from the mesh structure
    Nx = u.domain.dims[0]
    G = u.domain.cell_numbers()
    #DX = u.domain.cellsize.x
    DXe = u.domain.cellsize.x[2:]
    DXw = u.domain.cellsize.x[0:-2]
    DXp = u.domain.cellsize.x[1:-1]
    rp = u.domain.cellcenters.x
    rf = u.domain.facecenters.x
    # reassign the east, west for code readability
    ue = rf[1:Nx+1]*u.xvalue[1:Nx+1]/(rp*(DXp+DXe))
    uw = rf[0:Nx]*u.xvalue[0:Nx]/(rp*(DXp+DXw))
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


def convectionUpwindTermCylindrical1D(u: FaceVariable, *args):
    # u is a face variable
    if len(args) > 0:
        u_upwind = args[0]
    else:
        u_upwind = u
    # extract data from the mesh structure
    Nx = u.domain.dims[0]
    G = u.domain.cell_numbers()
    DXp = u.domain.cellsize.x[1:-1]
    rp = u.domain.cellcenters.x
    rf = u.domain.facecenters.x
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
    AE = rf[1:Nx+1]*ue_min/(rp*DXp)
    AW = -rf[0:Nx]*uw_max/(rp*DXp)
    APx = (rf[1:Nx+1]*ue_max-rf[0:Nx]*uw_min)/(rp*DXp)
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


def convectionTvdRHSCylindrical1D(u: FaceVariable, phi: CellVariable,
                                  FL, *args):
    # u is a face variable
    # phi is a cell variable
    # a function to avoid division by zero
    if len(args) > 0:
        u_upwind = args[0]
    else:
        u_upwind = u
    # extract data from the mesh structure
    Nx = u.domain.dims[0]
    DXp = u.domain.cellsize.x[1:-1]
    r = u.domain.cellcenters.x
    rf = u.domain.facecenters.x
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
    RHS[1:Nx+1] = -(1.0/(r*DXp))*(rf[1:Nx+1]*(ue_max*psi_p[1:Nx+1]+ue_min*psi_m[1:Nx+1]) -
                                  rf[0:Nx]*(uw_max*psi_p[0:Nx]+uw_min*psi_m[0:Nx]))
    return RHS

    # ------------------------- 2D functions ------------------------


def convectionTerm2D(u: FaceVariable):
    # u is a face variable
    # extract data from the mesh structure
    Nx, Ny = u.domain.dims
    G = u.domain.cell_numbers()
    DXe = u.domain.cellsize.x[2:][:, np.newaxis]
    DXw = u.domain.cellsize.x[0:-2][:, np.newaxis]
    DXp = u.domain.cellsize.x[1:-1][:, np.newaxis]
    DYn = u.domain.cellsize.y[2:]
    DYs = u.domain.cellsize.y[0:-2]
    DYp = u.domain.cellsize.y[1:-1]
    # define the vectors to store the sparse matrix data
    mn = Nx*Ny
    # reassign the east, west for code readability
    ue = u.xvalue[1:Nx+1, :]/(DXp+DXe)
    uw = u.xvalue[0:Nx, :]/(DXp+DXw)
    vn = u.yvalue[:, 1:Ny+1]/(DYp+DYn)
    vs = u.yvalue[:, 0:Ny]/(DYp+DYs)
    # calculate the coefficients for the internal cells
    AE = ue.ravel()
    AW = -uw.ravel()
    AN = vn.ravel()
    AS = -vs.ravel()
    APx = ((ue*DXe-uw*DXw)/DXp).ravel()
    APy = ((vn*DYn-vs*DYs)/DYp).ravel()

    # build the sparse matrix based on the numbering system
    ii = np.tile(G[1:Nx+1, 1:Ny+1].ravel(), 3)
    jjx = np.hstack([G[0:Nx, 1:Ny+1].ravel(),
                     G[1:Nx+1, 1:Ny+1].ravel(),
                     G[2:Nx+2, 1:Ny+1].ravel()])
    jjy = np.hstack([G[1:Nx+1, 0:Ny].ravel(),
                     G[1:Nx+1, 1:Ny+1].ravel(),
                     G[1:Nx+1, 2:Ny+2].ravel()])
    sx = np.hstack([AW, APx, AE])
    sy = np.hstack([AS, APy, AN])

    # build the sparse matrix
    kx = ky = 3*mn
    Mx = csr_array((sx[0:kx], (ii[0:kx], jjx[0:kx])),
                   shape=((Nx+2)*(Ny+2), (Nx+2)*(Ny+2)))
    My = csr_array((sy[0:kx], (ii[0:ky], jjy[0:ky])),
                   shape=((Nx+2)*(Ny+2), (Nx+2)*(Ny+2)))
    M = Mx + My
    return M, Mx, My


def convectionUpwindTerm2D(u: FaceVariable, u_upwind: FaceVariable):
    # u is a face variable
    # extract data from the mesh structure
    Nx, Ny = u.domain.dims
    G = u.domain.cell_numbers()
    DXp = u.domain.cellsize.x[1:-1][:, np.newaxis]
    DYp = u.domain.cellsize.y[1:-1]
    # define the vectors to store the sparse matrix data
    mn = Nx*Ny
    # find the velocity direction for the upwind scheme
    ue_min = u.xvalue[1:Nx+1, :]
    ue_max = u.xvalue[1:Nx+1, :]
    uw_min = u.xvalue[0:Nx, :]
    uw_max = u.xvalue[0:Nx, :]
    vn_min = u.yvalue[:, 1:Ny+1]
    vn_max = u.yvalue[:, 1:Ny+1]
    vs_min = u.yvalue[:, 0:Ny]
    vs_max = u.yvalue[:, 0:Ny]
    ue_min[u_upwind.xvalue[1:Nx+1, :] > 0.0] = 0.0
    ue_max[u_upwind.xvalue[1:Nx+1, :] < 0.0] = 0.0
    uw_min[u_upwind.xvalue[0:Nx, :] > 0.0] = 0.0
    uw_max[u_upwind.xvalue[0:Nx, :] < 0.0] = 0.0
    vn_min[u_upwind.yvalue[:, 1:Ny+1] > 0.0] = 0.0
    vn_max[u_upwind.yvalue[:, 1:Ny+1] < 0.0] = 0.0
    vs_min[u_upwind.yvalue[:, 0:Ny] > 0.0] = 0.0
    vs_max[u_upwind.yvalue[:, 0:Ny] < 0.0] = 0.0
    # calculate the coefficients for the internal cells, not reshape
    AE = ue_min/DXp
    AW = -uw_max/DXp
    AN = vn_min/DYp
    AS = -vs_max/DYp
    APx = (ue_max-uw_min)/DXp
    APy = (vn_max-vs_min)/DYp
    # Also correct for the boundary cells (not the ghost cells)
    # Left boundary:
    APx[0, :] = APx[0, :]-uw_max[0, :]/(2.0*DXp[0])
    AW[0, :] = AW[0, :]/2.0
    # Right boundary:
    AE[-1, :] = AE[-1, :]/2.0
    APx[-1, :] = APx[-1, :]+ue_min[-1, :]/(2.0*DXp[-1])
    # Bottom boundary:
    APy[:, 0] = APy[:, 0]-vs_max[:, 0]/(2.0*DYp[0])
    AS[:, 0] = AS[:, 0]/2.0
    # Top boundary:
    AN[:, -1] = AN[:, -1]/2.0
    APy[:, -1] = APy[:, -1]+vn_min[:, -1]/(2.0*DYp[-1])
    # build the sparse matrix based on the numbering system
    ii = np.tile(G[1:Nx+1, 1:Ny+1].ravel(), 3)
    jjx = np.hstack([G[0:Nx, 1:Ny+1].ravel(),
                     G[1:Nx+1, 1:Ny+1].ravel(),
                     G[2:Nx+2,  1:Ny+1].ravel()])
    jjy = np.hstack([G[1:Nx+1, 0:Ny].ravel(),
                     G[1:Nx+1, 1:Ny+1].ravel(),
                     G[1:Nx+1, 2:Ny+2].ravel()])
    sx = np.hstack([AW.ravel(), APx.ravel(), AE.ravel()])
    sy = np.hstack([AS.ravel(), APy.ravel(), AN.ravel()])
    # build the sparse matrix
    kx = ky = 3*mn
    Mx = csr_array((sx[0:kx], (ii[0:kx], jjx[0:kx])),
                   shape=((Nx+2)*(Ny+2), (Nx+2)*(Ny+2)))
    My = csr_array((sy[0:kx], (ii[0:ky], jjy[0:ky])),
                   shape=((Nx+2)*(Ny+2), (Nx+2)*(Ny+2)))
    M = Mx + My
    return M, Mx, My

    # ----------------------- User call ----------------------
def convectionTerm(u: FaceVariable) -> csr_array:
    if (type(u.domain) is Mesh1D):
        return convectionTerm1D(u)
    elif (type(u.domain) is MeshCylindrical1D):
        return convectionTermCylindrical1D(u)
    elif (type(u.domain) is Mesh2D):
        return convectionTerm2D(u)
    elif (type(u.domain) is MeshCylindrical2D):
        raise Exception("Not implemented yet. Work in progress")
        # return convectionTermCylindrical2D(u)
    elif (type(u.domain) is MeshRadial2D):
        raise Exception("Not implemented yet. Work in progress")
        # return convectionTermRadial2D(u)
    elif (type(u.domain) is Mesh3D):
        raise Exception("Not implemented yet. Work in progress")
        # return convectionTerm3D(u)
    elif (type(u.domain) is MeshCylindrical3D):
        raise Exception("Not implemented yet. Work in progress")
        # return convectionTermCylindrical3D(u)
    else:
        raise Exception("convectionTerm is not defined for this Mesh type.")

def convectionUpwindTerm(u: FaceVariable) -> csr_array:
    if (type(u.domain) is Mesh1D):
        return convectionUpwindTerm1D(u)
    elif (type(u.domain) is MeshCylindrical1D):
        return convectionUpwindTermCylindrical1D(u)
    elif (type(u.domain) is Mesh2D):
        return convectionUpwindTerm2D(u)
    elif (type(u.domain) is MeshCylindrical2D):
        raise Exception("Not implemented yet. Work in progress")
        # return convectionUpwindTermCylindrical2D(u)
    elif (type(u.domain) is MeshRadial2D):
        raise Exception("Not implemented yet. Work in progress")
        # return convectionUpwindTermRadial2D(u)
    elif (type(u.domain) is Mesh3D):
        raise Exception("Not implemented yet. Work in progress")
        # return convectionUpwindTerm3D(u)
    elif (type(u.domain) is MeshCylindrical3D):
        raise Exception("Not implemented yet. Work in progress")
        # return convectionUpwindTermCylindrical3D(u)
    else:
        raise Exception("convectionUpwindTerm is not defined for this Mesh type.")

def convectionTvdRHSTerm(u: FaceVariable) -> np.ndarray:
    if (type(u.domain) is Mesh1D):
        return convectionTvdRHS1D(u)
    elif (type(u.domain) is MeshCylindrical1D):
        return convectionTvdRHSCylindrical1D(u)
    elif (type(u.domain) is Mesh2D):
        raise Exception("Not implemented yet. Work in progress")
        # return convectionTvdRHS2D(u)
    elif (type(u.domain) is MeshCylindrical2D):
        raise Exception("Not implemented yet. Work in progress")
        # return convectionTvdRHSCylindrical2D(u)
    elif (type(u.domain) is MeshRadial2D):
        raise Exception("Not implemented yet. Work in progress")
        # return convectionTvdRHSRadial2D(u)
    elif (type(u.domain) is Mesh3D):
        raise Exception("Not implemented yet. Work in progress")
        # return convectionTvdRHS3D(u)
    elif (type(u.domain) is MeshCylindrical3D):
        raise Exception("Not implemented yet. Work in progress")
        # return convectionTvdRHSCylindrical3D(u)
    else:
        raise Exception("convectionTvdRHSTerm is not defined for this Mesh type.")