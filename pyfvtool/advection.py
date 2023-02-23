# advection terms
import numpy as np
from scipy.sparse import csr_array
from .mesh import *
from .utilities import *
from .cell import *
from .face import *

# ------------------- Utility functions ---------------------


def _upwind_min_max(u: FaceVariable, u_upwind: FaceVariable):
    if issubclass(type(u.domain), Mesh1D):
        ux_min = np.copy(u.xvalue)
        ux_max = np.copy(u.xvalue)
        ux_min[u_upwind.xvalue > 0.0] = 0.0
        ux_max[u_upwind.xvalue < 0.0] = 0.0
        return ux_min, ux_max
    elif issubclass(type(u.domain), Mesh2D):
        ux_min = np.copy(u.xvalue)
        ux_max = np.copy(u.xvalue)
        uy_min = np.copy(u.yvalue)
        uy_max = np.copy(u.yvalue)
        ux_min[u_upwind.xvalue > 0.0] = 0.0
        ux_max[u_upwind.xvalue < 0.0] = 0.0
        uy_min[u_upwind.yvalue > 0.0] = 0.0
        uy_max[u_upwind.yvalue < 0.0] = 0.0
        return ux_min, ux_max, uy_min, uy_max
    elif issubclass(type(u.domain), Mesh3D):
        ux_min = np.copy(u.xvalue)
        ux_max = np.copy(u.xvalue)
        uy_min = np.copy(u.yvalue)
        uy_max = np.copy(u.yvalue)
        uz_min = np.copy(u.zvalue)
        uz_max = np.copy(u.zvalue)
        ux_min[u_upwind.xvalue > 0.0] = 0.0
        ux_max[u_upwind.xvalue < 0.0] = 0.0
        uy_min[u_upwind.yvalue > 0.0] = 0.0
        uy_max[u_upwind.yvalue < 0.0] = 0.0
        uz_min[u_upwind.zvalue > 0.0] = 0.0
        uz_max[u_upwind.zvalue < 0.0] = 0.0
        return ux_min, ux_max, uy_min, uy_max, uz_min, uz_max

def _fsign(phi_in, eps1=1e-16):
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
    # build the sparse matrix based on the numbering system
    iix = np.tile(G[1:Nx+1].ravel(), 3)
    jjx = np.hstack([G[0:Nx], G[1:Nx+1], G[2:Nx+2]])
    sx = np.hstack([-uw, (ue*DXe-uw*DXw)/DXp, ue])
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
    # find the velocity direction for the upwind scheme
    ux_min, ux_max = _upwind_min_max(u, u_upwind)
    ue_min = ux_min[1:Nx+1]
    ue_max = ux_max[1:Nx+1]
    uw_min = ux_min[0:Nx]
    uw_max = ux_max[0:Nx]
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
    rp = dphi_p[0:-1]/_fsign(dphi_p[1:])
    psi_p[1:Nx+1] = 0.5*FL(rp)*(phi.value[2:Nx+2]-phi.value[1:Nx+1])
    psi_p[0] = 0.0  # left boundary will be handled explicitly
    # calculate the upstream to downstream gradient ratios for u<0 (- ratio)
    rm = dphi_p[1:]/_fsign(dphi_p[0:-1])
    psi_m[0:Nx] = 0.5*FL(rm)*(phi.value[0:Nx]-phi.value[1:Nx+1])
    psi_m[Nx] = 0.0  # right boundary will be handled explicitly
    # find the velocity direction for the upwind scheme
    ux_min, ux_max = _upwind_min_max(u, u_upwind)
    ue_min = ux_min[1:Nx+1]
    ue_max = ux_max[1:Nx+1]
    uw_min = ux_min[0:Nx]
    uw_max = ux_max[0:Nx]
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
    # find the velocity direction for the upwind scheme
    ux_min, ux_max = _upwind_min_max(u, u_upwind)
    ue_min = ux_min[1:Nx+1]
    ue_max = ux_max[1:Nx+1]
    uw_min = ux_min[0:Nx]
    uw_max = ux_max[0:Nx]
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
    rp = dphi_p[0:-1]/_fsign(dphi_p[1:])
    psi_p[1:Nx+1] = 0.5*FL(rp)*(phi.value[2:Nx+2]-phi.value[1:Nx+1])
    psi_p[0] = 0.0  # left boundary will be handled explicitly
    # calculate the upstream to downstream gradient ratios for u<0 (- ratio)
    rm = dphi_p[1:]/_fsign(dphi_p[0:-1])
    psi_m[0:Nx] = 0.5*FL(rm)*(phi.value[0:Nx]-phi.value[1:Nx+1])
    psi_m[Nx] = 0.0  # right boundary will be handled explicitly
    # find the velocity direction for the upwind scheme
    ux_min, ux_max = _upwind_min_max(u, u_upwind)
    ue_min = ux_min[1:Nx+1]
    ue_max = ux_max[1:Nx+1]
    uw_min = ux_min[0:Nx]
    uw_max = ux_max[0:Nx]
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


def convectionUpwindTerm2D(u: FaceVariable, *args):
    # u is a face variable
    # extract data from the mesh structure
    if len(args) > 0:
        u_upwind = args[0]
    else:
        u_upwind = u
    Nx, Ny = u.domain.dims
    G = u.domain.cell_numbers()
    DXp = u.domain.cellsize.x[1:-1][:, np.newaxis]
    DYp = u.domain.cellsize.y[1:-1]
    # define the vectors to store the sparse matrix data
    mn = Nx*Ny
    # find the velocity direction for the upwind scheme
    ux_min, ux_max, uy_min, uy_max = _upwind_min_max(u, u_upwind)
    ue_min, ue_max = ux_min[1:Nx+1, :], ux_max[1:Nx+1, :]
    uw_min, uw_max = ux_min[0:Nx, :], ux_max[0:Nx, :]
    vn_min, vn_max = uy_min[:, 1:Ny+1], uy_max[:, 1:Ny+1]
    vs_min, vs_max = uy_min[:, 0:Ny], uy_max[:, 0:Ny]
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


def convectionTvdRHS2D(u: FaceVariable, phi: CellVariable, FL, *args):
    # u is a face variable
    # phi is a cell variable
    if len(args) > 0:
        u_upwind = args[0]
    else:
        u_upwind = u
    # a function to avoid division by zero
    # extract data from the mesh structure
    Nx, Ny = u.domain.dims
    G = u.domain.cell_numbers()
    DXp = u.domain.cellsize.x[1:-1][:, np.newaxis]
    DYp = u.domain.cellsize.y[1:-1][np.newaxis, :]
    dx = 0.5*(u.domain.cellsize.x[0:-1]+u.domain.cellsize.x[1:])[:, np.newaxis]
    dy = 0.5*(u.domain.cellsize.y[0:-1]+u.domain.cellsize.y[1:])[np.newaxis, :]
    psiX_p = np.zeros((Nx+1, Ny))
    psiX_m = np.zeros((Nx+1, Ny))
    psiY_p = np.zeros((Nx, Ny+1))
    psiY_m = np.zeros((Nx, Ny+1))
    # calculate the upstream to downstream gradient ratios for u>0 (+ ratio)
    # x direction
    dphiX_p = (phi.value[1:Nx+2, 1:Ny+1]-phi.value[0:Nx+1, 1:Ny+1])/dx
    rX_p = dphiX_p[0:-1, :]/_fsign(dphiX_p[1:, :])
    psiX_p[1:Nx+1, :] = 0.5*FL(rX_p)*(phi.value[2:Nx+2, 1:Ny+1] -
                                      phi.value[1:Nx+1, 1:Ny+1])
    psiX_p[0, :] = 0.0  # left boundary will be handled in the main matrix
    # y direction
    dphiY_p = (phi.value[1:Nx+1, 1:Ny+2]-phi.value[1:Nx+1, 0:Ny+1])/dy
    rY_p = dphiY_p[:, 0:-1]/_fsign(dphiY_p[:, 1:])
    psiY_p[:, 1:Ny+1] = 0.5*FL(rY_p)*(phi.value[1:Nx+1, 2:Ny+2] -
                                      phi.value[1:Nx+1, 1:Ny+1])
    psiY_p[:, 0] = 0.0  # Bottom boundary will be handled in the main matrix
    # calculate the upstream to downstream gradient ratios for u<0 (- ratio)
    # x direction
    rX_m = dphiX_p[1:, :]/_fsign(dphiX_p[0:-1, :])
    psiX_m[0:Nx, :] = 0.5*FL(rX_m)*(phi.value[0:Nx, 1:Ny+1] -
                                    phi.value[1:Nx+1, 1:Ny+1])
    psiX_m[-1, :] = 0.0  # right boundary
    # y direction
    rY_m = dphiY_p[:, 1:]/_fsign(dphiY_p[:, 0:-1])
    psiY_m[:, 0:Ny] = 0.5*FL(rY_m)*(phi.value[1:Nx+1, 0:Ny] -
                                    phi.value[1:Nx+1, 1:Ny+1])
    psiY_m[:, -1] = 0.0  # top boundary will be handled in the main matrix
    # find the velocity direction for the upwind scheme
    ux_min, ux_max, uy_min, uy_max = _upwind_min_max(u, u_upwind)
    ue_min, ue_max = ux_min[1:Nx+1, :], ux_max[1:Nx+1, :]
    uw_min, uw_max = ux_min[0:Nx, :], ux_max[0:Nx, :]
    vn_min, vn_max = uy_min[:, 1:Ny+1], uy_max[:, 1:Ny+1]
    vs_min, vs_max = uy_min[:, 0:Ny], uy_max[:, 0:Ny]

    # calculate the TVD correction term
    div_x = -(1.0/DXp)*((ue_max*psiX_p[1:Nx+1, :]+ue_min*psiX_m[1:Nx+1, :]) -
                        (uw_max*psiX_p[0:Nx, :]+uw_min*psiX_m[0:Nx, :]))
    div_y = -(1.0/DYp)*((vn_max*psiY_p[:, 1:Ny+1]+vn_min*psiY_m[:, 1:Ny+1]) -
                        (vs_max*psiY_p[:, 0:Ny]+vs_min*psiY_m[:, 0:Ny]))

    # define the RHS Vector
    RHS = np.zeros((Nx+2)*(Ny+2))
    RHSx = np.zeros((Nx+2)*(Ny+2))
    RHSy = np.zeros((Nx+2)*(Ny+2))

    # assign the values of the RHS vector
    rowx_index = rowy_index = G[1:Nx+1, 1:Ny+1].ravel()  # main diagonal x, y
    RHS[rowx_index] = (div_x+div_y).ravel()
    RHSx[rowx_index] = div_x.ravel()
    RHSy[rowy_index] = div_y.ravel()

    return RHS, RHSx, RHSy

    # ----------------------- User call ----------------------


def convectionTermCylindrical2D(u: FaceVariable):
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
    rp = u.domain.cellcenters.x[:, np.newaxis]
    rf = u.domain.facecenters.x[:, np.newaxis]
    # define the vectors to store the sparse matrix data
    mn = Nx*Ny
    # reassign the east, west for code readability
    ue = rf[1:Nx+1, :]*u.xvalue[1:Nx+1, :]/(rp*(DXp+DXe))
    uw = rf[0:Nx, :]*u.xvalue[0:Nx, :]/(rp*(DXp+DXe))
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


def convectionUpwindTermCylindrical2D(u: FaceVariable, *args):
    # u is a face variable
    # extract data from the mesh structure
    if len(args) > 0:
        u_upwind = args[0]
    else:
        u_upwind = u
    Nx, Ny = u.domain.dims
    G = u.domain.cell_numbers()
    DXp = u.domain.cellsize.x[1:-1][:, np.newaxis]
    DYp = u.domain.cellsize.y[1:-1]
    rp = u.domain.cellcenters.x[:, np.newaxis]
    rf = u.domain.facecenters.x[:, np.newaxis]
    re = rf[1:Nx+1, :]
    rw = rf[0:Nx, :]
    # define the vectors to store the sparse matrix data
    mn = Nx*Ny
    # find the velocity direction for the upwind scheme
    ux_min, ux_max, uy_min, uy_max = _upwind_min_max(u, u_upwind)
    ue_min, ue_max = ux_min[1:Nx+1, :], ux_max[1:Nx+1, :]
    uw_min, uw_max = ux_min[0:Nx, :], ux_max[0:Nx, :]
    vn_min, vn_max = uy_min[:, 1:Ny+1], uy_max[:, 1:Ny+1]
    vs_min, vs_max = uy_min[:, 0:Ny], uy_max[:, 0:Ny]
    # calculate the coefficients for the internal cells, not reshape
    AE = re*ue_min/(DXp*rp)
    AW = -rw*uw_max/(DXp*rp)
    AN = vn_min/DYp
    AS = -vs_max/DYp
    APx = (re*ue_max-rw*uw_min)/(DXp*rp)
    APy = (vn_max-vs_min)/DYp
    # Also correct for the boundary cells (not the ghost cells)
    # Left boundary:
    APx[0, :] = APx[0, :]-rw[0, :]*uw_max[0, :]/(2.0*DXp[0]*rp[0, :])
    AW[0, :] = AW[0, :]/2.0
    # Right boundary:
    AE[-1, :] = AE[-1, :]/2.0
    APx[-1, :] = APx[-1, :]+re[-1, :]*ue_min[-1, :]/(2.0*DXp[-1]*rp[-1, :])
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


def convectionTvdRHSCylindrical2D(u: FaceVariable, phi: CellVariable, FL, *args):
    # u is a face variable
    # phi is a cell variable
    if len(args) > 0:
        u_upwind = args[0]
    else:
        u_upwind = u
    # a function to avoid division by zero
    # extract data from the mesh structure
    Nx, Ny = u.domain.dims
    G = u.domain.cell_numbers()
    DXp = u.domain.cellsize.x[1:-1][:, np.newaxis]
    DYp = u.domain.cellsize.y[1:-1][np.newaxis, :]
    rp = u.domain.cellcenters.x[:, np.newaxis]
    rf = u.domain.facecenters.x[:, np.newaxis]
    re = rf[1:Nx+1, :]
    rw = rf[0:Nx, :]
    dx = 0.5*(u.domain.cellsize.x[0:-1]+u.domain.cellsize.x[1:])[:, np.newaxis]
    dy = 0.5*(u.domain.cellsize.y[0:-1]+u.domain.cellsize.y[1:])[np.newaxis, :]
    psiX_p = np.zeros((Nx+1, Ny))
    psiX_m = np.zeros((Nx+1, Ny))
    psiY_p = np.zeros((Nx, Ny+1))
    psiY_m = np.zeros((Nx, Ny+1))
    # calculate the upstream to downstream gradient ratios for u>0 (+ ratio)
    # x direction
    dphiX_p = (phi.value[1:Nx+2, 1:Ny+1]-phi.value[0:Nx+1, 1:Ny+1])/dx
    rX_p = dphiX_p[0:-1, :]/_fsign(dphiX_p[1:, :])
    psiX_p[1:Nx+1, :] = 0.5*FL(rX_p)*(phi.value[2:Nx+2, 1:Ny+1] -
                                      phi.value[1:Nx+1, 1:Ny+1])
    psiX_p[0, :] = 0.0  # left boundary will be handled in the main matrix
    # y direction
    dphiY_p = (phi.value[1:Nx+1, 1:Ny+2]-phi.value[1:Nx+1, 0:Ny+1])/dy
    rY_p = dphiY_p[:, 0:-1]/_fsign(dphiY_p[:, 1:])
    psiY_p[:, 1:Ny+1] = 0.5*FL(rY_p)*(phi.value[1:Nx+1, 2:Ny+2] -
                                      phi.value[1:Nx+1, 1:Ny+1])
    psiY_p[:, 0] = 0.0  # Bottom boundary will be handled in the main matrix
    # calculate the upstream to downstream gradient ratios for u<0 (- ratio)
    # x direction
    rX_m = dphiX_p[1:, :]/_fsign(dphiX_p[0:-1, :])
    psiX_m[0:Nx, :] = 0.5*FL(rX_m)*(phi.value[0:Nx, 1:Ny+1] -
                                    phi.value[1:Nx+1, 1:Ny+1])
    psiX_m[-1, :] = 0.0  # right boundary
    # y direction
    rY_m = dphiY_p[:, 1:]/_fsign(dphiY_p[:, 0:-1])
    psiY_m[:, 0:Ny] = 0.5*FL(rY_m)*(phi.value[1:Nx+1, 0:Ny] -
                                    phi.value[1:Nx+1, 1:Ny+1])
    psiY_m[:, -1] = 0.0  # top boundary will be handled in the main matrix
    # find the velocity direction for the upwind scheme
    ux_min, ux_max, uy_min, uy_max = _upwind_min_max(u, u_upwind)
    ue_min, ue_max = ux_min[1:Nx+1, :], ux_max[1:Nx+1, :]
    uw_min, uw_max = ux_min[0:Nx, :], ux_max[0:Nx, :]
    vn_min, vn_max = uy_min[:, 1:Ny+1], uy_max[:, 1:Ny+1]
    vs_min, vs_max = uy_min[:, 0:Ny], uy_max[:, 0:Ny]
    # calculate the TVD correction term
    div_x = -(1.0/(rp*DXp))*(re*(ue_max*psiX_p[1:Nx+1, :]+ue_min*psiX_m[1:Nx+1, :]) -
                             rw*(uw_max*psiX_p[0:Nx, :]+uw_min*psiX_m[0:Nx, :]))
    div_y = -(1.0/DYp)*((vn_max*psiY_p[:, 1:Ny+1]+vn_min*psiY_m[:, 1:Ny+1]) -
                        (vs_max*psiY_p[:, 0:Ny]+vs_min*psiY_m[:, 0:Ny]))
    # define the RHS Vector
    RHS = np.zeros((Nx+2)*(Ny+2))
    RHSx = np.zeros((Nx+2)*(Ny+2))
    RHSy = np.zeros((Nx+2)*(Ny+2))
    # assign the values of the RHS vector
    rowx_index = rowy_index = G[1:Nx+1, 1:Ny+1].ravel()  # main diagonal x, y
    RHS[rowx_index] = (div_x+div_y).ravel()
    RHSx[rowx_index] = div_x.ravel()
    RHSy[rowy_index] = div_y.ravel()
    return RHS, RHSx, RHSy


def convectionTermRadial2D(u: FaceVariable):
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
    rp = u.domain.cellcenters.x[:, np.newaxis]
    rf = u.domain.facecenters.x[:, np.newaxis]
    # define the vectors to store the sparse matrix data
    mn = Nx*Ny
    # reassign the east, west for code readability
    ue = rf[1:Nx+1, :]*u.xvalue[1:Nx+1, :]/(rp*(DXp+DXe))
    uw = rf[0:Nx, :]*u.xvalue[0:Nx, :]/(rp*(DXp+DXe))
    vn = u.yvalue[:, 1:Ny+1]/(rp*(DYp+DYn))
    vs = u.yvalue[:, 0:Ny]/(rp*(DYp+DYs))
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


def convectionUpwindTermRadial2D(u: FaceVariable, *args):
    # u is a face variable
    # extract data from the mesh structure
    if len(args) > 0:
        u_upwind = args[0]
    else:
        u_upwind = u
    Nx, Ny = u.domain.dims
    G = u.domain.cell_numbers()
    DXp = u.domain.cellsize.x[1:-1][:, np.newaxis]
    DYp = u.domain.cellsize.y[1:-1]
    rp = u.domain.cellcenters.x[:, np.newaxis]
    rf = u.domain.facecenters.x[:, np.newaxis]
    re = rf[1:Nx+1, :]
    rw = rf[0:Nx, :]
    # define the vectors to store the sparse matrix data
    mn = Nx*Ny
    # find the velocity direction for the upwind scheme
    ux_min, ux_max, uy_min, uy_max = _upwind_min_max(u, u_upwind)
    ue_min, ue_max = ux_min[1:Nx+1, :], ux_max[1:Nx+1, :]
    uw_min, uw_max = ux_min[0:Nx, :], ux_max[0:Nx, :]
    vn_min, vn_max = uy_min[:, 1:Ny+1], uy_max[:, 1:Ny+1]
    vs_min, vs_max = uy_min[:, 0:Ny], uy_max[:, 0:Ny]
    # calculate the coefficients for the internal cells, not reshape
    AE = re*ue_min/(DXp*rp)
    AW = -rw*uw_max/(DXp*rp)
    AN = vn_min/(rp*DYp)
    AS = -vs_max/(rp*DYp)
    APx = (re*ue_max-rw*uw_min)/(DXp*rp)
    APy = (vn_max-vs_min)/(DYp*rp)
    # Also correct for the boundary cells (not the ghost cells)
    # Left boundary:
    APx[0, :] = APx[0, :]-rw[0, :]*uw_max[0, :]/(2.0*DXp[0]*rp[0, :])
    AW[0, :] = AW[0, :]/2.0
    # Right boundary:
    AE[-1, :] = AE[-1, :]/2.0
    APx[-1, :] = APx[-1, :]+re[-1, :]*ue_min[-1, :]/(2.0*DXp[-1]*rp[-1, :])
    # Bottom boundary:
    APy[:, 0] = APy[:, 0]-vs_max[:, 0]/(2.0*DYp[0]*rp[:, 0])
    AS[:, 0] = AS[:, 0]/2.0
    # Top boundary:
    AN[:, -1] = AN[:, -1]/2.0
    APy[:, -1] = APy[:, -1]+vn_min[:, -1]/(2.0*DYp[-1]*rp[:, -1])
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


def convectionTvdRHSRadial2D(u: FaceVariable, phi: CellVariable, FL, *args):
    # u is a face variable
    # phi is a cell variable
    if len(args) > 0:
        u_upwind = args[0]
    else:
        u_upwind = u
    # a function to avoid division by zero
    # extract data from the mesh structure
    Nx, Ny = u.domain.dims
    G = u.domain.cell_numbers()
    DXp = u.domain.cellsize.x[1:-1][:, np.newaxis]
    DYp = u.domain.cellsize.y[1:-1][np.newaxis, :]
    rp = u.domain.cellcenters.x[:, np.newaxis]
    rf = u.domain.facecenters.x[:, np.newaxis]
    re = rf[1:Nx+1, :]
    rw = rf[0:Nx, :]
    dx = 0.5*(u.domain.cellsize.x[0:-1]+u.domain.cellsize.x[1:])[:, np.newaxis]
    dy = 0.5*(u.domain.cellsize.y[0:-1]+u.domain.cellsize.y[1:])[np.newaxis, :]
    psiX_p = np.zeros((Nx+1, Ny))
    psiX_m = np.zeros((Nx+1, Ny))
    psiY_p = np.zeros((Nx, Ny+1))
    psiY_m = np.zeros((Nx, Ny+1))
    # calculate the upstream to downstream gradient ratios for u>0 (+ ratio)
    # x direction
    dphiX_p = (phi.value[1:Nx+2, 1:Ny+1]-phi.value[0:Nx+1, 1:Ny+1])/dx
    rX_p = dphiX_p[0:-1, :]/_fsign(dphiX_p[1:, :])
    psiX_p[1:Nx+1, :] = 0.5*FL(rX_p)*(phi.value[2:Nx+2, 1:Ny+1] -
                                      phi.value[1:Nx+1, 1:Ny+1])
    psiX_p[0, :] = 0.0  # left boundary will be handled in the main matrix
    # y direction
    dphiY_p = (phi.value[1:Nx+1, 1:Ny+2]-phi.value[1:Nx+1, 0:Ny+1])/dy
    rY_p = dphiY_p[:, 0:-1]/_fsign(dphiY_p[:, 1:])
    psiY_p[:, 1:Ny+1] = 0.5*FL(rY_p)*(phi.value[1:Nx+1, 2:Ny+2] -
                                      phi.value[1:Nx+1, 1:Ny+1])
    psiY_p[:, 0] = 0.0  # Bottom boundary will be handled in the main matrix
    # calculate the upstream to downstream gradient ratios for u<0 (- ratio)
    # x direction
    rX_m = dphiX_p[1:, :]/_fsign(dphiX_p[0:-1, :])
    psiX_m[0:Nx, :] = 0.5*FL(rX_m)*(phi.value[0:Nx, 1:Ny+1] -
                                    phi.value[1:Nx+1, 1:Ny+1])
    psiX_m[-1, :] = 0.0  # right boundary
    # y direction
    rY_m = dphiY_p[:, 1:]/_fsign(dphiY_p[:, 0:-1])
    psiY_m[:, 0:Ny] = 0.5*FL(rY_m)*(phi.value[1:Nx+1, 0:Ny] -
                                    phi.value[1:Nx+1, 1:Ny+1])
    psiY_m[:, -1] = 0.0  # top boundary will be handled in the main matrix
    # find the velocity direction for the upwind scheme
    ux_min, ux_max, uy_min, uy_max = _upwind_min_max(u, u_upwind)
    ue_min, ue_max = ux_min[1:Nx+1, :], ux_max[1:Nx+1, :]
    uw_min, uw_max = ux_min[0:Nx, :], ux_max[0:Nx, :]
    vn_min, vn_max = uy_min[:, 1:Ny+1], uy_max[:, 1:Ny+1]
    vs_min, vs_max = uy_min[:, 0:Ny], uy_max[:, 0:Ny]
    # calculate the TVD correction term
    div_x = -(1.0/(rp*DXp))*(re*(ue_max*psiX_p[1:Nx+1, :]+ue_min*psiX_m[1:Nx+1, :]) -
                             rw*(uw_max*psiX_p[0:Nx, :]+uw_min*psiX_m[0:Nx, :]))
    div_y = -(1.0/(DYp*rp))*((vn_max*psiY_p[:, 1:Ny+1]+vn_min*psiY_m[:, 1:Ny+1]) -
                             (vs_max*psiY_p[:, 0:Ny]+vs_min*psiY_m[:, 0:Ny]))
    # define the RHS Vector
    RHS = np.zeros((Nx+2)*(Ny+2))
    RHSx = np.zeros((Nx+2)*(Ny+2))
    RHSy = np.zeros((Nx+2)*(Ny+2))
    # assign the values of the RHS vector
    rowx_index = rowy_index = G[1:Nx+1, 1:Ny+1].ravel()  # main diagonal x, y
    RHS[rowx_index] = (div_x+div_y).ravel()
    RHSx[rowx_index] = div_x.ravel()
    RHSy[rowy_index] = div_y.ravel()
    return RHS, RHSx, RHSy


def convectionTerm3D(u: FaceVariable):
    # u is a face variable
    # extract data from the mesh structure
    Nx, Ny, Nz = u.domain.dims
    G = u.domain.cell_numbers()
    DXe = u.domain.cellsize.x[2:][:, np.newaxis, np.newaxis]
    DXw = u.domain.cellsize.x[0:-2][:, np.newaxis, np.newaxis]
    DXp = u.domain.cellsize.x[1:-1][:, np.newaxis, np.newaxis]
    DYn = u.domain.cellsize.y[2:][np.newaxis, :, np.newaxis]
    DYs = u.domain.cellsize.y[0:-2][np.newaxis, :, np.newaxis]
    DYp = u.domain.cellsize.y[1:-1][np.newaxis, :, np.newaxis]
    DZf = u.domain.cellsize.z[2:][np.newaxis, np.newaxis, :]
    DZb = u.domain.cellsize.z[0:-2][np.newaxis, np.newaxis, :]
    DZp = u.domain.cellsize.z[1:-1][np.newaxis, np.newaxis, :]
    # define the vectors to stores the sparse matrix data
    mn = Nx*Ny*Nz
    # reassign the east, west, north, and south velocity vectors for the
    # code readability
    ue = u.xvalue[1:Nx+1, :, :]/(DXp+DXe)
    uw = u.xvalue[0:Nx, :, :]/(DXp+DXw)
    vn = u.yvalue[:, 1:Ny+1, :]/(DYp+DYn)
    vs = u.yvalue[:, 0:Ny, :]/(DYp+DYs)
    wf = u.zvalue[:, :, 1:Nz+1]/(DZp+DZf)
    wb = u.zvalue[:, :, 0:Nz]/(DZp+DZb)
    # calculate the coefficients for the internal cells
    AE = ue.ravel()
    AW = -uw.ravel()
    AN = vn.ravel()
    AS = -vs.ravel()
    AF = wf.ravel()
    AB = -wb.ravel()
    APx = ((ue*DXe-uw*DXw)/DXp).ravel()
    APy = ((vn*DYn-vs*DYs)/DYp).ravel()
    APz = ((wf*DZf-wb*DZb)/DZp).ravel()

    # build the sparse matrix based on the numbering system
    ii = np.tile(G[1:Nx+1, 1:Ny+1, 1:Nz+1].ravel(), 3)
    jjx = np.hstack([G[0:Nx, 1:Ny+1, 1:Nz+1].ravel(),
                     G[1:Nx+1, 1:Ny+1, 1:Nz+1].ravel(),
                     G[2:Nx+2, 1:Ny+1, 1:Nz+1].ravel()])
    jjy = np.hstack([G[1:Nx+1, 0:Ny, 1:Nz+1].ravel(),
                     G[1:Nx+1, 1:Ny+1, 1:Nz+1].ravel(),
                     G[1:Nx+1, 2:Ny+2, 1:Nz+1].ravel()])
    jjz = np.hstack([G[1:Nx+1, 1:Ny+1, 0:Nz].ravel(),
                     G[1:Nx+1, 1:Ny+1, 1:Nz+1].ravel(),
                     G[1:Nx+1, 1:Ny+1, 2:Nz+2].ravel()])
    sx = np.hstack([AW, APx, AE])
    sy = np.hstack([AS, APy, AN])
    sz = np.hstack([AB, APz, AF])

    # build the sparse matrix
    kx = ky = kz = 3*mn
    Mx = csr_array((sx[0:kx], (ii[0:kx], jjx[0:kx])),
                   shape=((Nx+2)*(Ny+2)*(Nz+2), (Nx+2)*(Ny+2)*(Nz+2)))
    My = csr_array((sy[0:kx], (ii[0:ky], jjy[0:ky])),
                   shape=((Nx+2)*(Ny+2)*(Nz+2), (Nx+2)*(Ny+2)*(Nz+2)))
    Mz = csr_array((sz[0:kz], (ii[0:kz], jjz[0:kz])),
                   shape=((Nx+2)*(Ny+2)*(Nz+2), (Nx+2)*(Ny+2)*(Nz+2)))
    M = Mx + My + Mz
    return M, Mx, My, Mz

def convectionUpwindTerm3D(u: FaceVariable):
    # u is a face variable
    # extract data from the mesh structure
    Nx, Ny, Nz = u.domain.dims
    G = u.domain.cell_numbers()
    DXp = u.domain.cellsize.x[1:-1][:, np.newaxis, np.newaxis]
    DYp = u.domain.cellsize.y[1:-1][np.newaxis, :, np.newaxis]
    DZp = u.domain.cellsize.z[1:-1][np.newaxis, np.newaxis, :]

    # define the vectors to stores the sparse matrix data
    mn = Nx*Ny*Nz

    # find the velocity direction for the upwind scheme
    ux_min, ux_max, uy_min, uy_max, uz_min, uz_max = _upwind_min_max(u, u_upwind)
    ue_min, ue_max = ux_min[1:Nx+1, :, :], ux_max[1:Nx+1, :, :]
    uw_min, uw_max = ux_min[0:Nx, :, :], ux_max[0:Nx, :, :]
    vn_min, vn_max = uy_min[:, 1:Ny+1, :], uy_max[:, 1:Ny+1, :]
    vs_min, vs_max = uy_min[:, 0:Ny, :], uy_max[:, 0:Ny, :]
    wf_min, wf_max = uz_min[:, :, 1:Nz+1], uz_max[:, :, 1:Nz+1]
    wb_min, wb_max = uz_min[:, :, 0:Nz], uz_max[:, :, 0:Nz]

    # calculate the coefficients for the internal cells
    AE = ue_min/DXp
    AW = -uw_max/DXp
    AN = vn_min/DYp
    AS = -vs_max/DYp
    AF = wf_min/DZp
    AB = -wb_max/DZp
    APx = (ue_max-uw_min)/DXp
    APy = (vn_max-vs_min)/DYp
    APz = (wf_max-wb_min)/DZp

    # Also correct for the boundary cells (not the ghost cells)
    # Left boundary:
    APx[0,:,:] = APx[0,:,:]-uw_max[0,:,:]/(2.0*DXp[0,:,:])
    AW[0,:,:] = AW[0,:,:]/2.0
    # Right boundary:
    AE[-1,:,:] = AE[-1,:,:]/2.0
    APx[-1,:,:] = APx[-1,:,:]+ue_min[-1,:,:]/(2.0*DXp[-1,:,:])
    # Bottom boundary:
    APy[:,0,:] = APy[:,0,:]-vs_max[:,0,:]/(2.0*DYp[:,0,:])
    AS[:,0,:] = AS[:,0,:]/2.0
    # Top boundary:
    AN[:,-1,:] = AN[:,-1,:]/2.0
    APy[:,-1,:] = APy[:,-1,:]+vn_min[:,-1,:]/(2.0*DYp[:,-1,:])
    # Back boundary:
    APz[:,:,1] = APz[:,:,1]-wb_max[:,:,1]/(2.0*DZp[1,:,:])
    AB[:,:,1] = AB[:,:,1]/2.0
    # Front boundary:
    AF[:,:,-1] = AF[:,:,-1]/2.0
    APz[:,:,-1] = APz[:,:,-1]+wf_min[:,:,-1]/(2.0*DZp[-1,:,:])

    AE = AE.ravel()
    AW = AW.ravel()
    AN = AN.ravel()
    AS = AS.ravel()
    AF = AF.ravel()
    AB = AB.ravel()
    APx = APx.ravel()
    APy = APy.ravel()
    APz = APz.ravel()

    # build the sparse matrix based on the numbering system
    ii = np.tile(G[1:Nx+1, 1:Ny+1, 1:Nz+1].ravel(), 3)
    jjx = np.hstack([G[1:Nx,2:Ny+1,2:Nz+1].ravel(),
                    G[2:Nx+1,2:Ny+1,2:Nz+1].ravel(), 
                    G[3:Nx+2,2:Ny+1,2:Nz+1].ravel()])
    jjy = np.hstack([G[2:Nx+1,1:Ny,2:Nz+1].ravel(), 
                     G[2:Nx+1,2:Ny+1,2:Nz+1].ravel(),
                     G[2:Nx+1,3:Ny+2,2:Nz+1].ravel()])
    jjz = np.hstack([G[2:Nx+1,2:Ny+1,1:Nz].ravel(), 
                     G[2:Nx+1,2:Ny+1,2:Nz+1].ravel(),
                     G[2:Nx+1,2:Ny+1,3:Nz+2]]
    sx = np.hstack([AW.ravel(), APx.ravel(), AE.ravel()])
    sy = np.hstack([AS.ravel(), APy.ravel(), AN.ravel()])
    sz = np.hstack([AB.ravel(), APz.ravel(), AF.ravel()])

    # build the sparse matrix
    kx = 3*mnx
    ky = 3*mny
    kz = 3*mnz
    Mx = sparse(iix[1:kx], jjx[1:kx], sx[1:kx], (Nx+2)*(Ny+2)*(Nz+2), (Nx+2)*(Ny+2)*(Nz+2))
    My = sparse(iiy[1:ky], jjy[1:ky], sy[1:ky], (Nx+2)*(Ny+2)*(Nz+2), (Nx+2)*(Ny+2)*(Nz+2))
    Mz = sparse(iiz[1:kz], jjz[1:kz], sz[1:kz], (Nx+2)*(Ny+2)*(Nz+2), (Nx+2)*(Ny+2)*(Nz+2))
    M = Mx + My + Mz;

def convectionTerm(u: FaceVariable) -> csr_array:
    if (type(u.domain) is Mesh1D):
        return convectionTerm1D(u)
    elif (type(u.domain) is MeshCylindrical1D):
        return convectionTermCylindrical1D(u)
    elif (type(u.domain) is Mesh2D):
        return convectionTerm2D(u)
    elif (type(u.domain) is MeshCylindrical2D):
        # raise Exception("Not implemented yet. Work in progress")
        return convectionTermCylindrical2D(u)
    elif (type(u.domain) is MeshRadial2D):
        # raise Exception("Not implemented yet. Work in progress")
        return convectionTermRadial2D(u)
    elif (type(u.domain) is Mesh3D):
        # raise Exception("Not implemented yet. Work in progress")
        return convectionTerm3D(u)
    elif (type(u.domain) is MeshCylindrical3D):
        raise Exception("Not implemented yet. Work in progress")
        # return convectionTermCylindrical3D(u)
    else:
        raise Exception("convectionTerm is not defined for this Mesh type.")


def convectionUpwindTerm(u: FaceVariable, *args) -> csr_array:
    if (type(u.domain) is Mesh1D):
        return convectionUpwindTerm1D(u, *args)
    elif (type(u.domain) is MeshCylindrical1D):
        return convectionUpwindTermCylindrical1D(u, *args)
    elif (type(u.domain) is Mesh2D):
        return convectionUpwindTerm2D(u, *args)
    elif (type(u.domain) is MeshCylindrical2D):
        # raise Exception("Not implemented yet. Work in progress")
        return convectionUpwindTermCylindrical2D(u)
    elif (type(u.domain) is MeshRadial2D):
        # raise Exception("Not implemented yet. Work in progress")
        return convectionUpwindTermRadial2D(u)
    elif (type(u.domain) is Mesh3D):
        raise Exception("Not implemented yet. Work in progress")
        # return convectionUpwindTerm3D(u)
    elif (type(u.domain) is MeshCylindrical3D):
        raise Exception("Not implemented yet. Work in progress")
        # return convectionUpwindTermCylindrical3D(u)
    else:
        raise Exception(
            "convectionUpwindTerm is not defined for this Mesh type.")


def convectionTvdRHSTerm(u: FaceVariable) -> np.ndarray:
    if (type(u.domain) is Mesh1D):
        return convectionTvdRHS1D(u)
    elif (type(u.domain) is MeshCylindrical1D):
        return convectionTvdRHSCylindrical1D(u)
    elif (type(u.domain) is Mesh2D):
        # raise Exception("Not implemented yet. Work in progress")
        return convectionTvdRHS2D(u)
    elif (type(u.domain) is MeshCylindrical2D):
        # raise Exception("Not implemented yet. Work in progress")
        return convectionTvdRHSCylindrical2D(u)
    elif (type(u.domain) is MeshRadial2D):
        # raise Exception("Not implemented yet. Work in progress")
        return convectionTvdRHSRadial2D(u)
    elif (type(u.domain) is Mesh3D):
        raise Exception("Not implemented yet. Work in progress")
        # return convectionTvdRHS3D(u)
    elif (type(u.domain) is MeshCylindrical3D):
        raise Exception("Not implemented yet. Work in progress")
        # return convectionTvdRHSCylindrical3D(u)
    else:
        raise Exception(
            "convectionTvdRHSTerm is not defined for this Mesh type.")
