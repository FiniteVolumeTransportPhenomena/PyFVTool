# diffusion terms
import numpy as np
from scipy.sparse import csr_array
from mesh import *
from utilities import *
from cell import *
from face import *


class DiffusionTerm:
    def __init__(self, D: FaceVariable):
        self.D = D


def diffusionTerm1D(D: FaceVariable) -> csr_array:
    # extract data from the mesh structure
    Nx = D.domain.dims[0]
    G = D.domain.cell_numbers()
    DX = D.domain.cellsize.x
    dx = 0.5*(DX[0:-1]+DX[1:])

    # define the vectors to store the sparse matrix data
    iix = np.zeros(3*(Nx+2), dtype=int)
    jjx = np.zeros(3*(Nx+2), dtype=int)
    sx = np.zeros(3*(Nx+2), dtype=float)

    # extract the velocity data
    # note: size(Dx) = [1:m+1, 1:n] and size(Dy) = [1:m, 1:n+1]
    Dx = D.xvalue

    # reassign the east, west, north, and south velocity vectors for the
    # code readability
    De = Dx[1:Nx+1]/(dx[1:Nx+1]*DX[1:Nx+1])
    Dw = Dx[0:Nx]/(dx[0:Nx]*DX[1:Nx+1])

    # calculate the coefficients for the internal cells
    AE = De  # reshape(De,Nx,1)
    AW = Dw  # reshape(Dw,Nx,1)
    APx = -(AE+AW)

    # build the sparse matrix based on the numbering system
    rowx_index = G[1:Nx+1]  # reshape(G[1:Nx+1],Nx,1) # main diagonal x
    iix[0:3*Nx] = np.tile(rowx_index, 3)
    jjx[0:3*Nx] = np.hstack([G[0:Nx], G[1:Nx+1], G[2:Nx+2]])
    sx[0:3*Nx] = np.hstack([AW, APx, AE])

    # build the sparse matrix
    kx = 3*Nx
    return csr_array((sx[0:kx], (iix[0:kx], jjx[0:kx])), shape=(Nx+2, Nx+2))


def diffusionTerm2D(D: FaceVariable) -> csr_array:
    # D is a face variable
    # extract data from the mesh structure
    Nx, Ny = D.domain.dims
    G = D.domain.cell_numbers()
    DX = D.domain.cellsize.x
    DY = D.domain.cellsize.y
    dx = 0.5*(DX[0:-1]+DX[1:])
    dy = 0.5*(DY[0:-1]+DY[1:])

    # define the vectors to store the sparse matrix data
    iix = np.zeros(3*(Nx+2)*(Ny+2), dtype=int)
    iiy = np.zeros(3*(Nx+2)*(Ny+2), dtype=int)
    jjx = np.zeros(3*(Nx+2)*(Ny+2), dtype=int)
    jjy = np.zeros(3*(Nx+2)*(Ny+2), dtype=int)
    sx = np.zeros(3*(Nx+2)*(Ny+2), dtype=float)
    sy = np.zeros(3*(Nx+2)*(Ny+2), dtype=float)
    mnx = Nx*Ny
    mny = Nx*Ny

    # reassign the east, west for code readability (use broadcasting in Julia)
    De = D.xvalue[1:Nx+1, :]/(dx[1:Nx+1]*DX[1:Nx+1])[:, np.newaxis]
    Dw = D.xvalue[0:Nx, :]/(dx[0:Nx]*DX[1:Nx+1])[:, np.newaxis]
    Dn = D.yvalue[:, 1:Ny+1]/(dy[1:Ny+1]*DY[1:Ny+1])
    Ds = D.yvalue[:, 0:Ny]/(dy[0:Ny]*DY[1:Ny+1])

    # calculate the coefficients for the internal cells
    AE = De.ravel()
    AW = Dw.ravel()
    AN = Dn.ravel()
    AS = Ds.ravel()
    APx = -(AE+AW)
    APy = -(AN+AS)

    # build the sparse matrix based on the numbering system
    rowx_index = G[1:Nx+1, 1:Ny+1].ravel()  # main diagonal x
    iix[0:3*mnx] = np.tile(rowx_index, 3)
    rowy_index = G[1:Nx+1, 1:Ny+1].ravel()  # main diagonal y
    iiy[0:3*mny] = np.tile(rowy_index, 3)
    jjx[0:3*mnx] = np.hstack([G[0:Nx, 1:Ny+1].ravel(),
                              G[1:Nx+1, 1:Ny+1].ravel(),
                              G[2:Nx+2, 1:Ny+1].ravel()])
    jjy[0:3*mny] = np.hstack([G[1:Nx+1, 0:Ny].ravel(),
                             G[1:Nx+1, 1:Ny+1].ravel(), G[1:Nx+1, 2:Ny+2].ravel()])
    sx[0:3*mnx] = np.hstack([AW, APx, AE])
    sy[0:3*mny] = np.hstack([AS, APy, AN])

    # build the sparse matrix
    kx = 3*mnx
    ky = 3*mny
    Mx = csr_array((sx[0:kx], (iix[0:kx], jjx[0:kx])),
                   shape=((Nx+2)*(Ny+2), (Nx+2)*(Ny+2)))
    My = csr_array((sy[0:kx], (iiy[0:ky], jjy[0:ky])),
                   shape=((Nx+2)*(Ny+2), (Nx+2)*(Ny+2)))
    M = Mx + My
    return (M, Mx, My)


def diffusionTerm(D: FaceVariable) -> csr_array:
    if isinstance(type(D.domain), Mesh1D):
        return diffusionTerm1D(D)
    elif isinstance(type(D.domain), Mesh1D):
        pass
