# Diffusion terms

import numpy as np
from scipy.sparse import csr_array

from .mesh import Grid1D, Grid2D, Grid3D
from .mesh import CylindricalGrid1D, CylindricalGrid2D
from .mesh import PolarGrid2D, CylindricalGrid3D
from .face import FaceVariable



def diffusionTerm1D(D: FaceVariable) -> csr_array:
    # extract data from the mesh structure
    Nx = D.domain.dims[0]
    G = D.domain.cell_numbers()
    DX = D.domain.cellsize.x
    dx = 0.5*(DX[0:-1]+DX[1:])

    # extract the velocity data
    # note: size(Dx) = [1:m+1, 1:n] and size(Dy) = [1:m, 1:n+1]
    Dx = D.xvalue

    # reassign the east, west, north, and south velocity vectors for the
    # code readability
    De = Dx[1:Nx+1]/(dx[1:Nx+1]*DX[1:Nx+1])
    Dw = Dx[0:Nx]/(dx[0:Nx]*DX[1:Nx+1])

    # calculate the coefficients for the internal cells
    AE = De  # De,Nx,1)
    AW = Dw  # Dw,Nx,1)
    APx = -(AE+AW)

    # build the sparse matrix based on the numbering system
    iix = np.tile(G[1:Nx+1], 3)  # main diagonal x
    jjx = np.hstack([G[0:Nx], G[1:Nx+1], G[2:Nx+2]])
    sx = np.hstack([AW, APx, AE])

    # build the sparse matrix
    kx = 3*Nx
    return csr_array((sx[0:kx], (iix[0:kx], jjx[0:kx])), shape=(Nx+2, Nx+2))


def diffusionTermCylindrical1D(D: FaceVariable) -> csr_array:
    # extract data from the mesh structure
    Nx = D.domain.dims[0]
    G = D.domain.cell_numbers()
    DX = D.domain.cellsize.x
    dx = 0.5*(DX[0:-1]+DX[1:])
    rp = D.domain.cellcenters.x
    rf = D.domain.facecenters.x

    # extract the velocity data
    # note: size(Dx) = [1:m+1, 1:n] and size(Dy) = [1:m, 1:n+1]
    Dx = D.xvalue

    # reassign the east, west, north, and south velocity vectors for the
    # code readability
    De = rf[1:Nx+1]*Dx[1:Nx+1]/(rp*dx[1:Nx+1]*DX[1:Nx+1])
    Dw = rf[0:Nx]*Dx[0:Nx]/(rp*dx[0:Nx]*DX[1:Nx+1])

    # calculate the coefficients for the internal cells
    AE = De  # De,Nx,1)
    AW = Dw  # Dw,Nx,1)
    APx = -(AE+AW)

    # build the sparse matrix based on the numbering system
    iix = np.tile(G[1:Nx+1], 3)  # main diagonal x
    jjx = np.hstack([G[0:Nx], G[1:Nx+1], G[2:Nx+2]])
    sx = np.hstack([AW, APx, AE])

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
    mn = Nx*Ny

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
    ii = np.tile(G[1:Nx+1, 1:Ny+1].ravel(), 3)  # main diagonal x, y
    jjx = np.hstack([G[0:Nx, 1:Ny+1].ravel(),
                     G[1:Nx+1, 1:Ny+1].ravel(),
                     G[2:Nx+2, 1:Ny+1].ravel()])
    jjy = np.hstack([G[1:Nx+1, 0:Ny].ravel(),
                     G[1:Nx+1, 1:Ny+1].ravel(),
                     G[1:Nx+1, 2:Ny+2].ravel()])
    sx = np.hstack([AW, APx, AE])
    sy = np.hstack([AS, APy, AN])

    # build the sparse matrix
    kx = 3*mn
    ky = 3*mn
    Mx = csr_array((sx[0:kx], (ii[0:kx], jjx[0:kx])),
                   shape=((Nx+2)*(Ny+2), (Nx+2)*(Ny+2)))
    My = csr_array((sy[0:kx], (ii[0:ky], jjy[0:ky])),
                   shape=((Nx+2)*(Ny+2), (Nx+2)*(Ny+2)))
    M = Mx + My
    return M, Mx, My


def diffusionTermCylindrical2D(D: FaceVariable) -> csr_array:
    # D is a face variable
    # extract data from the mesh structure
    Nx, Ny = D.domain.dims
    G = D.domain.cell_numbers()
    DX = D.domain.cellsize.x
    DY = D.domain.cellsize.y
    dx = 0.5*(DX[0:-1]+DX[1:])
    dy = 0.5*(DY[0:-1]+DY[1:])
    rp = D.domain.cellcenters.x
    rf = D.domain.facecenters.x[:, np.newaxis]
    mn = Nx*Ny

    # reassign the east, west for code readability (use broadcasting in Julia)
    De = rf[1:Nx+1]*D.xvalue[1:Nx+1, :] / \
        (rp*dx[1:Nx+1]*DX[1:Nx+1])[:, np.newaxis]
    Dw = rf[0:Nx]*D.xvalue[0:Nx, :]/(rp*dx[0:Nx]*DX[1:Nx+1])[:, np.newaxis]
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
    ii = np.tile(G[1:Nx+1, 1:Ny+1].ravel(), 3)  # main diagonal x, y
    jjx = np.hstack([G[0:Nx, 1:Ny+1].ravel(),
                     G[1:Nx+1, 1:Ny+1].ravel(),
                     G[2:Nx+2, 1:Ny+1].ravel()])
    jjy = np.hstack([G[1:Nx+1, 0:Ny].ravel(),
                     G[1:Nx+1, 1:Ny+1].ravel(),
                     G[1:Nx+1, 2:Ny+2].ravel()])
    sx = np.hstack([AW, APx, AE])
    sy = np.hstack([AS, APy, AN])

    # build the sparse matrix
    kx = 3*mn
    ky = 3*mn
    Mx = csr_array((sx[0:kx], (ii[0:kx], jjx[0:kx])),
                   shape=((Nx+2)*(Ny+2), (Nx+2)*(Ny+2)))
    My = csr_array((sy[0:kx], (ii[0:ky], jjy[0:ky])),
                   shape=((Nx+2)*(Ny+2), (Nx+2)*(Ny+2)))
    M = Mx + My
    return M, Mx, My


def diffusionTermPolar2D(D: FaceVariable) -> csr_array:
    # D is a face variable
    # extract data from the mesh structure
    Nx, Ny = D.domain.dims
    G = D.domain.cell_numbers()
    DX = D.domain.cellsize.x
    DY = D.domain.cellsize.y
    dx = 0.5*(DX[0:-1]+DX[1:])
    dy = 0.5*(DY[0:-1]+DY[1:])
    rp = D.domain.cellcenters.x[:, np.newaxis]
    rf = D.domain.facecenters.x[:, np.newaxis]
    mn = Nx*Ny

    # reassign the east, west for code readability (use broadcasting in Julia)
    De = rf[1:Nx+1]*D.xvalue[1:Nx+1, :] / \
        (rp*dx[1:Nx+1][:, np.newaxis]*DX[1:Nx+1][:, np.newaxis])
    Dw = rf[0:Nx]*D.xvalue[0:Nx, :] / \
        (rp*dx[0:Nx][:, np.newaxis]*DX[1:Nx+1][:, np.newaxis])
    Dn = D.yvalue[:, 1:Ny+1] / \
        (rp*rp*dy[1:Ny+1][np.newaxis, :]*DY[1:Ny+1][np.newaxis, :])
    Ds = D.yvalue[:, 0:Ny]/(rp*rp*dy[0:Ny][np.newaxis, :]
                            * DY[1:Ny+1][np.newaxis, :])

    # calculate the coefficients for the internal cells
    AE = De #.ravel()
    AW = Dw #.ravel()
    AN = Dn #.ravel()
    AS = Ds #.ravel()
    APx = -(AE+AW)
    APy = -(AN+AS)

    # build the sparse matrix based on the numbering system
    ii = np.tile(G[1:Nx+1, 1:Ny+1].ravel(), 3)  # main diagonal x, y
    jjx = np.hstack([G[0:Nx, 1:Ny+1].ravel(),
                     G[1:Nx+1, 1:Ny+1].ravel(),
                     G[2:Nx+2, 1:Ny+1].ravel()])
    jjy = np.hstack([G[1:Nx+1, 0:Ny].ravel(),
                     G[1:Nx+1, 1:Ny+1].ravel(),
                     G[1:Nx+1, 2:Ny+2].ravel()])
    sx = np.hstack([AW, APx, AE]).ravel()
    sy = np.hstack([AS, APy, AN]).ravel()

    # build the sparse matrix
    kx = 3*mn
    ky = 3*mn
    Mx = csr_array((sx[0:kx], (ii[0:kx], jjx[0:kx])),
                   shape=((Nx+2)*(Ny+2), (Nx+2)*(Ny+2)))
    My = csr_array((sy[0:kx], (ii[0:ky], jjy[0:ky])),
                   shape=((Nx+2)*(Ny+2), (Nx+2)*(Ny+2)))
    M = Mx + My
    return M, Mx, My


def diffusionTerm3D(D: FaceVariable) -> csr_array:
    # D is a face variable
    # extract data from the mesh structure
    Nx, Ny, Nz = D.domain.dims
    G = D.domain.cell_numbers()
    DX = D.domain.cellsize.x
    DY = D.domain.cellsize.y
    DZ = D.domain.cellsize.z
    dx = 0.5*(DX[0:-1]+DX[1:])
    dy = 0.5*(DY[0:-1]+DY[1:])
    dz = 0.5*(DZ[0:-1]+DZ[1:])

    # define the vectors to store the sparse matrix data
    mn = Nx*Ny*Nz

    # reassign the east, west, north, and south velocity vectors for the
    # code readability (use broadcasting)
    De = D.xvalue[1:Nx+1, :, :] / \
        (dx[1:Nx+1]*DX[1:Nx+1])[:, np.newaxis, np.newaxis]
    Dw = D.xvalue[0:Nx, :, :]/(dx[0:Nx]*DX[1:Nx+1])[:, np.newaxis, np.newaxis]
    Dn = D.yvalue[:, 1:Ny+1, :] / \
        (dy[1:Ny+1]*DY[1:Ny+1])[np.newaxis, :, np.newaxis]
    Ds = D.yvalue[:, 0:Ny, :]/(dy[0:Ny]*DY[1:Ny+1])[np.newaxis, :, np.newaxis]
    Df = D.zvalue[:, :, 1:Nz+1] / \
        (dz[1:Nz+1]*DZ[1:Nz+1])[np.newaxis, np.newaxis, :]
    Db = D.zvalue[:, :, 0:Nz]/(dz[0:Nz]*DZ[1:Nz+1])[np.newaxis, np.newaxis, :]

    # calculate the coefficients for the internal cells
    AE = De.ravel()
    AW = Dw.ravel()
    AN = Dn.ravel()
    AS = Ds.ravel()
    AF = Df.ravel()
    AB = Db.ravel()
    APx = -(De+Dw).ravel()
    APy = -(Dn+Ds).ravel()
    APz = -(Df+Db).ravel()

    # build the sparse matrix based on the numbering system
    ii = np.tile(G[1:Nx+1, 1:Ny+1, 1:Nz+1].ravel(), 3)  # main diagonal x, y, z
    jjx = np.hstack([G[0:Nx, 1:Ny+1, 1:Nz+1].ravel(),
                    G[1:Nx+1, 1:Ny+1, 1:Nz+1].ravel(),
                     G[2:Nx+2, 1:Ny+1, 1:Nz+1].ravel()])
    jjy = np.hstack([G[1:Nx+1, 0:Ny, 1:Nz+1].ravel(),
                    G[1:Nx+1, 1:Ny+1, 1:Nz+1].ravel(),
                     G[1:Nx+1, 2:Ny+2, 1:Nz+1].ravel()])
    jjz = np.hstack([G[1:Nx+1, 1:Ny+1, 0:Nz].ravel(),
                    G[1:Nx+1, 1:Ny+1, 1:Nz+1].ravel(),
                     G[1:Nx+1, 1:Ny+1, 2:Nz+2].ravel()])
    sx = np.hstack([AW, APx, AE]) #.ravel()
    sy = np.hstack([AS, APy, AN]) #.ravel()
    sz = np.hstack([AB, APz, AF]) #.ravel()

    # build the sparse matrix
    kx = ky = kz = 3*mn
    m_shape = ((Nx+2)*(Ny+2)*(Nz+2), (Nx+2)*(Ny+2)*(Nz+2))
    Mx = csr_array((sx[0:kx], (ii[0:kx], jjx[0:kx])), shape=m_shape)
    My = csr_array((sy[0:ky], (ii[0:ky], jjy[0:ky])), shape=m_shape)
    Mz = csr_array((sz[0:kz], (ii[0:kz], jjz[0:kz])), shape=m_shape)
    M = Mx + My + Mz
    return (M, Mx, My, Mz)


def diffusionTermCylindrical3D(D: FaceVariable) -> csr_array:
    # D is a face variable
    # extract data from the mesh structure
    Nx, Ny, Nz = D.domain.dims
    G = D.domain.cell_numbers()
    DX = D.domain.cellsize.x
    DY = D.domain.cellsize.y
    DZ = D.domain.cellsize.z
    dx = 0.5*(DX[0:-1]+DX[1:])
    dy = 0.5*(DY[0:-1]+DY[1:])
    dz = 0.5*(DZ[0:-1]+DZ[1:])
    rp = D.domain.cellcenters.x[:, np.newaxis, np.newaxis]
    rf = D.domain.facecenters.x[:, np.newaxis, np.newaxis]

    # define the vectors to store the sparse matrix data
    mn = Nx*Ny*Nz

    # reassign the east, west, north, and south velocity vectors for the
    # code readability (use broadcasting)
    De = rf[1:Nx+1]*D.xvalue[1:Nx+1, :, :] / \
        (rp*dx[1:Nx+1][:, np.newaxis, np.newaxis]
         * DX[1:Nx+1][:, np.newaxis, np.newaxis])
    Dw = rf[0:Nx]*D.xvalue[0:Nx, :, :] / \
        (rp*dx[0:Nx][:, np.newaxis, np.newaxis]
         * DX[1:Nx+1][:, np.newaxis, np.newaxis])
    Dn = D.yvalue[:, 1:Ny+1, :]/(rp*rp*dy[1:Ny+1][np.newaxis,
                                 :, np.newaxis]*DY[1:Ny+1][np.newaxis, :, np.newaxis])
    Ds = D.yvalue[:, 0:Ny, :]/(rp*rp*dy[0:Ny][np.newaxis,
                               :, np.newaxis]*DY[1:Ny+1][np.newaxis, :, np.newaxis])
    Df = D.zvalue[:, :, 1:Nz+1] / \
        (dz[1:Nz+1]*DZ[1:Nz+1])[np.newaxis, np.newaxis, :]
    Db = D.zvalue[:, :, 0:Nz]/(dz[0:Nz]*DZ[1:Nz+1])[np.newaxis, np.newaxis, :]

    # calculate the coefficients for the internal cells
    AE = De.ravel()
    AW = Dw.ravel()
    AN = Dn.ravel()
    AS = Ds.ravel()
    AF = Df.ravel()
    AB = Db.ravel()
    APx = -(De+Dw).ravel()
    APy = -(Dn+Ds).ravel()
    APz = -(Df+Db).ravel()

    # build the sparse matrix based on the numbering system
    ii = np.tile(G[1:Nx+1, 1:Ny+1, 1:Nz+1].ravel(), 3)  # main diagonal x, y, z
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
    m_shape = ((Nx+2)*(Ny+2)*(Nz+2), (Nx+2)*(Ny+2)*(Nz+2))
    Mx = csr_array((sx[0:kx], (ii[0:kx], jjx[0:kx])), shape=m_shape)
    My = csr_array((sy[0:ky], (ii[0:ky], jjy[0:ky])), shape=m_shape)
    Mz = csr_array((sz[0:kz], (ii[0:kz], jjz[0:kz])), shape=m_shape)
    M = Mx + My + Mz
    return (M, Mx, My, Mz)


def diffusionTerm(D: FaceVariable) -> csr_array:
    """Builds the discretized diffusion term in matrix form for a given
    Cartesian mesh.

    Parameters
    ----------
    D : FaceVariable
        The diffusion coefficient.
    
    Returns
    -------
    csr_array
        The discretized diffusion term in matrix form.

    Examples
    --------
    >>> import pyfvtool as pf
    >>> m = pf.Grid1D(10, 1.0)
    >>> D = pf.FaceVariable(m, 1.0)
    >>> M = pf.diffusionTerm(D)
    """
    if (type(D.domain) is Grid1D):
        return diffusionTerm1D(D)
    elif (type(D.domain) is CylindricalGrid1D):
        return diffusionTermCylindrical1D(D)
    elif (type(D.domain) is Grid2D):
        return diffusionTerm2D(D)[0]
    elif (type(D.domain) is CylindricalGrid2D):
        return diffusionTermCylindrical2D(D)[0]
    elif (type(D.domain) is PolarGrid2D):
        return diffusionTermPolar2D(D)[0]
    elif (type(D.domain) is Grid3D):
        return diffusionTerm3D(D)[0]
    elif (type(D.domain) is CylindricalGrid3D):
        return diffusionTermCylindrical3D(D)[0]
    else:
        raise Exception("DiffusionTerm is not defined for this Mesh type.")
