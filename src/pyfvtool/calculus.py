import numpy as np

from .mesh import Grid1D, Grid2D, Grid3D
from .mesh import CylindricalGrid1D, CylindricalGrid2D
from .mesh import PolarGrid2D, CylindricalGrid3D, SphericalGrid1D, SphericalGrid3D
from .cell import CellVariable
from .face import FaceVariable



def gradientTerm(phi: CellVariable):
    """
    This function calculates the gradient of a cell variable. The output is a face variable.

    Parameters
    ----------
    phi : CellVariable
        The cell variable for which the gradient is calculated.
    
    Returns
    -------
    FaceVariable
        The gradient of the cell variable.
    
    Examples
    --------
    >>> import pyfvtool as pf
    >>> m = pf.Grid1D(10, 1.0)
    >>> phi = pf.CellVariable(m, 1.0)
    >>> gradPhi = pf.gradientTerm(phi)
    >>> gradPhi._xvalue
    """
    # calculates the gradient of a variable
    # the output is a face variable
    if issubclass(type(phi.domain), Grid1D):
        dx = 0.5*(phi.domain.cellsize._x[0:-1]+phi.domain.cellsize._x[1:])
        return FaceVariable(phi.domain,
                     (phi._value[1:]-phi._value[0:-1])/dx,
                     np.array([]),
                     np.array([]))
    elif (type(phi.domain) is Grid2D) or (type(phi.domain) is CylindricalGrid2D):
        dx = 0.5*(phi.domain.cellsize._x[0:-1]+phi.domain.cellsize._x[1:])
        dy = 0.5*(phi.domain.cellsize._y[0:-1]+phi.domain.cellsize._y[1:])
        return FaceVariable(phi.domain,
                     (phi._value[1:, 1:-1]-phi._value[0:-1, 1:-1])/dx[:,np.newaxis],
                     (phi._value[1:-1, 1:]-phi._value[1:-1, 0:-1])/dy,
                     np.array([]))
    elif (type(phi.domain) is PolarGrid2D):
        dx = 0.5*(phi.domain.cellsize._x[0:-1]+phi.domain.cellsize._x[1:])
        dtheta = 0.5*(phi.domain.cellsize._y[0:-1]+phi.domain.cellsize._y[1:])
        rp = phi.domain.cellcenters._x
        return FaceVariable(phi.domain,
                     (phi._value[1:, 1:-1]-phi._value[0:-1, 1:-1])/dx[:,np.newaxis],
                     (phi._value[1:-1, 1:]-phi._value[1:-1, 0:-1])/(dtheta[np.newaxis,:]*rp[:,np.newaxis]),
                     np.array([]))
    elif (type(phi.domain) is Grid3D):
        dx = 0.5*(phi.domain.cellsize._x[0:-1]+phi.domain.cellsize._x[1:])
        dy = 0.5*(phi.domain.cellsize._y[0:-1]+phi.domain.cellsize._y[1:])
        dz = 0.5*(phi.domain.cellsize._z[0:-1]+phi.domain.cellsize._z[1:])
        return FaceVariable(phi.domain,
                     (phi._value[1:, 1:-1, 1:-1] -
                     phi._value[0:-1, 1:-1, 1:-1])/dx[:,np.newaxis,np.newaxis],
                     (phi._value[1:-1, 1:, 1:-1] -
                      phi._value[1:-1, 0:-1, 1:-1])/dy[np.newaxis,:,np.newaxis],
                     (phi._value[1:-1, 1:-1, 1:] -
                     phi._value[1:-1, 1:-1, 0:-1])/dz[np.newaxis,np.newaxis,:])
    elif (type(phi.domain) is CylindricalGrid3D):
        dx = 0.5*(phi.domain.cellsize._x[0:-1]+phi.domain.cellsize._x[1:])
        dy = 0.5*(phi.domain.cellsize._y[0:-1]+phi.domain.cellsize._y[1:])
        dz = 0.5*(phi.domain.cellsize._z[0:-1]+phi.domain.cellsize._z[1:])
        rp = phi.domain.cellcenters._x
        return FaceVariable(phi.domain,
                     (phi._value[1:, 1:-1, 1:-1] -
                      phi._value[0:-1, 1:-1, 1:-1])/dx[:,np.newaxis,np.newaxis],
                     (phi._value[1:-1, 1:, 1:-1] -
                      phi._value[1:-1, 0:-1, 1:-1])/(dy[np.newaxis,:,np.newaxis]*rp[:,np.newaxis,np.newaxis]),
                     (phi._value[1:-1, 1:-1, 1:] -
                     phi._value[1:-1, 1:-1, 0:-1])/dz[np.newaxis,np.newaxis,:])
    elif (type(phi.domain) is SphericalGrid3D):
        dx = 0.5*(phi.domain.cellsize._x[0:-1]+phi.domain.cellsize._x[1:])
        dy = 0.5*(phi.domain.cellsize._y[0:-1]+phi.domain.cellsize._y[1:])
        dz = 0.5*(phi.domain.cellsize._z[0:-1]+phi.domain.cellsize._z[1:])
        rp = phi.domain.cellcenters._x[:, np.newaxis, np.newaxis]
        thetap = phi.domain.cellcenters._y[np.newaxis, :, np.newaxis]
        return FaceVariable(phi.domain,
                     (phi._value[1:, 1:-1, 1:-1] -
                      phi._value[0:-1, 1:-1, 1:-1])/dx[:,np.newaxis,np.newaxis],
                     (phi._value[1:-1, 1:, 1:-1] -
                      phi._value[1:-1, 0:-1, 1:-1])/(dy[np.newaxis,:,np.newaxis]*rp),
                     (phi._value[1:-1, 1:-1, 1:] -
                     phi._value[1:-1, 1:-1, 0:-1])/(dz[np.newaxis,np.newaxis,:]*rp*np.sin(thetap)))

# =============== Divergence 1D Term ============================
def divergenceTerm1D(F: FaceVariable):
    # This def calculates the divergence of a field
    # using its face
    # extract data from the mesh structure
    Nx = F.domain.dims[0]
    G = F.domain.cell_numbers()
    DX = F.domain.cellsize._x[1:-1]
    # define the vector of cell index
    row_index = G[1:Nx+1] # main diagonal
    # compute the divergence
    div_x = (F._xvalue[1:Nx+1]-F._xvalue[0:Nx])/DX
    # define the RHS Vector
    RHSdiv = np.zeros(Nx+2)
    # assign the values of the RHS vector
    RHSdiv[row_index] = div_x
    return RHSdiv


# =============== Divergence Cylindrical 1D Term ============================
def divergenceTermCylindrical1D(F: FaceVariable):
    # This def calculates the divergence of a field
    # using its face
    # extract data from the mesh structure
    Nx = F.domain.dims[0]
    G = F.domain.cell_numbers()
    DX = F.domain.cellsize._x[1:-1]
    rp = F.domain.cellcenters._x
    rf = F.domain.facecenters._x
    # define the vector of cell index
    row_index = G[1:Nx+1] # main diagonal
    # reassign the east, west, north, and south flux vectors for the
    # code readability
    Fe = F._xvalue[1:Nx+1]
    Fw = F._xvalue[0:Nx]
    re = rf[1:Nx+1]
    rw = rf[0:Nx]
    # compute the divergence
    div_x = (re*Fe-rw*Fw)/(DX*rp)
    # define the RHS Vector
    RHSdiv = np.zeros(Nx+2)
    # assign the values of the RHS vector
    RHSdiv[row_index] = div_x
    return RHSdiv

# =============== Divergence 2D Term ============================
def divergenceTerm2D(F: FaceVariable):
    # This def calculates the divergence of a field
    # using its face
    # extract data from the mesh structure
    Nx, Ny = F.domain.dims
    G= F.domain.cell_numbers()
    DX = F.domain.cellsize._x[1:-1][:, np.newaxis]
    DY = F.domain.cellsize._y[1:-1][np.newaxis, :]
    # define the vector of cell index
    row_index = G[1:Nx+1,1:Ny+1].ravel() # main diagonal
    # reassign the east, west, north, and south flux vectors for the
    # code readability
    Fe = F._xvalue[1:Nx+1,:]
    Fw = F._xvalue[0:Nx,:]
    Fn = F._yvalue[:,1:Ny+1]
    Fs = F._yvalue[:,0:Ny]
    # compute the divergence
    div_x = (Fe - Fw)/DX
    div_y = (Fn - Fs)/DY
    # define the RHS Vector
    RHSdiv = np.zeros((Nx+2)*(Ny+2))
    RHSdivx = np.zeros((Nx+2)*(Ny+2))
    RHSdivy = np.zeros((Nx+2)*(Ny+2))
    # assign the values of the RHS vector
    RHSdiv[row_index] = (div_x+div_y).ravel()
    RHSdivx[row_index] = div_x.ravel()
    RHSdivy[row_index] = div_y.ravel()
    return RHSdiv, RHSdivx, RHSdivy



# =============== Divergence 2D Cylindrical Term ============================
def divergenceTermCylindrical2D(F:FaceVariable):
    # This def calculates the divergence of a field
    # using its face
    # extract data from the mesh structure
    Nr, Nz = F.domain.dims
    G= F.domain.cell_numbers()
    dr = F.domain.cellsize._x[1:-1][:, np.newaxis]
    dz = F.domain.cellsize._y[1:-1][np.newaxis, :]
    rp = F.domain.cellcenters._x[:, np.newaxis]
    rf = F.domain.facecenters._x[:, np.newaxis]
    # define the vector of cell index
    row_index = G[1:Nr+1,1:Nz+1].ravel() # main diagonal
    # reassign the east, west, north, and south flux vectors for the
    # code readability
    Fe = F._xvalue[1:Nr+1,:]
    Fw = F._xvalue[0:Nr,:]
    Fn = F._yvalue[:,1:Nz+1]
    Fs = F._yvalue[:,0:Nz]
    re = rf[1:Nr+1]
    rw = rf[0:Nr]
    # compute the divergence
    div_x = (re*Fe - rw*Fw)/(dr*rp)
    div_y = (Fn - Fs)/dz
    # define the RHS Vector
    RHSdiv = np.zeros((Nr+2)*(Nz+2))
    RHSdivx = np.zeros((Nr+2)*(Nz+2))
    RHSdivy = np.zeros((Nr+2)*(Nz+2))
    # assign the values of the RHS vector
    RHSdiv[row_index] = (div_x+div_y).ravel()
    RHSdivx[row_index] = div_x.ravel()
    RHSdivy[row_index] = div_y.ravel()
    return RHSdiv, RHSdivx, RHSdivy


# =============== Divergence 2D Polar Term ============================
def divergenceTermPolar2D(F:FaceVariable):
    # This def calculates the divergence of a field
    # using its face
    # extract data from the mesh structure
    Nr, Ntheta = F.domain.dims
    G=F.domain.cell_numbers()
    dr = F.domain.cellsize._x[1:-1][:, np.newaxis]
    dtheta= F.domain.cellsize._y[1:-1][np.newaxis, :]
    rp = F.domain.cellcenters._x[:, np.newaxis]
    rf = F.domain.facecenters._x[:, np.newaxis]
    # define the vector of cell index
    row_index = G[1:Nr+1,1:Ntheta+1].ravel() # main diagonal
    # reassign the east, west, north, and south flux vectors for the
    # code readability
    Fe = F._xvalue[1:Nr+1,:]
    Fw = F._xvalue[0:Nr,:]
    Fn = F._yvalue[:,1:Ntheta+1]
    Fs = F._yvalue[:,0:Ntheta]
    re = rf[1:Nr+1]
    rw = rf[0:Nr]
    # compute the divergence
    div_x = (re*Fe-rw*Fw)/(dr*rp)
    div_y = (Fn-Fs)/(dtheta*rp)
    # define the RHS Vector
    RHSdiv = np.zeros((Nr+2)*(Ntheta+2))
    RHSdivx = np.zeros((Nr+2)*(Ntheta+2))
    RHSdivy = np.zeros((Nr+2)*(Ntheta+2))
    # assign the values of the RHS vector
    RHSdiv[row_index] = (div_x+div_y).ravel()
    RHSdivx[row_index] = div_x.ravel()
    RHSdivy[row_index] = div_y.ravel()
    return RHSdiv, RHSdivx, RHSdivy

# =============== Divergence 3D Term ============================
def divergenceTerm3D(F:FaceVariable):
    # This def calculates the divergence of a field
    # using its face
    # extract data from the mesh structure
    Nx, Ny, Nz = F.domain.dims
    G=F.domain.cell_numbers()
    dx = F.domain.cellsize._x[1:-1][:,np.newaxis,np.newaxis]
    dy = F.domain.cellsize._y[1:-1][np.newaxis,:,np.newaxis]
    dz = F.domain.cellsize._z[1:-1][np.newaxis,np.newaxis,:]
    # define the vector of cell index
    row_index = G[1:Nx+1,1:Ny+1,1:Nz+1].ravel() # main diagonal
    # reassign the east, west, north, and south flux vectors for the
    # code readability
    Fe = F._xvalue[1:Nx+1,:,:]
    Fw = F._xvalue[0:Nx,:,:]
    Fn = F._yvalue[:,1:Ny+1,:]
    Fs = F._yvalue[:,0:Ny,:]
    Ff = F._zvalue[:,:,1:Nz+1]
    Fb = F._zvalue[:,:,0:Nz]
    # compute the divergence
    div_x = (Fe - Fw)/dx
    div_y = (Fn - Fs)/dy
    div_z = (Ff - Fb)/dz
    # define the RHS Vector
    RHSdiv = np.zeros((Nx+2)*(Ny+2)*(Nz+2))
    RHSdivx = np.zeros((Nx+2)*(Ny+2)*(Nz+2))
    RHSdivy = np.zeros((Nx+2)*(Ny+2)*(Nz+2))
    RHSdivz = np.zeros((Nx+2)*(Ny+2)*(Nz+2))
    # assign the values of the RHS vector
    RHSdiv[row_index] = (div_x+div_y+div_z).ravel()
    RHSdivx[row_index] = div_x.ravel()
    RHSdivy[row_index] = div_y.ravel()
    RHSdivz[row_index] = div_z.ravel()
    return RHSdiv, RHSdivx, RHSdivy, RHSdivz

# =============== Divergence 3D Cylindrical Term ============================
def divergenceTermCylindrical3D(F:FaceVariable):
    # This def calculates the divergence of a field
    # using its face
    # extract data from the mesh structure
    Nx, Ny, Nz = F.domain.dims
    G=F.domain.cell_numbers()
    dx = F.domain.cellsize._x[1:-1][:,np.newaxis,np.newaxis]
    dy = F.domain.cellsize._y[1:-1][np.newaxis,:,np.newaxis]
    dz = F.domain.cellsize._z[1:-1][np.newaxis,np.newaxis,:]
    rp = F.domain.cellcenters._x[:,np.newaxis,np.newaxis]
    rf = F.domain.facecenters._x[:,np.newaxis,np.newaxis]
    # define the vector of cell index
    row_index = G[1:Nx+1,1:Ny+1,1:Nz+1].ravel() # main diagonal
    # reassign the east, west, north, and south flux vectors for the
    # code readability
    Fe = F._xvalue[1:Nx+1,:,:]
    Fw = F._xvalue[0:Nx,:,:]
    Fn = F._yvalue[:,1:Ny+1,:]
    Fs = F._yvalue[:,0:Ny,:]
    Ff = F._zvalue[:,:,1:Nz+1]
    Fb = F._zvalue[:,:,0:Nz]
    # compute the divergence
    div_x = (rf[1:Nx+1,:,:]*Fe - rf[0:Nx,:,:]*Fw)/(dx*rp)
    div_y = (Fn - Fs)/(dy*rp)
    div_z = (Ff - Fb)/dz
    # define the RHS Vector
    RHSdiv = np.zeros((Nx+2)*(Ny+2)*(Nz+2))
    RHSdivx = np.zeros((Nx+2)*(Ny+2)*(Nz+2))
    RHSdivy = np.zeros((Nx+2)*(Ny+2)*(Nz+2))
    RHSdivz = np.zeros((Nx+2)*(Ny+2)*(Nz+2))
    # assign the values of the RHS vector
    RHSdiv[row_index] = (div_x+div_y+div_z).ravel()
    RHSdivx[row_index] = div_x.ravel()
    RHSdivy[row_index] = div_y.ravel()
    RHSdivz[row_index] = div_z.ravel()
    return RHSdiv, RHSdivx, RHSdivy, RHSdivz

def divergenceTermSpherical1D(F:FaceVariable):
    # This def calculates the divergence of a field
    # using its face
    # extract data from the mesh structure
    Nx = F.domain.dims[0]
    G = F.domain.cell_numbers()
    DX = F.domain.cellsize._x[1:-1]
    rp = F.domain.cellcenters._x
    rf = F.domain.facecenters._x
    # define the vector of cell index
    row_index = G[1:Nx+1] # main diagonal
    # reassign the east, west, north, and south flux vectors for the
    # code readability
    Fe = F._xvalue[1:Nx+1]
    Fw = F._xvalue[0:Nx]
    re = rf[1:Nx+1]
    rw = rf[0:Nx]
    # compute the divergence
    div_x = (re*re*Fe-rw*Fw)/(DX*rp*rp)
    # define the RHS Vector
    RHSdiv = np.zeros(Nx+2)
    # assign the values of the RHS vector
    RHSdiv[row_index] = div_x
    return RHSdiv

def divergenceTermSpherical3D(F:FaceVariable):
    # This def calculates the divergence of a field
    # using its face
    # extract data from the mesh structure
    Nx, Ny, Nz = F.domain.dims
    G=F.domain.cell_numbers()
    dx = F.domain.cellsize._x[1:-1][:,np.newaxis,np.newaxis]
    dy = F.domain.cellsize._y[1:-1][np.newaxis,:,np.newaxis]
    dz = F.domain.cellsize._z[1:-1][np.newaxis,np.newaxis,:]
    rp = F.domain.cellcenters._x[:,np.newaxis,np.newaxis]
    rf = F.domain.facecenters._x[:,np.newaxis,np.newaxis]
    thetap = F.domain.cellcenters._y[np.newaxis,:,np.newaxis]
    thetaf = F.domain.facecenters._y[np.newaxis,:,np.newaxis]
    # define the vector of cell index
    row_index = G[1:Nx+1,1:Ny+1,1:Nz+1].ravel() # main diagonal
    # compute the divergence
    div_x = (rf[1:Nx+1,:,:]**2*F._xvalue[1:Nx+1,:,:] - rf[0:Nx,:,:]**2*F._xvalue[0:Nx,:,:])/(dx*rp**2)
    div_y = (np.sin(thetaf[:,1:Ny+1,:])*F._yvalue[:,1:Ny+1,:] - np.sin(thetaf[:,0:Ny,:])*F._yvalue[:,0:Ny,:])/(dy*rp*np.sin(thetap))
    div_z = (F._zvalue[:,:,1:Nz+1] - F._zvalue[:,:,0:Nz])/(dz*rp*np.sin(thetap))
    # define the RHS Vector
    RHSdiv = np.zeros((Nx+2)*(Ny+2)*(Nz+2))
    RHSdivx = np.zeros((Nx+2)*(Ny+2)*(Nz+2))
    RHSdivy = np.zeros((Nx+2)*(Ny+2)*(Nz+2))
    RHSdivz = np.zeros((Nx+2)*(Ny+2)*(Nz+2))
    # assign the values of the RHS vector
    RHSdiv[row_index] = (div_x+div_y+div_z).ravel()
    RHSdivx[row_index] = div_x.ravel()
    RHSdivy[row_index] = div_y.ravel()
    RHSdivz[row_index] = div_z.ravel()
    return RHSdiv, RHSdivx, RHSdivy, RHSdivz

def divergenceTerm(F: FaceVariable):
    """
    parameters
    ----------
    F : FaceVariable
        The face variable for which the divergence is calculated.

    Returns
    -------
    RHS : ndarray
        The divergence of the face variable returned as a RHS vector.

    Examples
    --------
    >>> import pyfvtool as pf
    >>> m = pf.Grid1D(10, 1.0)
    >>> phi = pf.CellVariable(m, 1.0)
    >>> gradPhi = pf.gradientTerm(phi)
    >>> RHSdiv = pf.divergenceTerm(gradPhi)
    """
    if (type(F.domain) is Grid1D):
        RHSdiv = divergenceTerm1D(F)
    elif (type(F.domain) is CylindricalGrid1D):
        RHSdiv = divergenceTermCylindrical1D(F)
    elif (type(F.domain) is Grid2D):
        RHSdiv, RHSdivx, RHSdivy = divergenceTerm2D(F)
    elif (type(F.domain) is CylindricalGrid2D):
        RHSdiv, RHSdivx, RHSdivy = divergenceTermCylindrical2D(F)
    elif (type(F.domain) is PolarGrid2D):
        RHSdiv, RHSdivx, RHSdivy = divergenceTermPolar2D(F)
    elif (type(F.domain) is Grid3D):
        RHSdiv, RHSdivx, RHSdivy, RHSdivz = divergenceTerm3D(F)
    elif (type(F.domain) is CylindricalGrid3D):
        RHSdiv, RHSdivx, RHSdivy, RHSdivz = divergenceTermCylindrical3D(F)
    else:
        raise Exception("DivergenceTerm is not defined for this Mesh type.")
    return RHSdiv


def gradientTermFixedBC(phi):
    """
    Warning: 
        
    Unless you know for sure that you need this function, do not use it!
    This function calculates the gradient of parameter phi in x,y, and z 
    directions. It takes care of the often nonphysical values of the ghost
    cells. Note that phi is not a variable but a parameter calculated with a
    function over a domain. 
    
    Make sure that phi is calculated by cellValuesWithBoundaries, usually
    (but not necessarily) in combination with celleval. Otherwise, do not use 
    this function as it leads to wrong values at the boundaries.
    
    It checks for the availability of the ghost variables and use them, 
    otherwise estimate them, assuming a zero gradient  on the boundaries.
    
    Note: I'm not happy with this implementation but it was the fastest 
    solution that came into my mind while onboard the Geilo-Oslo train.
    I have to find a better way to do this. The problem is that it is almost 
    always used for a cell variable calculated as f(phi) so having a boundary
    condition does not really help. I have to think about it.

    parameters
    ----------
    phi : CellVariable
        The cell variable for which the gradient is calculated.
    
    Returns
    -------
    FaceVariable
        The gradient of the cell variable.
    
    Examples
    --------
    >>> import pyfvtool as pf
    >>> import numpy as np
    >>> m = pf.Grid1D(10, 1.0)
    >>> phi = pf.CellVariable(m, 1.0)
    >>> sin_phi = pf.celleval(np.sin, sw.BC2GhostCells())
    >>> gradPhi = pf.gradientTermFixedBC(sin_phi)
    """
    faceGrad = gradientTerm(phi)
    if issubclass(type(phi.domain), Grid1D):
        faceGrad._xvalue[0] = 2*faceGrad._xvalue[0]
        faceGrad._xvalue[-1] = 2*faceGrad._xvalue[-1]
    elif issubclass(type(phi.domain), Grid2D):
        faceGrad._xvalue[0, :] = 2*faceGrad._xvalue[0, :]
        faceGrad._xvalue[-1, :] = 2*faceGrad._xvalue[-1, :]
        faceGrad._yvalue[:, 0] = 2*faceGrad._yvalue[:, 0]
        faceGrad._yvalue[:, -1] = 2*faceGrad._yvalue[:, -1]
    elif issubclass(type(phi.domain), Grid3D):
        faceGrad._xvalue[0, :, :] = 2*faceGrad._xvalue[0, :, :]
        faceGrad._xvalue[-1, :, :] = 2*faceGrad._xvalue[-1, :, :]
        faceGrad._yvalue[:, 0, :] = 2*faceGrad._yvalue[:, 0, :]
        faceGrad._yvalue[:, -1, :] = 2*faceGrad._yvalue[:, -1, :]
        faceGrad._zvalue[:, :, 0] = 2*faceGrad._zvalue[:, :, 0]
        faceGrad._zvalue[:, :, -1] = 2*faceGrad._zvalue[:, :, -1]
    return faceGrad