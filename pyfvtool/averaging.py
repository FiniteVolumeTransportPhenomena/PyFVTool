"""This module contains functions for averaging node values to face values.

functions:
    cell_size_array
    
    linearMean - linear interpolation of adjacent node values to face positions
    
    arithmeticMean - interpolate node values to face values via arithmetic averaging
    
    geometricMean - interpolate node values to face values via geometric averaging

    harmonicMean - interpolate node values to face values via harmonic averaging
    
    upwindMean - interpolate node values to face value based on local background velocity at face position
    
    tvdMean - NotImplemented
"""
import numpy as np


from .mesh import MeshStructure
from .mesh import Mesh1D, Mesh2D, Mesh3D
from .cell import CellVariable
from .face import FaceVariable

def cell_size_array(m: MeshStructure):
    if issubclass(type(m), Mesh1D):
       dx = m.cellsize.x
       return dx
    elif issubclass(type(m), Mesh2D):
        dx = m.cellsize.x[:,np.newaxis]
        dy = m.cellsize.y[np.newaxis,:]
        return dx, dy
    elif issubclass(type(m), Mesh3D):
        dx = m.cellsize.x[:,np.newaxis,np.newaxis]
        dy = m.cellsize.y[np.newaxis,:,np.newaxis]
        dz = m.cellsize.z[np.newaxis,np.newaxis,:]
        return dx, dy, dz

    
def linearMean(phi: CellVariable) -> FaceVariable:
    """
    Linearly interpolate a mesh-variable defined on mesh-nodes to mesh-faces.   
    
    This function gets the value of the field variable phi defined over the MeshStructure and calculates the value on the cell faces via linear interpolation, for a uniform Mesh.  

    Parameters
    ----------
    phi : {CellVariable object}
        Variable defined at mesh-nodes

    Returns
    -------
    out : {FaceVariable object}
        Interpolated node values at face positions

    See Also
    --------
    arithmeticMean : interpolate node values to face values via arithmetic averaging of node values
    geometricMean : interpolate node values to face values via geometric averaging of node values
    harmonicMean : interpolate node values to face values via harmonic averaging of node values

    Notes
    -----

    Examples
    --------
    >>>     
    
    """    
    if issubclass(type(phi.domain), Mesh1D):
        dx = cell_size_array(phi.domain)
        return FaceVariable(phi.domain,
                     (dx[1:]*phi.value[0:-1]+dx[0:-1] *
                      phi.value[1:])/(dx[1:]+dx[0:-1]),
                     np.array([]),
                     np.array([]))
    elif issubclass(type(phi.domain), Mesh2D):
        dx, dy = cell_size_array(phi.domain)
        return FaceVariable(phi.domain,
                     (dx[1:]*phi.value[0:-1, 1:-1]+dx[0:-1] *
                      phi.value[1:, 1:-1])/(dx[1:]+dx[0:-1]),
                     (dy[:,1:]*phi.value[1:-1, 0:-1]+dy[:,0:-1] *
                      phi.value[1:-1, 1:])/(dy[:,1:]+dy[:,0:-1]),
                     np.array([]))
    elif issubclass(type(phi.domain), Mesh3D):
        dx, dy, dz = cell_size_array(phi.domain)
        return FaceVariable(phi.domain,
                     (dx[1:]*phi.value[0:-1, 1:-1, 1:-1]+dx[0:-1] *
                      phi.value[1:, 1:-1, 1:-1])/(dx[1:]+dx[0:-1]),
                     (dy[:,1:]*phi.value[1:-1, 0:-1, 1:-1]+dy[:,0:-1] *
                      phi.value[1:-1, 1:, 1:-1])/(dy[:,0:-1]+dy[:,1:]),
                     (dz[:,:,1:]*phi.value[1:-1, 1:-1, 0:-1]+dz[:,:,0:-1] *
                      phi.value[1:-1, 1:-1, 1:])/(dz[:,:,0:-1]+dz[:,:,1:]))

    
def arithmeticMean(phi: CellVariable):
    """
    Interpolate a mesh-variable defined on mesh-nodes to mesh-faces by arithmetic averaging adjacent node values.   
    
    This function gets the value of the field variable phi defined over the MeshStructure and calculates the arithmetic average on the cell faces, for a uniform Mesh. 
            phi[face] = 1/#_adjacent_nodes * \sum(phi[adjacent nodes]) 
                --> 
            phi[face] = 0.5 * (phi[face-0.5*dx] + phi[face+0.5*dx])
                --> 
            phi[face] =  (cell_width[-]*phi[-] + cell_width[+]*phi[+])/(cell_width[-]+cell_width[+])
            

    Parameters
    ----------
    phi : {CellVariable object}
        Variable defined at mesh-nodes

    Returns
    -------
    out : {FaceVariable object}
        Arithmetic average of node values

    See Also
    --------
    linearMean : estimate node values to face values via linear-interpolation of node values
    geometricMean : interpolate node values to face values via geometric averaging of node values
    harmonicMean : interpolate node values to face values via harmonic averaging of node values

    Notes
    -----
    The arithmetic mean emphasizes (is sensitive to) outliers with large values.
    
    Weighting the average by the cell length in the appropriate dimension is necessary for multi-dimensional grids, and is a necessary step in the extension to non-uniform grid spacing in 1-dimension. 

    Examples
    --------
    >>>     
    
    """
    if issubclass(type(phi.domain), Mesh1D):
        dx = cell_size_array(phi.domain)
        return FaceVariable(phi.domain,
            (dx[0:-1]*phi.value[0:-1]+dx[1:]*phi.value[1:])/(dx[1:]+dx[0:-1]),
            np.array([]),
            np.array([]))
    elif issubclass(type(phi.domain), Mesh2D):
        dx, dy = cell_size_array(phi.domain)
        return FaceVariable(phi.domain,
            (dx[0:-1]*phi.value[0:-1,1:-1]+dx[1:]*phi.value[1:,1:-1])/(dx[1:]+dx[0:-1]),
            (dy[:,0:-1]*phi.value[1:-1,0:-1]+dy[:,1:]*phi.value[1:-1,1:])/(dy[:,1:]+dy[:,0:-1]),
            np.array([]))
    elif issubclass(type(phi.domain), Mesh3D):
        dx, dy, dz = cell_size_array(phi.domain)
        return FaceVariable(phi.domain,
            (dx[0:-1]*phi.value[0:-1,1:-1,1:-1]+dx[1:]*phi.value[1:,1:-1,1:-1])/(dx[1:]+dx[0:-1]),
            (dy[:,0:-1]*phi.value[1:-1,0:-1,1:-1]+dy[:,1:]*phi.value[1:-1,1:,1:-1])/(dy[:,0:-1]+dy[:,1:]),
            (dz[:,:,0:-1]*phi.value[1:-1,1:-1,0:-1]+dz[:,:,1:]*phi.value[1:-1,1:-1,1:])/(dz[:,:,0:-1]+dz[:,:,1:]))

    
def geometricMean(phi: CellVariable):
    """
    Interpolate a mesh-variable defined on mesh-nodes to mesh-faces by geometric averaging adjacent node values.   
    
    This function gets the value of the field variable phi defined over the MeshStructure and calculates the geometric average on the cell faces, for a uniform Mesh. 
            phi[face] = ( \prod(phi_nodes) )^(1/#_adjacent_nodes)
                --> 
            phi[face] = (phi[face-0.5*dx]*phi[face+0.5*dx])^(0.5)
                --> 
            phi[face] =  exp(
                    ( cell_width[-] * ln(phi[-]) + cell_width[+] * ln(phi[+]) ) / ( cell_width[-] + cell_width[+] )
                )

    Parameters
    ----------
    phi : {CellVariable object}
        Variable defined at mesh-nodes

    Returns
    -------
    out : {FaceVariable object}
        Geometric mean of node values at face positions

    See Also
    --------
    linearMean : estimate node values to face values via linear-interpolation of node values    
    arithmeticMean : interpolate node values to face values via arithmetic averaging of node values
    harmonicMean : interpolate node values to face values via harmonic averaging of node values

    Notes
    -----
    The geometric mean is equivalent to the arithmetic mean on a log-scale. 
                phi_face = ( \prod(phi_nodes) )^(1/N_nodes)
                         = exp( ln( arithmeticMean(...) ) )

    It is always intermediate to the harmonic or arithmetic means, and tends to the central value of th eset by 'weight'.

    Weighting the average by the cell length in the appropriate dimension is necessary for multi-dimensional grids, and is a necessary step in the extension to non-uniform grid spacing in 1-dimension. 

    Examples
    --------
    >>>     
    
    """
    if issubclass(type(phi.domain), Mesh1D):
        dx = cell_size_array(phi.domain)
        n= phi.domain.dims[0]
        phix=np.zeros(n+1)
        for i in np.arange(0,n+1):
            if phi.value[i]==0.0 or phi.value[i+1]==0.0:
                phix[i]=0.0
            else:
                phix[i]=np.exp((dx[i]*np.log(phi.value[i])+dx[i+1]*np.log(phi.value[i+1]))/(dx[i+1]+dx[i]))
        return FaceVariable(phi.domain,
            phix,
            np.array([]),
            np.array([]))
    elif issubclass(type(phi.domain), Mesh2D):
        dx, dy = cell_size_array(phi.domain)
        return FaceVariable(phi.domain,
            np.exp((dx[0:-1]*np.log(phi.value[0:-1,1:-1])+dx[1:]*np.log(phi.value[1:,1:-1]))/(dx[1:]+dx[0:-1])),
            np.exp((dy[:,0:-1]*np.log(phi.value[1:-1,0:-1])+dy[:,1:]*np.log(phi.value[1:-1,1:]))/(dy[:,1:]+dy[:,0:-1])),
            np.array([]))
    elif issubclass(type(phi.domain), Mesh3D):
        dx, dy, dz = cell_size_array(phi.domain)
        return FaceVariable(phi.domain,
            np.exp((dx[0:-1]*np.log(phi.value[0:-1,1:-1,1:-1])+dx[1:]*np.log(phi.value[1:,1:-1,1:-1]))/(dx[1:]+dx[0:-1])),
            np.exp((dy[:,0:-1]*np.log(phi.value[1:-1,0:-1,1:-1])+dy[:,1:]*np.log(phi.value[1:-1,1:,1:-1]))/(dy[:,0:-1]+dy[:,1:])),
            np.exp((dz[:,:,0:-1]*np.log(phi.value[1:-1,1:-1,0:-1])+dz[:,:,1:]*np.log(phi.value[1:-1,1:-1,1:]))/(dz[:,:,0:-1]+dz[:,:,1:])))
    
    

def harmonicMean(phi: CellVariable):
    """
    Interpolate a mesh-variable defined on mesh-nodes to mesh-faces by harmonic averaging adjacent node values.   
    
    This function gets the value of the field variable phi defined over the MeshStructure and calculates the harmonic average on the cell faces, for a uniform Mesh. 
            phi[face] = \sum(1.0/phi_nodes)^(-1)
                --> 
            phi[face] = (1/phi[face-0.5*dx] + 1/phi[face+0.5*dx])^(-1)
                --> 
            phi[face] = ( cell_width[-] + cell_width[+] ) / (  cell_width[-]/phi[-] + cell_width[+]/phi[+]  ) 

    Parameters
    ----------
    phi : {CellVariable object}
        Variable defined at mesh-nodes

    Returns
    -------
    out : {FaceVariable object}
        Harmonic average of node values

    See Also
    --------
    linearMean : estimate node values to face values via linear-interpolation of node values    
    geometricMean : interpolate node values to face values via geometric averaging of node values
    arithmeticMean : interpolate node values to face values via arithmetic averaging of node values

    Notes
    -----
    The harmonic mean emphasizes (is sensitive to) outliers with small values.

    Examples
    --------
    >>>     
    
    """
    
    # calculates the average values of a cell variable. The output is a
    # face variable

    if issubclass(type(phi.domain), Mesh1D):
        dx = cell_size_array(phi.domain)
        n=phi.domain.dims[0]
        phix=np.zeros(n+1)
        for i in np.arange(0,n+1):
            if phi.value[i]==0.0 or phi.value[i+1]==0.0:
                phix[i]=0.0
            else:
                phix[i]=(dx[i+1]+dx[i])/(dx[i+1]/phi.value[i+1]+dx[i]/phi.value[i])
        return FaceVariable(phi.domain,
            phix,
            np.array([]),
            np.array([]))
    elif issubclass(type(phi.domain), Mesh2D):
        dx, dy = cell_size_array(phi.domain)
        return FaceVariable(phi.domain,
            phi.value[1:,1:-1]*phi.value[0:-1,1:-1]*(dx[1:]+dx[0:-1])/(dx[1:]*phi.value[0:-1,1:-1]+dx[0:-1]*phi.value[1:,1:-1]),
            phi.value[1:-1,1:]*phi.value[1:-1,0:-1]*(dy[:,1:]+dy[:,0:-1])/(dy[:,1:]*phi.value[1:-1,0:-1]+dy[:,0:-1]*phi.value[1:-1,1:]),
            np.array([]))
    elif issubclass(type(phi.domain), Mesh3D):
        dx, dy, dz = cell_size_array(phi.domain)
        return FaceVariable(phi.domain,
            phi.value[1:,1:-1,1:-1]*phi.value[0:-1,1:-1,1:-1]*(dx[1:]+dx[0:-1])/(dx[1:]*phi.value[0:-1,1:-1,1:-1]+dx[0:-1]*phi.value[1:,1:-1,1:-1]),
            phi.value[1:-1,1:,1:-1]*phi.value[1:-1,0:-1,1:-1]*(dy[:,0:-1]+dy[:,1:])/(dy[:,1:]*phi.value[1:-1,0:-1,1:-1]+dy[:,0:-1]*phi.value[1:-1,1:,1:-1]),
            phi.value[1:-1,1:-1,1:]*phi.value[1:-1,1:-1,0:-1]*(dz[:,:,0:-1]+dz[:,:,1:])/(dz[:,:,1:]*phi.value[1:-1,1:-1,0:-1]+dz[:,:,0:-1]*phi.value[1:-1,1:-1,1:]))
    
    
def upwindMean(phi: CellVariable, u: FaceVariable):
    """
    Interpolate a mesh-variable defined on mesh-nodes to mesh-faces by matching the node value carried by the flow.   
    
    This function gets the value of the field variable phi defined over the MeshStructure and determines the value on the cell faces by applying a zeroeth order hold in the direction of the flow. 
            phi[face] = phi[adjacent node from flow]

    Parameters
    ----------
    phi : {CellVariable object}
        Variable defined at mesh-nodes
        
    u : {FaceVariable object}
        Flow-velocity at mesh-faces    

    Returns
    -------
    out : {FaceVariable object}
        Zeroeth order hold in flow direction

    See Also
    --------
    linearMean : estimate node values to face values via linear-interpolation of node values    
    arithmeticMean : interpolate node values to face values via arithmetic averaging of node values    
    geometricMean : interpolate node values to face values via geometric averaging of node values
    harmonicMean : interpolate node values to face values via harmonic averaging of node values

    Notes
    -----
    The upwind mean determines the direction of information flow based on the current local value of the background flow.

    Examples
    --------
    >>>     
    
    """    
    
    # calculates the average values of a cell variable. The output is a
    # face variable
    # TBD: needs to be fixed. It must be a linear mean, adjusted for velocity
    # currently, it assumes a uniform mesh.
    phi_tmp = np.copy(phi.value)
    if issubclass(type(phi.domain), Mesh1D):
        ux = u.xvalue
        # assign the value of the left boundary to the left ghost cell
        phi_tmp[0] = 0.5*(phi.value[0]+phi.value[1])
        # assign the value of the right boundary to the right ghost cell
        phi_tmp[-1] = 0.5*(phi.value[-1]+phi.value[-2])
        return FaceVariable(phi.domain,
            (ux>0.0)*phi_tmp[0:-1]+(ux<0.0)*phi_tmp[1:]+
            0.5*(ux==0)*(phi.value[0:-1]+phi.value[1:]),
            np.array([]),
            np.array([]))
    elif issubclass(type(phi.domain), Mesh2D):
        ux = u.xvalue
        uy = u.yvalue
        # assign the value of the left boundary to the left ghost cells
        phi_tmp[0,:] = 0.5*(phi.value[0,:]+phi.value[1,:])
        # assign the value of the right boundary to the right ghost cells
        phi_tmp[-1,:] = 0.5*(phi.value[-1,:]+phi.value[-2,:])
        # assign the value of the bottom boundary to the bottom ghost cells
        phi_tmp[:,0] = 0.5*(phi.value[:,0]+phi.value[:,1])
        # assign the value of the top boundary to the top ghost cells
        phi_tmp[:,-1] = 0.5*(phi.value[:,-1]+phi.value[:,-2])
        return FaceVariable(phi.domain,
            (ux>0.0)*phi_tmp[0:-1,1:-1]+
            (ux<0.0)*phi_tmp[1:,1:-1]+
            0.5*(ux==0.0)*(phi.value[0:-1,1:-1]+phi.value[1:,1:-1]),
            (uy>0.0)*phi_tmp[1:-1,0:-1]+
            (uy<0.0)*phi_tmp[1:-1,1:]+
            0.5*(uy==0.0)*(phi.value[1:-1,0:-1]+phi.value[1:-1,1:]),
            np.array([]))
    elif issubclass(type(phi.domain), Mesh3D):
        ux = u.xvalue
        uy = u.yvalue
        uz = u.zvalue
        # assign the value of the left boundary to the left ghost cells
        phi_tmp[0,:,:] = 0.5*(phi.value[0,:,:]+phi.value[1,:,:])
        # assign the value of the right boundary to the right ghost cells
        phi_tmp[-1,:,:] = 0.5*(phi.value[-1,:,:]+phi.value[-2,:,:])
        # assign the value of the bottom boundary to the bottom ghost cells
        phi_tmp[:,0,:] = 0.5*(phi.value[:,0,:]+phi.value[:,1,:])
        # assign the value of the top boundary to the top ghost cells
        phi_tmp[:,-1,:] = 0.5*(phi.value[:,-1,:]+phi.value[:,-2,:])
        # assign the value of the back boundary to the back ghost cells
        phi_tmp[:,:,0] = 0.5*(phi.value[:,:,0]+phi.value[:,:,1])
        # assign the value of the front boundary to the front ghost cells
        phi_tmp[:,:,-1] = 0.5*(phi.value[:,:,-1]+phi.value[:,:,-2])
        return FaceVariable(phi.domain,
            (ux>0.0)*phi_tmp[0:-1,1:-1,1:-1]+
            (ux<0.0)*phi_tmp[1:,1:-1,1:-1]+
            0.5*(ux==0.0)*(phi.value[0:-1,1:-1,1:-1]+phi.value[1:,1:-1,1:-1]),
            (uy>0.0)*phi_tmp[1:-1,0:-1,1:-1]+
            (uy<0.0)*phi_tmp[1:-1,1:,1:-1]+
            0.5*(uy==0.0)*(phi.value[1:-1,0:-1,1:-1]+phi.value[1:-1,1:,1:-1]),
            (uz>0.0)*phi_tmp[1:-1,1:-1,0:-1]+
            (uz<0.0)*phi_tmp[1:-1,1:-1,1:]+
            0.5*(uz==0.0)*(phi.value[1:-1,1:-1,0:-1]+phi.value[1:-1,1:-1,1:]))


# ================== TVD averaging scheme ==================
def tvdMean(phi: CellVariable, u: FaceVariable, FL):
    """
    Interpolate a mesh-variable defined on mesh-nodes to mesh-faces by matching the node value carried by the flow on a grid with a Total Variation Diminishing (TVD) discretization.   
    
    This function gets the value of the field variable phi defined over the MeshStructure and determines the value on the cell faces by applying a flow-weighted average / including a linearization of the local flow (first order hold in the direction of the flow). 

    Parameters
    ----------
    phi : {CellVariable object}
        Variable defined at mesh-nodes
        
    u : {FaceVariable object}
        Flow-velocity at mesh-faces    

    Returns
    -------
    out : {FaceVariable object}
        Interpolated value of phi at the mesh-faces

    See Also
    --------
    upwindMean :  determines the direction of information flow based on the current local value of the background flow.

    Notes
    -----
    The TVD mean determines the direction of information flow based on a local linearization of the current local value of the background velocity. s. https://en.wikipedia.org/wiki/Total_variation_diminishing

    Examples
    --------
    >>>     
    
    """        
    raise Exception("tvdMean is not implemented yet!")
#     # u is a face variable
#     # phi is a cell variable

#     # a def to avoid division by zero
#     eps1 = 1.0e-20
#     fsign(phi_in) = (abs(phi_in)>=eps1)*phi_in+eps1*(phi_in==0.0)+eps1*(abs(phi_in)<eps1)*sign(phi_in)



#     if issubclass(type(phi.domain), Mesh1D):
#         # extract data from the mesh structure
#         Nx = u.domain.dims[0]
#         dx = 0.5*(u.domain.cellsize.x[0:-1]+u.domain.cellsize.x[1:])
#         phi_p = np.zeros(Float64, Nx+1)
#         phi_m = np.zeros(Float64, Nx+1)

#         # extract the velocity data
#         ux = u.xvalue

#         # calculate the upstream to downstream gradient ratios for u>0 (+ ratio)
#         dphi_p = (phi.value[1:Nx+2]-phi.value[0:Nx+1])/dx
#         rp = dphi_p[0:-1]/fsign(dphi_p[1:])
#         phi_p[1:Nx+1] = phi.value[1:Nx+1]+0.5*FL(rp)*(phi.value[2:Nx+2]-phi.value[1:Nx+1])
#         phi_p[0] = (phi.value[0]+phi.value[2])/2.0 # left boundary

#         # calculate the upstream to downstream gradient ratios for u<0 (- ratio)
#         rm = dphi_p[1:]/fsign(dphi_p[0:-1])
#         phi_m[0:Nx] = phi.value[1:Nx+1]+0.5*FL(rm)*(phi.value[0:Nx]-phi.value[1:Nx+1])
#         phi_m[Nx+1] = (phi.value[-1]+phi.value[-1])/2.0 # right boundary

#         return FaceVariable(phi.domain,
#             (ux>0.0)*phi_p+(ux<0.0)*phi_m+
#             0.5*(ux==0)*(phi.value[0:-1]+phi.value[1:]),
#             np.array([]),
#             np.array([]))
#     elif issubclass(type(phi.domain), Mesh2D):
#         # extract data from the mesh structure
#         Nx = u.domain.dims[0]
#         Ny = u.domain.dims[2]
#         dx=0.5*(u.domain.cellsize.x[0:-1]+u.domain.cellsize.x[1:])
#         dy=np.zeros( 1, Ny+1)
#         dy[:]=0.5*(u.domain.cellsize.y[0:-1]+u.domain.cellsize.y[1:])
#         phi_p = np.zeros(Float64, Nx+1)
#         phi_m = np.zeros(Float64, Nx+1)
#         phiX_p = np.zeros(Float64, Nx+1, Ny)
#         phiX_m = np.zeros(Float64, Nx+1,Ny)
#         phiY_p = np.zeros(Float64, Nx,Ny+1)
#         phiY_m = np.zeros(Float64, Nx,Ny+1)

#         # extract the velocity data
#         ux = u.xvalue
#         uy = u.yvalue

#         # calculate the upstream to downstream gradient ratios for u>0 (+ ratio)
#         # x direction
#         dphiX_p = (phi.value[1:Nx+2, 1:Ny+1]-phi.value[0:Nx+1, 1:Ny+1])/dx
#         rX_p = dphiX_p[0:-1,:]/fsign(dphiX_p[1:,:])
#         phiX_p[1:Nx+1,:] = phi.value[1:Nx+1, 1:Ny+1]+0.5*FL(rX_p)*
#         (phi.value[2:Nx+2,1:Ny+1]-phi.value[1:Nx+1, 1:Ny+1])
#         phiX_p[0, :] = (phi.value[0, 1:Ny+1]+phi.value[1, 1:Ny+1])/2.0  # left boundary
#         # y direction
#         dphiY_p = (phi.value[1:Nx+1, 1:Ny+2]-phi.value[1:Nx+1, 0:Ny+1])/dy
#         rY_p = dphiY_p[:,0:-1]/fsign(dphiY_p[:,1:])
#         phiY_p[:,1:Ny+1] = phi.value[1:Nx+1, 1:Ny+1]+0.5*FL(rY_p)*
#             (phi.value[1:Nx+1,2:Ny+2]-phi.value[1:Nx+1, 1:Ny+1])
#         phiY_p[:,1] = (phi.value[1:Nx+1,1]+phi.value[1:Nx+1,2])/2.0  # Bottom boundary

#         # calculate the upstream to downstream gradient ratios for u<0 (- ratio)
#         # x direction
#         rX_m = dphiX_p[1:,:]/fsign(dphiX_p[0:-1,:])
#         phiX_m[0:Nx,:] = phi.value[1:Nx+1, 1:Ny+1]+0.5*FL(rX_m)*
#             (phi.value[0:Nx, 1:Ny+1]-phi.value[1:Nx+1, 1:Ny+1])
#         phiX_m[Nx+1,:] = (phi.value[-1, 1:Ny+1]+phi.value[-1, 1:Ny+1])/2.0  # right boundary
#         # y direction
#         rY_m = dphiY_p[:,1:]/fsign(dphiY_p[:,0:-1])
#         phiY_m[:,0:Ny] = phi.value[1:Nx+1, 1:Ny+1]+0.5*FL(rY_m)*
#             (phi.value[1:Nx+1, 0:Ny]-phi.value[1:Nx+1, 1:Ny+1])
#         phiY_m[:, Ny+1] = (phi.value[1:Nx+1, -1]+phi.value[1:Nx+1, -1])/2.0  # top boundary

#         return FaceVariable(phi.domain,
#             (ux>0.0)*phiX_p+(ux<0.0)*phiX_m+
#                 0.5*(ux==0.0)*(phi.value[0:Nx+1,1:Ny+1]+phi.value[1:Nx+2,1:Ny+1]),
#             (uy>0.0)*phiY_p+(uy<0.0)*phiY_m+
#                 0.5*(uy==0.0)*(phi.value[1:Nx+1,0:Ny+1]+phi.value[1:Nx+1,1:Ny+2]),
#             np.array([]))

#     elif issubclass(type(phi.domain), Mesh3D):
#         # extract data from the mesh structure
#         Nx = u.domain.dims[0]
#         Ny = u.domain.dims[2]
#         Nz = u.domain.dims[3]
#         dx=0.5*(u.domain.cellsize.x[0:-1]+u.domain.cellsize.x[1:])
#         dy=np.zeros( 1, Ny+1)
#         dy[:]=0.5*(u.domain.cellsize.y[0:-1]+u.domain.cellsize.y[1:])
#         dz=np.zeros( 1, 1, Nz+1)
#         dz[:]=0.5*(u.domain.cellsize.z[0:-1]+u.domain.cellsize.z[1:])
#         # extract the velocity data
#         ux = u.xvalue
#         uy = u.yvalue
#         uz = u.zvalue

#         # define the tvd face vectors
#         phiX_p = np.zeros(Float64, Nx+1,Ny,Nz)
#         phiX_m = np.zeros(Float64, Nx+1,Ny,Nz)
#         phiY_p = np.zeros(Float64, Nx,Ny+1,Nz)
#         phiY_m = np.zeros(Float64, Nx,Ny+1,Nz)
#         phiZ_p = np.zeros(Float64, Nx,Ny,Nz+1)
#         phiZ_m = np.zeros(Float64, Nx,Ny,Nz+1)

#         # calculate the upstream to downstream gradient ratios for u>0 (+ ratio)
#         # x direction
#         dphiX_p = (phi.value[1:Nx+2, 1:Ny+1, 1:Nz+1]-phi.value[0:Nx+1, 1:Ny+1, 1:Nz+1])/dx
#         rX_p = dphiX_p[0:-1,:,:]/fsign(dphiX_p[1:,:,:])
#         phiX_p[1:Nx+1,:,:] = phi.value[1:Nx+1, 1:Ny+1, 1:Nz+1]+0.5*FL(rX_p)*
#             (phi.value[2:Nx+2,1:Ny+1,1:Nz+1]-phi.value[1:Nx+1,1:Ny+1,1:Nz+1])
#         phiX_p[0,:,:] = (phi.value[0,1:Ny+1,1:Nz+1]+phi.value[1,1:Ny+1,1:Nz+1])/2.0  # left boundary
#         # y direction
#         dphiY_p = (phi.value[1:Nx+1, 1:Ny+2, 1:Nz+1]-phi.value[1:Nx+1, 0:Ny+1, 1:Nz+1])/dy
#         rY_p = dphiY_p[:,0:-1,:]/fsign(dphiY_p[:,1:,:])
#         phiY_p[:,1:Ny+1,:] = phi.value[1:Nx+1, 1:Ny+1, 1:Nz+1]+0.5*FL(rY_p)*
#             (phi.value[1:Nx+1,2:Ny+2,1:Nz+1]-phi.value[1:Nx+1, 1:Ny+1,1:Nz+1])
#         phiY_p[:,1,:] = (phi.value[1:Nx+1,1,1:Nz+1]+phi.value[1:Nx+1,2,1:Nz+1])/2.0  # Bottom boundary
#         # z direction
#         dphiZ_p = (phi.value[1:Nx+1, 1:Ny+1, 1:Nz+2]-phi.value[1:Nx+1, 1:Ny+1, 0:Nz+1])/dz
#         rZ_p = dphiZ_p[:,:,0:-1]/fsign(dphiZ_p[:,:,1:])
#         phiZ_p[:,:,1:Nz+1] = phi.value[1:Nx+1, 1:Ny+1, 1:Nz+1]+0.5*FL(rZ_p)*
#             (phi.value[1:Nx+1,1:Ny+1,2:Nz+2]-phi.value[1:Nx+1,1:Ny+1,1:Nz+1])
#         phiZ_p[:,:,1] = (phi.value[1:Nx+1,1:Ny+1,1]+phi.value[1:Nx+1,1:Ny+1,2])/2.0  # Back boundary

#         # calculate the upstream to downstream gradient ratios for u<0 (- ratio)
#         # x direction
#         rX_m = dphiX_p[1:,:,:]/fsign(dphiX_p[0:-1,:,:])
#         phiX_m[0:Nx,:,:] = phi.value[1:Nx+1, 1:Ny+1, 1:Nz+1]+0.5*FL(rX_m)*
#             (phi.value[0:Nx, 1:Ny+1, 1:Nz+1]-phi.value[1:Nx+1, 1:Ny+1, 1:Nz+1])
#         phiX_m[Nx+1,:,:] = (phi.value[-1,1:Ny+1,1:Nz+1]+phi.value[-1,1:Ny+1,1:Nz+1])/2.0  # right boundary
#         # y direction
#         rY_m = dphiY_p[:,1:,:]/fsign(dphiY_p[:,0:-1,:])
#         phiY_m[:,0:Ny,:] = phi.value[1:Nx+1,1:Ny+1,1:Nz+1]+0.5*FL(rY_m)*
#             (phi.value[1:Nx+1,0:Ny,1:Nz+1]-phi.value[1:Nx+1,1:Ny+1,1:Nz+1])
#         phiY_m[:,Ny+1,:] = (phi.value[1:Nx+1, end,1:Nz+1]+phi.value[1:Nx+1, -1,1:Nz+1])/2.0  # top boundary
#         # z direction
#         rZ_m = dphiZ_p[:,:,1:]/fsign(dphiZ_p[:,:,0:-1])
#         phiZ_m[:,:,0:Nz] = phi.value[1:Nx+1,1:Ny+1,1:Nz+1]+0.5*FL(rZ_m)*
#             (phi.value[1:Nx+1,1:Ny+1,0:Nz]-phi.value[1:Nx+1,1:Ny+1,1:Nz+1])
#         phiZ_m[:,:,Nz+1] = (phi.value[1:Nx+1,1:Ny+1,-1]+phi.value[1:Nx+1,1:Ny+1,-1])/2.0  # front boundary

#         return FaceVariable(phi.domain,
#             (ux>0.0)*phiX_p+(ux<0.0)*phiX_m+
#                 0.5*(ux==0.0)*(phi.value[0:Nx+1,1:Ny+1,1:Nz+1]+phi.value[1:Nx+2,1:Ny+1,1:Nz+1]),
#             (uy>0.0)*phiY_p+(uy<0)*phiY_m+
#                 0.5*(uy==0.0)*(phi.value[1:Nx+1,0:Ny+1,1:Nz+1]+phi.value[1:Nx+1,1:Ny+2,1:Nz+1]),
#             (uz>0.0)*phiZ_p+(uz<0)*phiZ_m+
#                 0.5*(uz==0.0)*(phi.value[1:Nx+1,1:Ny+1,0:Nz+1]+phi.value[1:Nx+1,1:Ny+1,1:Nz+2]))
