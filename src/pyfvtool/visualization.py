import numpy as np

from .mesh import Grid1D, Grid2D, Grid3D
from .mesh import CylindricalGrid2D
from .mesh import PolarGrid2D, CylindricalGrid3D
from .cell import CellVariable

import matplotlib.pyplot as plt



def visualizeCells(phi: CellVariable,
                   vmin = None,
                   vmax = None,
                   cmap = "viridis",
                   shading = "gouraud"):
    """
    Visualize the cell variable.
    
    Parameters
    ----------
    phi: CellVariable
         Cell variable to be visualized
    vmin: float
         Minimum value of the colormap
    vmax: float
         Maximum value of the colormap
    cmap: str
         Colormap
    shading: str
         Shading method
    
    Examples
    --------
    >>> import pyfvtool as pf
    >>> m = pf.Grid1D(10, 1.0)
    >>> phi = pf.CellVariable(m, 1.0)
    >>> pf.visualizeCells(phi)
    """
    if isinstance(phi.domain, Grid1D):
        x, phi0 = phi.plotprofile()
        plt.plot(x, phi0)
        # plt.show()

    elif (type(phi.domain) is Grid2D) or (type(phi.domain) is CylindricalGrid2D):
        x, y, phi0 = phi.plotprofile()
        if vmin is None:
            vmin = phi0.min()
        if vmax is None:
            vmax = phi0.max()
        plt.pcolormesh(x, y, phi0.T, 
                       vmin=vmin, vmax=vmax,
                       cmap=cmap, shading=shading)
        # plt.show()

    elif (type(phi.domain) is PolarGrid2D):
        x, y, phi0 = phi.plotprofile()
        plt.subplot(111, polar="true")
        plt.pcolor(y, x, phi0)
        # plt.show()

    elif (type(phi.domain) is Grid3D):
        x, y, z, phi0 = phi.plotprofile()
        vmin = np.min(phi0)
        vmax = np.max(phi0)
        mynormalize = lambda a:((a - vmin)/(vmax-vmin))
        Nx, Ny, Nz = phi.domain.dims
        a= np.ones((Nx+2,Ny+2,Nz+2))
        X = x*a
        Y = y*a
        Z = z*a

        fig = plt.figure()
        ax = fig.add_subplot(111, projection = "3d")

        ax.plot_surface(X[0,:,:], Y[0,:,:], Z[0,:,:],
                        facecolors=plt.cm.viridis(mynormalize(phi0[0,:,:])),
                        alpha=0.8)
        ax.plot_surface(X[-1,:,:], Y[-1,:,:], Z[-1,:,:],
                        facecolors=plt.cm.viridis(mynormalize(phi0[-1,:,:])),
                        alpha=0.8)
        ax.plot_surface(X[:,0,:], Y[:,0,:], Z[:,0,:],
                        facecolors=plt.cm.viridis(mynormalize(phi0[:,0,:])),
                        alpha=0.8)
        ax.plot_surface(X[:,-1,:], Y[:,-1,:], Z[:,-1,:],
                        facecolors=plt.cm.viridis(mynormalize(phi0[:,-1,:])),
                        alpha=0.8)
        ax.plot_surface(X[:,:,0], Y[:,:,0], Z[:,:,0],
                        facecolors=plt.cm.viridis(mynormalize(phi0[:,:,0])),
                        alpha=0.8)
        ax.plot_surface(X[:,:,-1], Y[:,:,-1], Z[:,:,-1],
                        facecolors=plt.cm.viridis(mynormalize(phi0[:,:,-1])),
                        alpha=0.8)
        # plt.show()

    elif (type(phi.domain) is CylindricalGrid3D):
        r, theta, z, phi0 = phi.plotprofile()
      
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        vmin = np.min(phi0)
        vmax = np.max(phi0)
        mynormalize = lambda a:((a - vmin)/(vmax-vmin))
        a = np.ones((Nx+2, Ny+2, Nz+2))
        X = x*a
        Y = y*a
        Z = z*a
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        alfa = 1.0
        ax.plot_surface(X[:, 0, :], Y[:, 0, :], Z[:, 0, :],
                       facecolors=plt.cm.viridis(mynormalize(phi0[:, 0, :])),
                       alpha=alfa)
        ax.plot_surface(X[:, int(Ny/2)+1, :], Y[:, int(Ny/2)+1, :], Z[:, int(Ny/2)+1, :],
                       facecolors=plt.cm.viridis(mynormalize(phi0[:, int(Ny/2)+1, :])),
                       alpha=alfa)
        ax.plot_surface(X[:, :, 0], Y[:, :, 0], Z[:, :, 0],
                       facecolors=plt.cm.viridis(mynormalize(phi0[:, :, 0])),
                       alpha=alfa)
        ax.plot_surface(X[:, :, 0], Y[:, :, 0], Z[:, :, 0],
                       facecolors=plt.cm.viridis(mynormalize(phi0[:, :, 0])),
                       alpha=alfa)
        ax.plot_surface(X[:, :, int(Nz/2)], Y[:, :, int(Nz/2)], Z[:, :, int(Nz/2)],
                       facecolors=plt.cm.viridis(mynormalize(phi0[:, :, int(Nz/2)])),
                       alpha=alfa)
        ax.plot_surface(X[:, :, -1], Y[:, :, -1], Z[:, :, -1],
                       facecolors=plt.cm.viridis(mynormalize(phi0[:, :, -1])),
                       alpha=alfa)
        # plt.show()
        
    else:
        # just in case...
        raise ValueError('Unsupported mesh: '+str(type(phi.domain)))
