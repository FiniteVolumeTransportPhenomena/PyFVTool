import numpy as np

from .mesh import Grid1D, Grid2D, Grid3D
from .mesh import CylindricalGrid2D
from .mesh import PolarGrid2D, CylindricalGrid3D, SphericalGrid3D
from .cell import CellVariable
from warnings import warn

import matplotlib.pyplot as plt



def visualizeCells(phi: CellVariable,
                   vmin = None,
                   vmax = None,
                   cmap = "viridis",
                   shading = "gouraud",
                   show = True):
    """
    Visualize the cell variable in a suitable graph.
    
    Parameters
    ----------
    phi: CellVariable
         Cell variable to be visualized
    vmin: float
         Minimum value of the colormap (2D and 3D plots)
    vmax: float
         Maximum value of the colormap (2D and 3D plots)
    cmap: str
         Colormap (2D and 3D plots)
    shading: str
         Shading method (2D and 3D plots)
    show: bool
         If True (default), call plt.show() at the end. Set to False
         to defer showing, e.g. to further customize the plot before
         displaying it, or (in the case of 1D plots) to superpose
         several curves in the same graph.

    Returns
    -------
    fig: matplotlib.figure.Figure
    ax: matplotlib.axes.Axes
         The figure and axes on which the variable was plotted. For
         Grid1D, repeated calls superpose onto the current axes
         (via plt.gca()); note this assumes the current axes, if any,
         is a standard 2D rectilinear axes rather than a leftover
         polar/3D one from a prior plot.
    
    Examples
    --------
    >>> import pyfvtool as pf
    >>> m = pf.Grid1D(10, 1.0)
    >>> phi = pf.CellVariable(m, 1.0)
    >>> pf.visualizeCells(phi)
    """
    if isinstance(phi.domain, Grid1D):
        ax = plt.gca()
        fig = ax.figure
        x, phi0 = phi.plotprofile()
        ax.plot(x, phi0)

    elif (type(phi.domain) is Grid2D) or (type(phi.domain) is CylindricalGrid2D):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x, y, phi0 = phi.plotprofile()
        if vmin is None:
            vmin = phi0.min()
        if vmax is None:
            vmax = phi0.max()
        ax.pcolormesh(x, y, phi0.T,
                     vmin=vmin, vmax=vmax,
                     cmap=cmap, shading=shading)

    elif (type(phi.domain) is PolarGrid2D):
        x, y, phi0 = phi.plotprofile()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="polar")
        ax.pcolor(y, x, phi0)

    elif (type(phi.domain) is Grid3D):
        x, y, z, phi0 = phi.plotprofile()
        vmin = np.min(phi0)
        vmax = np.max(phi0)
        mynormalize = lambda a:((a - vmin)/(vmax-vmin))
        Nx, Ny, Nz = phi.domain.dims
        a = np.ones((Nx+2,Ny+2,Nz+2))
        X = x*a
        Y = y*a
        Z = z*a

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

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

    elif (type(phi.domain) is CylindricalGrid3D):
        r, theta, z, phi0 = phi.plotprofile()
        Nx, Ny, Nz = phi.domain.dims
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
        ax.plot_surface(X[:, :, int(Nz/2)], Y[:, :, int(Nz/2)], Z[:, :, int(Nz/2)],
                       facecolors=plt.cm.viridis(mynormalize(phi0[:, :, int(Nz/2)])),
                       alpha=alfa)
        ax.plot_surface(X[:, :, -1], Y[:, :, -1], Z[:, :, -1],
                       facecolors=plt.cm.viridis(mynormalize(phi0[:, :, -1])),
                       alpha=alfa)

    elif (type(phi.domain) is SphericalGrid3D):
        warn("SphericalGrid3D visualization is not working properly yet.")
        r, theta, PHI, phi0 = phi.plotprofile()
        Nx, Ny, Nz = phi.domain.dims
        x = r*np.sin(theta)*np.cos(PHI)
        y = r*np.sin(theta)*np.sin(PHI)
        z = r*np.cos(theta)
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
        ax.plot_surface(X[-1, :, :], Y[-1, :, :], Z[-1, :, :],
                       facecolors=plt.cm.viridis(mynormalize(phi0[-1, :, :])),
                       alpha=alfa)

    else:
        # just in case...
        raise ValueError('Unsupported mesh: '+str(type(phi.domain)))

    if show:
        plt.show()

    return fig, ax
