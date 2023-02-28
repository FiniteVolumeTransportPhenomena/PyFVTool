import numpy as np
# from scipy.sparse import csr_array
# from scipy.sparse.linalg import spsolve
from .mesh import *
from .utilities import *
from .cell import *
from .face import *
import matplotlib.pyplot as plt


def visualizeCells(phi: CellVariable,
                   vmin=0.0,
                   vmax=0.0,
                   cmap="viridis",
                   shading="gouraud"):
    # copy of julia code that calls matplotlib
     if issubclass(type(phi.domain), Mesh1D):
          x = np.hstack([phi.domain.facecenters.x[0],
               phi.domain.cellcenters.x,
               phi.domain.facecenters.x[-1]])
          phi = np.hstack([0.5*(phi.value[0]+phi.value[1]),
               phi.value[1:-1],
               0.5*(phi.value[-2]+phi.value[-1])])
          plt.plot(x, phi)
          plt.show()
     elif (type(phi.domain) is Mesh2D) or (type(phi.domain) is MeshCylindrical2D):
          x = np.hstack([phi.domain.facecenters.x[0],
               phi.domain.cellcenters.x,
               phi.domain.facecenters.x[-1]])
          y = np.hstack([phi.domain.facecenters.y[0],
               phi.domain.cellcenters.y,
               phi.domain.facecenters.y[-1]])
          phi0 = np.copy(phi.value)
          phi0[:, 0] = 0.5*(phi0[:, 0]+phi0[:, 1])
          phi0[0, :] = 0.5*(phi0[0, :]+phi0[1, :])
          phi0[:, -1] = 0.5*(phi0[:, -1]+phi0[:, -2])
          phi0[-1, :] = 0.5*(phi0[-1, :]+phi0[-2, :])
          phi0[0, 0] = phi0[0, 1]
          phi0[0, -1] = phi0[0, -2]
          phi0[-1, 0] = phi0[-1, 1]
          phi0[-1, -1] = phi0[-1, -2]
          if vmin == 0.0 and vmax == 0.0:
               vmin = phi0.min()
               vmax = phi0.max()
          plt.pcolormesh(x, y, phi0.T, vmin=vmin, vmax=vmax, cmap=cmap, shading=shading)
          plt.show()
     elif (type(phi.domain) is MeshRadial2D):
          x = np.hstack([phi.domain.facecenters.x[0],
               phi.domain.cellcenters.x,
               phi.domain.facecenters.x[-1]])
          y = np.hstack([phi.domain.facecenters.y[0],
               phi.domain.cellcenters.y,
               phi.domain.facecenters.y[-1]])
          phi0 = np.copy(phi.value)
          phi0[:, 0] = 0.5*(phi0[:, 0]+phi0[:, 1])
          phi0[0, :] = 0.5*(phi0[0, :]+phi0[1, :])
          phi0[:, -1] = 0.5*(phi0[:, -1]+phi0[:, -2])
          phi0[-1, :] = 0.5*(phi0[-1, :]+phi0[-2, :])
          phi0[0, 0] = phi0[0, 1]
          phi0[0, -1] = phi0[0, -2]
          phi0[-1, 0] = phi0[-1, 1]
          phi0[-1, -1] = phi0[-1, -2]
          plt.subplot(111, polar="true")
          plt.pcolor(y, x, phi0)
          plt.show()
     elif (type(phi.domain) is Mesh3D):
          Nx, Ny, Nz = phi.domain.dims
          x = np.hstack([phi.domain.facecenters.x[0], phi.domain.cellcenters.x, phi.domain.facecenters.x[-1]])[:, np.newaxis, np.newaxis]
          y = np.hstack([phi.domain.facecenters.y[0], phi.domain.cellcenters.y, phi.domain.facecenters.y[-1]])[np.newaxis, :, np.newaxis]
          z = np.hstack([phi.domain.facecenters.z[0], phi.domain.cellcenters.z, phi.domain.facecenters.z[-1]])[np.newaxis, np.newaxis, :]

          phi0 = np.copy(phi.value)
          phi0[:,0,:]=0.5*(phi0[:,0,:]+phi0[:,1,:])
          phi0[:,-1,:]=0.5*(phi0[:,-2,:]+phi0[:,-1,:])
          phi0[:,:,0]=0.5*(phi0[:,:,0]+phi0[:,:,0])
          phi0[:,:,-1]=0.5*(phi0[:,:,-2]+phi0[:,:,-1])
          phi0[0,:,:]=0.5*(phi0[1,:,:]+phi0[2,:,:])
          phi0[-1,:,:]=0.5*(phi0[-2,:,:]+phi0[-1,:,:])

          vmin = np.min(phi0)
          vmax = np.max(phi0)
          mynormalize = lambda a:((a - vmin)/(vmax-vmin))

          a= np.ones((Nx+2,Ny+2,Nz+2))
          X = x*a
          Y = y*a
          Z = z*a

          fig = plt.figure()
          ax = fig.add_subplot(111, projection = "3d")
          # r = linspace(1.25, 1.25, 50)
          # p = linspace(0, 2π, 50)
          # R = repmat(r, 1, 50)
          # P = repmat(p', 50, 1)
          # Zc = rand(50, 50) # (P.^2-1).^2
          # Z = repmat(linspace(0, 2, 50), 1, 50)
          # X, Y = R.*cos.(P), R.*sin.(P)
          ax.plot_surface(X[0,:,:], Y[0,:,:], Z[0,:,:], facecolors=plt.cm.viridis(mynormalize(phi0[0,:,:])), alpha=0.8)
          ax.plot_surface(X[-1,:,:], Y[-1,:,:], Z[-1,:,:], facecolors=plt.cm.viridis(mynormalize(phi0[-1,:,:])), alpha=0.8)
          ax.plot_surface(X[:,0,:], Y[:,0,:], Z[:,0,:], facecolors=plt.cm.viridis(mynormalize(phi0[:,0,:])), alpha=0.8)
          ax.plot_surface(X[:,-1,:], Y[:,-1,:], Z[:,-1,:], facecolors=plt.cm.viridis(mynormalize(phi0[:,-1,:])), alpha=0.8)
          ax.plot_surface(X[:,:,0], Y[:,:,0], Z[:,:,0], facecolors=plt.cm.viridis(mynormalize(phi0[:,:,0])), alpha=0.8)
          ax.plot_surface(X[:,:,-1], Y[:,:,-1], Z[:,:,-1], facecolors=plt.cm.viridis(mynormalize(phi0[:,:,-1])), alpha=0.8)
          plt.show()

     elif (type(phi.domain) is MeshCylindrical3D):
          r = np.hstack([phi.domain.facecenters.x[0], phi.domain.cellcenters.x,
                    phi.domain.facecenters.x[-1]])[:, np.newaxis, np.newaxis]
          theta = np.hstack([phi.domain.facecenters.y[0], phi.domain.cellcenters.y,
                    phi.domain.facecenters.y[-1]])[np.newaxis, :, np.newaxis]
          z = np.hstack([phi.domain.facecenters.z[0], phi.domain.cellcenters.z,
                    phi.domain.facecenters.z[-1]])[np.newaxis, np.newaxis, :]
          x = r*np.cos(theta)
          y = r*np.sin(theta)

          phi0 = np.copy(phi.value)
          phi0[:, 0, :] = 0.5*(phi0[:, 0, :]+phi0[:, 1, :])
          phi0[:, -1, :] = 0.5*(phi0[:, -2, :]+phi0[:, -1, :])
          phi0[:, :, 0] = 0.5*(phi0[:, :, 0]+phi0[:, :, 0])
          phi0[:, :, -1] = 0.5*(phi0[:, :, -2]+phi0[:, :, -1])
          phi0[0, :, :] = 0.5*(phi0[1, :, :]+phi0[2, :, :])
          phi0[-1, :, :] = 0.5*(phi0[-2, :, :]+phi0[-1, :, :])
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
                         facecolors=plt.cm.viridis(mynormalize(phi0[:, 0, :])), alpha=alfa)
          ax.plot_surface(X[:, int(Ny/2)+1, :], Y[:, int(Ny/2)+1, :], Z[:, int(Ny/2)+1, :],
                         facecolors=plt.cm.viridis(mynormalize(phi0[:, int(Ny/2)+1, :])), alpha=alfa)
          ax.plot_surface(X[:, :, 0], Y[:, :, 0], Z[:, :, 0],
                         facecolors=plt.cm.viridis(mynormalize(phi0[:, :, 0])), alpha=alfa)
          ax.plot_surface(X[:, :, 0], Y[:, :, 0], Z[:, :, 0],
                         facecolors=plt.cm.viridis(mynormalize(phi0[:, :, 0])), alpha=alfa)
          ax.plot_surface(X[:, :, int(Nz/2)], Y[:, :, int(Nz/2)], Z[:, :, int(Nz/2)],
                         facecolors=plt.cm.viridis(mynormalize(phi0[:, :, int(Nz/2)])), alpha=alfa)
          ax.plot_surface(X[:, :, -1], Y[:, :, -1], Z[:, :, -1],
                         facecolors=plt.cm.viridis(mynormalize(phi0[:, :, -1])), alpha=alfa)
          plt.show()
