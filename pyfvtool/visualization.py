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
    # elseif d == 3
    # Nx = phi.domain.dims[1]
    # Ny = phi.domain.dims[2]
    # Nz = phi.domain.dims[3]
    # x = [phi.domain.facecenters.x[1]
    #      phi.domain.cellcenters.x
    #      phi.domain.facecenters.x[end]]
    # y = zeros(1, Ny+2)
    # y[:] = [phi.domain.facecenters.y[1]
    #         phi.domain.cellcenters.y
    #         phi.domain.facecenters.y[end]]
    # z = zeros(1, 1, Nz+2)
    # z[:] = [phi.domain.facecenters.z[1]
    #         phi.domain.cellcenters.z
    #         phi.domain.facecenters.z[end]]

    # phi0 = Base.copy(phi.value)
    # phi0[:, 1, :] = 0.5*(phi0[:, 1, :]+phi0[:, 2, :])
    # phi0[:, end, :] = 0.5*(phi0[:, end-1, :]+phi0[:, end, :])
    # phi0[:, :, 1] = 0.5*(phi0[:, :, 1]+phi0[:, :, 1])
    # phi0[:, :, end] = 0.5*(phi0[:, :, end-1]+phi0[:, :, end])
    # phi0[1, :, :] = 0.5*(phi0[1, :, :]+phi0[2, :, :])
    # phi0[end, :, :] = 0.5*(phi0[end-1, :, :]+phi0[end, :, :])

    # vmin = minimum(phi0)
    # vmax = maximum(phi0)
    # mynormalize = a -> (a .- vmin)./(vmax-vmin)

    # a = ones(Nx+2, Ny+2, Nz+2)
    # X = x.*a
    # Y = y.*a
    # Z = z.*a

    # fig = figure()
    # ax = fig[:add_subplot](111, projection="3d")
    # # r = linspace(1.25, 1.25, 50)
    # # p = linspace(0, 2π, 50)
    # # R = repmat(r, 1, 50)
    # # P = repmat(p', 50, 1)
    # # Zc = rand(50, 50) # (P.^2-1).^2
    # # Z = repmat(linspace(0, 2, 50), 1, 50)
    # # X, Y = R.*cos.(P), R.*sin.(P)
    # ax[:plot_surface](X[1, :, :], Y[1, :, :], Z[1, :, :], facecolors=PyPlot.cm[:viridis](
    #     mynormalize(phi0[1, :, :])), alpha=0.8)
    # ax[:plot_surface](X[end, :, :], Y[end, :, :], Z[end, :, :], facecolors=PyPlot.cm[:viridis](
    #     mynormalize(phi0[end, :, :])), alpha=0.8)
    # ax[:plot_surface](X[:, 1, :], Y[:, 1, :], Z[:, 1, :], facecolors=PyPlot.cm[:viridis](
    #     mynormalize(phi0[:, 1, :])), alpha=0.8)
    # ax[:plot_surface](X[:, end, :], Y[:, end, :], Z[:, end, :], facecolors=PyPlot.cm[:viridis](
    #     mynormalize(phi0[:, end, :])), alpha=0.8)
    # ax[:plot_surface](X[:, :, 1], Y[:, :, 1], Z[:, :, 1], facecolors=PyPlot.cm[:viridis](
    #     mynormalize(phi0[:, :, 1])), alpha=0.8)
    # ax[:plot_surface](X[:, :, end], Y[:, :, end], Z[:, :, end], facecolors=PyPlot.cm[:viridis](
    #     mynormalize(phi0[:, :, end])), alpha=0.8)
    # # s=mayavis.pipeline[:scalar_field](X,Y,Z,phi0)
    # #
    # # mayavis.pipeline[:image_plane_widget](s, plane_orientation="x_axes", slice_index=0, vmin=vmin, vmax=vmax)
    # # mayavis.pipeline[:image_plane_widget](s, plane_orientation="y_axes", slice_index=0, vmin=vmin, vmax=vmax)
    # # mayavis.pipeline[:image_plane_widget](s, plane_orientation="z_axes", slice_index=0, vmin=vmin, vmax=vmax)
    # # mayavis.pipeline[:image_plane_widget](s, plane_orientation="z_axes", slice_index=floor(Integer,Nz/2.0), vmin=vmin, vmax=vmax)
    # # mayavis.outline()

    # #   # 6 surfaces
    # #   # surfaces 1,2 (x=x[1], x=x[end])
    # #   Y=repmat(y,1,Nz)
    # #   Z=repmat(z,1,Ny)
    # #   mayavis.mesh(x[1]*ones(Ny,Nz),Y,Z',scalars=squeeze(phi.value[2,2:end-1,2:end-1],1), vmin=vmin, vmax=vmax, opacity=0.8)
    # #   mayavis.mesh(x[end]*ones(Ny,Nz),Y,Z',scalars=squeeze(phi.value[end-1,2:end-1,2:end-1],1), vmin=vmin, vmax=vmax, opacity=0.8)
    # #
    # #   # surfaces 3,4 (y=y[1], y=y[end]
    # #   X = repmat(x,1,Nz)
    # #   Z = repmat(z,1,Nx)
    # #   mayavis.mesh(X,y[1]*ones(Nx,Nz),Z',scalars=squeeze(phi.value[2:end-1,2,2:end-1],2), vmin=vmin, vmax=vmax, opacity=0.8)
    # #   mayavis.mesh(X,y[end]*ones(Nx,Nz),Z',scalars=squeeze(phi.value[2:end-1,end-1,2:end-1],2), vmin=vmin, vmax=vmax, opacity=0.8)
    # #   mayavis.axes()
    # #
    # #   # surfaces 5,6 (z=z[1], z=z[end]
    # #   X = repmat(x,1,Ny)
    # #   Y = repmat(y,1,Nx)
    # #   mayavis.mesh(X,Y',z[1]*ones(Nx,Ny),scalars=phi.value[2:end-1,2:end-1,2], vmin=vmin, vmax=vmax, opacity=0.8)
    # #   mayavis.mesh(X,Y',z[end]*ones(Nx,Ny),scalars=phi.value[2:end-1,2:end-1,end-1], vmin=vmin, vmax=vmax, opacity=0.8)

    # # mayavis.colorbar()
    # # mayavis.axes()
    # # mshot= mayavis.screenshot()
    # # mayavis.show()
    # # return mshot

    # elseif d==3.2
    # Nx = phi.domain.dims[1]
    # Ny = phi.domain.dims[2]
    # Nz = phi.domain.dims[3]
    # r=[phi.domain.facecenters.x[1]; phi.domain.cellcenters.x; phi.domain.facecenters.x[end]]
    # theta = zeros(1,Ny+2)
    # theta[:]=[phi.domain.facecenters.y[1]; phi.domain.cellcenters.y; phi.domain.facecenters.y[end]]
    # z=zeros(1,1,Nz+2)
    # z[:]=[phi.domain.facecenters.z[1]; phi.domain.cellcenters.z; phi.domain.facecenters.z[end]]
    # a=ones(Nx+2,Ny+2,Nz+2)
    # R=r.*a
    # TH = theta.*a
    # Z = z.*a

    # X=R.*cos(TH)
    # Y=R.*sin(TH)

    # phi0 = Base.copy(phi.value)
    # phi0[:,1,:]=0.5*(phi0[:,1,:]+phi0[:,2,:])
    # phi0[:,end,:]=0.5*(phi0[:,end-1,:]+phi0[:,end,:])
    # phi0[:,:,1]=0.5*(phi0[:,:,1]+phi0[:,:,1])
    # phi0[:,:,end]=0.5*(phi0[:,:,end-1]+phi0[:,:,end])
    # phi0[1,:,:]=0.5*(phi0[1,:,:]+phi0[2,:,:])
    # phi0[end,:,:]=0.5*(phi0[end-1,:,:]+phi0[end,:,:])

    # vmin = minimum(phi0)
    # vmax = maximum(phi0)
    # # 6 surfaces
    # # surfaces 1,2 (x=x[1], x=x[end])
    # mayavis.mesh(squeeze(X[floor(Integer,Nx/2.0),:,:],1),squeeze(Y[floor(Integer,Nx/2.0),:,:],1),squeeze(Z[floor(Integer,Nx/2.0),:,:],1),
    #     scalars=squeeze(phi0[floor(Integer,Nx/2.0)+1,:,:],1), vmin=vmin, vmax=vmax, opacity=0.8)
    # mayavis.mesh(squeeze(X[Nx,:,:],1),squeeze(Y[Nx,:,:],1),squeeze(Z[Nx,:,:],1),
    #     scalars=squeeze(phi0[Nx+2,:,:],1), vmin=vmin, vmax=vmax, opacity=0.8)

    # # surfaces 3,4 (y=y[1], y=y[end]
    # mayavis.mesh(squeeze(X[:,floor(Integer,Ny/2.0),:],2),squeeze(Y[:,floor(Integer,Ny/2.0),:],2),squeeze(Z[:,floor(Integer,Ny/2.0),:],2),
    #     scalars=squeeze(phi0[:,floor(Integer,Ny/2.0)+1,:],2), vmin=vmin, vmax=vmax, opacity=0.8)
    # mayavis.mesh(squeeze(X[:,Ny,:],2),squeeze(Y[:,Ny,:],2),squeeze(Z[:,Ny,:],2),
    #     scalars=squeeze(phi0[:,Ny+2,:],2), vmin=vmin, vmax=vmax, opacity=0.8)

    # # surfaces 5,6 (z=z[1], z=z[end]
    # mayavis.mesh(X[:,:,floor(Integer,Nz/2.0)],Y[:,:,floor(Integer,Nz/2.0)],Z[:,:,floor(Integer,Nz/2.0)],
    #     scalars=phi0[:,:,floor(Integer,Nz/2.0)+1], vmin=vmin, vmax=vmax, opacity=0.8)
    # mayavis.mesh(X[:,:,Nz],Y[:,:,Nz],Z[:,:,Nz],
    #     scalars=phi0[:,:,Nz+1], vmin=vmin, vmax=vmax, opacity=0.8)
    # mayavis.colorbar()
    # mayavis.axes()
    # mshot=mayavis.screenshot()
    # mayavis.show()
    # return mshot
    # end
    # end
