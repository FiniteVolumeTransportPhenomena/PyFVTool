#!/usr/bin/env python
# coding: utf-8

# ADAPTED from the corresponding Notebook example

# # Advection and Diffusion in Spherical Coordinates with PyFVTool
# 
# In this notebook, we will solve the advection equation in spherical coordinates using the finite volume method. The notebook serves principally for testing the implementation of 'spherical' advection and diffusion terms in `PyFVTool`, but also for illustrating their use. 
# 
# The notebook contains several examples of advection and diffusion equations, and first attempts at visualizing 3D spherical results (suggestions welcome!).

# In[1]:

SILENT = True # no graphical output if silent

import numpy as np
if not SILENT:
    import matplotlib.pyplot as plt
from tqdm import tqdm



# In[3]:


import pyfvtool as pf


# When working with random numbers (`numpy.random`), set the random seed explicitly.
# This is good practice.
# Make sure that you trust your random number generator to be sufficiently 
# random (Numpy's default is OK).
# With a fixed value for the initial seed, you are sure to always obtain the same
# sequence of random numbers from run to run. This makes testing more practical.
# If you want to have a different sequence of random numbers, manually change the
# seed, but make sure that you remember which seed number was used.

# In[4]:


# Set the random seed explicitly
np.random.seed(seed=12345)


# ## First explorations of the spherical finite volume elements

# In[5]:


X = np.array([0.01, 0.1, 0.3, 0.5, 0.55, 1.0])
Y = np.array([0.0, 0.1, 1.0, 1.5, 2.9, 3.0, np.pi, np.pi])
Z = np.array([0.0, 0.01, 0.1, 0.5, 0.7, 0.95, 1.0, 1.25, 1.39, 2.0])
m_non = pf.SphericalGrid3D(X, Y, Z)


# In[6]:


# test the volume calculations
x = np.linspace(0.0, 1.0, 10)
y = np.linspace(0.0, 2*np.pi, 10)
z = np.linspace(0.0, 2*np.pi, 20)
mc = pf.CylindricalGrid3D(x, y, z)
vc = mc._getCellVolumes()
v_cyl = np.pi*x[-1]**2*z[-1]
print(v_cyl, np.sum(vc))

mp = pf.PolarGrid2D(x, y)
vp = mp._getCellVolumes()
v_pol = np.pi*x[-1]*x[-1]
print(v_pol, np.sum(vp))


# In[7]:


x = np.linspace(0.0, 1.0, 5)
y = np.linspace(0.0, np.pi, 10)
z = np.linspace(0.0, 2*np.pi, 10)
ms = pf.SphericalGrid3D(x, y, z)
vs = ms._getCellVolumes()
v_sph = 4/3*np.pi*x[-1]**3
print(v_sph, np.sum(vs))


# ## Diffusion equations: spherical and cylindrical coordinates, 1D and 3D
# 
# Solving a diffusion equation with "no flux" concentration at the "left boundary" (the center) and a fixed concentration on the "right boundary" (the outer surface).
# 
# The same equation is solved in spherical and cylindrical coordinates. Both 1D and 3D geometries are used to test consistency.

# In[8]:


# Calculation parameters
def diffusion_spherical(mesh, t_simulation=7200.0, dt=60.0):
    c_left = 1.0  # left boundary concentration
    c_init = 0.0  # initial concentration
    D_val = 1e-5  # diffusion coefficient (gas phase)

    # Create a cell variable with initial concentration
    # By default, 'no flux' boundary conditions are applied
    c = pf.CellVariable(mesh, c_init)

    # Switch the right boundary to Dirichlet: fixed concentration
    c.BCs.right.a = 0.0
    c.BCs.right.b = 1.0
    c.BCs.right.c = c_left
    if type(mesh) == pf.SphericalGrid3D:
        # make top and bottom boundaries periodic
        c.BCs.back.periodic = True
        c.BCs.front.periodic = True

    # Assign diffusivity to cells
    D_cell = pf.CellVariable(mesh, D_val)
    D_face = pf.geometricMean(
        D_cell
    )  # average value of diffusivity at the interfaces between cells

    # Time loop
    t = 0
    while t < t_simulation:
        # Compose discretized terms for matrix equation
        eqnterms = [pf.transientTerm(c, dt, 1.0), -pf.diffusionTerm(D_face)]

        # Solve PDE
        pf.solvePDE(c, eqnterms)
        t += dt
    return c


# In[9]:


# Calculation parameters
Nx = 20  # number of finite volume cells
Ntheta = 6  # number of cells in the theta direction; avoid theta=0 and theta=pi
Nphi = 5  # number of cells in the phi direction
Lx = 1.0  # [m] length of the domain

# Define mesh
mesh1 = pf.SphericalGrid1D(Nx, Lx)
c1 = diffusion_spherical(mesh1)
mesh3 = pf.SphericalGrid3D(Nx, Ntheta, Nphi, Lx, np.pi, 2 * np.pi)
c3 = diffusion_spherical(mesh3)
mesh1_rad = pf.CylindricalGrid1D(Nx, Lx)
c1_rad = diffusion_spherical(mesh1_rad)
mesh3_cyl = pf.CylindricalGrid3D(Nx, Ntheta, Nphi, Lx, 2 * np.pi, 2 * np.pi)
c3_cyl = diffusion_spherical(mesh3_cyl)


# In[10]:

if not SILENT:
    plt.figure(1)
    plt.clf()
    plt.plot(mesh1.cellcenters.r, c1.value, "--", label="1D spherical")
    plt.plot(mesh3.cellcenters.r, c3.value[:, 0, 0], label="3D spherical")
    plt.plot(mesh1_rad.cellcenters.r, c1_rad.value, "--", label="1D Cylindrical")
    plt.plot(mesh3_cyl.cellcenters.r, c3_cyl.value[:, 0, 0], label="3D cylindrical")
    plt.xlabel("r [m]")
    plt.ylabel("Concentration [-]")
    plt.ylim(-0.05, 1.05)
    plt.legend()


# ## Spherical advection in 3D (with visualization)
# 
# The spherical geometry defined here is that of a layer of a certain thickness at the surface of sphere, *e.g.* a virtual atmosphere on the surface of a virtual Earth.

# In[11]:


r_earth = 6.371e6  # Earth radius [m]
v_wind_max = 10.0  # wind speed [m/s]
# ignoring diffusion
Nr = 20  # number of cells in the r direction
Ntheta = 36  # number of cells in the theta direction
Nphi = 36  # number of cells in the phi direction
Lr = 20e3  # [m] length of the domain in the r direction
r_face = np.linspace(r_earth, r_earth + Lr, Nr + 1)
theta_face = np.linspace(0, np.pi, Ntheta + 1)
phi_face = np.linspace(0, 2 * np.pi, Nphi + 1)
mesh = pf.SphericalGrid3D(r_face, theta_face, phi_face)
c = pf.CellVariable(mesh, 0.0)
# assign a concentration of 1.0 to 20 random locations
c.value[0, np.random.randint(0, Ntheta, 50), np.random.randint(0, Nphi, 50)] = 1000.0

# BC
# left boundary is a fixed concentration
c.BCs.left.a[:] = 0.0
c.BCs.left.b[:] = 1.0
c.BCs.left.c[:] = 0.0
# assign a concentration of 1.0 to a patch of left boundary
# c.BCs.left.c[0:10, 0:10] = 1.0
# right boundary is a fixed concentration
# c.BCs.right.a[:] = 0.0
# c.BCs.right.b[:] = 1.0
# c.BCs.right.c[:] = 0.0
# top and bottom boundaries are periodic
# c.BCs.top.periodic = True
# c.BCs.bottom.periodic = True
c.BCs.back.periodic = True
c.BCs.front.periodic = True
c.apply_BCs() # working with solveMatrixPDE() instead of solvePDE(),
              #  so need to call apply_BCs for updating ghost cells
Mbc, RHSbc = pf.boundaryConditionsTerm(c.BCs)

# create a constant velocity field
v = pf.FaceVariable(mesh, [0.1, 10, 10])

Mc = pf.convectionUpwindTerm(v)

# Time loop
t = 0
dt = 10000.0
n_steps = 5
for i in tqdm(range(n_steps), bar_format="{desc}: {percentage:3.0f}% completed"):
    # Compose discretized terms for matrix equation
    Mt, RHSt = pf.transientTerm(c, dt, 1.0)
    M = Mt + Mc + Mbc
    RHS = RHSt + RHSbc
    c_new = pf.solveMatrixPDE(mesh, M, RHS)
    c.update_value(c_new)
    t += dt
    
# print(c.value[0, :, :])


# ### Visualization
# 
# Meaningful and insightful visualization of 3D spherical and cylindrical FVM 
# results can be quite challenging. It requires thinking carefully on how to 
# best present the features that illustrate the results of your calculations. Of course, it also requires to have right code to make the drawing.
# 
# The visualization is part of the 'post-processing' of numerical calculations. (BTW, already for relatively simple cases, it is wise to separate the number cruching code from the post-processing, by storing the calculation results in a file. This avoids having to wait over and over again for the same calculation to terminate.)
# 
# Here, we show our first suggestions for the graphical output of 3D spherical (and cylindrical) FVM results.

# In[12]:


# visualize the result: use a polar plot for plotting a 'slice' at a certain value of 'phi'
# (lemon - or mellon - style slice)

# select the 'phi' using its index
iphi = 5

r = mesh.cellcenters.r
theta = mesh.cellcenters.theta
phi = mesh.cellcenters.phi

R, Theta = np.meshgrid(r, theta)
X = R * np.sin(Theta)
Y = R * np.cos(Theta)
Z = c.value[:, :, iphi].transpose() 

if not SILENT:
    fig = plt.figure(2)
    plt.clf()
    phideg = np.rad2deg(phi[iphi])
    fig.suptitle(f'phi = {phideg}°')
    ax = fig.add_subplot(111, polar=True)
    # set the color range between 0 and 1
    cax = ax.pcolormesh(Theta, R, Z, cmap="viridis")
    fig.colorbar(cax, label="Concentration [-]")
    # zoom in to the region of interest
    ax.set_ylim([r_earth - Lr, r_earth + Lr])





# In[13]:


# First version of a 3D plotting routine
# Plot a spherical view of a single layer at height/depth 'r' 
# (onion-layer style)

# select r by index
ir = 10 # halfway

data = c.value[ir, :, :] 

# Generate theta and phi values
theta = np.linspace(0, np.pi, data.shape[0])
phi = np.linspace(0, 2 * np.pi, data.shape[1])
theta, phi = np.meshgrid(theta, phi)

# Convert spherical coordinates to Cartesian coordinates
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

if not SILENT:
    # Create a 3D plot
    fig = plt.figure(3)
    plt.clf()
    fig.suptitle(f'r = {mesh.cellcenters.r[ir]}')
    ax = fig.add_subplot(111, projection="3d")
    
    # Plot the surface
    ax.plot_surface(
        x, y, z, facecolors=plt.cm.viridis(data), rstride=1, cstride=1, antialiased=False
    )
    
    # rotate the plot
    ax.view_init(elev=20, azim=80)



# In[14]:


print(c.value.min(), c.value.max())


# ## Advective transport 3D cylindrical example
# 
# Here, we run a similar simulation of a cylindrical version of Earth, to test 3D cylindrical coordinates.

# In[15]:


r_earth = 6.371e6  # Earth radius [m]
v_wind_max = 10.0  # wind speed [m/s]
L = r_earth
# ignoring diffusion
Nr = 20  # number of cells in the r direction
Ntheta = 16  # number of cells in the theta direction
Nz = 16  # number of cells in the phi direction
Lr = 20e3  # [m] length of the domain in the r direction
r_face = np.linspace(0, Lr, Nr + 1)
theta_face = np.linspace(0, 2 * np.pi, Ntheta + 1)
L_face = np.linspace(0, L, Nz + 1)
mesh = pf.CylindricalGrid3D(r_face, theta_face, L_face)
c = pf.CellVariable(mesh, 0.0)
# assign a concentration of 1.0 to 20 random locations
c.value[0, np.random.randint(0, Ntheta, 20), np.random.randint(0, Nz, 20)] = 1000.0

# BC
# top and bottom boundaries are periodic
c.BCs.top.periodic = True
c.BCs.bottom.periodic = True
c.apply_BCs() # c.value changed, explicit call to apply_BCs
Mbc, RHSbc = pf.boundaryConditionsTerm(c.BCs)

# create a velocity field of random values between 0 and v_wind_max
v = pf.FaceVariable(mesh, [0, 0.001, 0.0])
# v.thetavalue = np.random.rand(Nr, Ntheta+1, Nz) * v_wind_max
# v.thetavalue[:, 0, :] = 0.0  # no wind at the poles
# v.thetavalue[:, -1, :] = 0.0  # no wind at the poles

Mc = pf.convectionUpwindTerm(v)

# Time loop
t = 0
dt = 100000.0
t_simulation = 1 * dt
while (t < t_simulation):
    # Compose discretized terms for matrix equation
    Mt, RHSt = pf.transientTerm(c, dt, 1.0)
    M = Mt + Mc + Mbc
    RHS = RHSt + RHSbc
    c_new = pf.solveMatrixPDE(mesh, M, RHS)
    c.update_value(c_new)
    t += dt
    
    
# print(c.value[0, :, :])


# ## Hollow sphere (with visualization)
# 
# This example still needs further explanations (and perhaps some tweaking).

# In[16]:


# hollow sphere
r_in = 0.5
r_out = 1.0
Nr = 20  # number of cells in the r direction
Ntheta = 16  # number of cells in the theta direction
Nphi = 16  # number of cells in the phi direction
r_face = np.linspace(r_in, r_out, Nr + 1)
theta_face = np.linspace(0.0, np.pi, Ntheta + 1)
phi_face = np.linspace(0.0, 2 * np.pi, Nphi + 1)
mesh = pf.SphericalGrid3D(r_face, theta_face, phi_face)
c = pf.CellVariable(mesh, 0.0)
# assign a concentration of 1 to left boundary
c.BCs.left.a[0:5, 0:5] = 0.0
c.BCs.left.b[0:5, 0:5] = 1.0
c.BCs.left.c[0:5, 0:5] = 1.0
# right boundary is a fixed concentration
c.BCs.right.a[:] = 0.0
c.BCs.right.b[:] = 1.0
c.BCs.right.c[:] = 0.0

# top and back boundaries are periodic
# c.BCs.top.periodic = True
c.BCs.front.periodic = True

c.apply_BCs() # not using solvePDE(), but solveMatrixPDE(), apply
              # BCs manually (for initializing ghost cells)
Mbc, RHSbc = pf.boundaryConditionsTerm(c.BCs)

v_wind_max = 10  # wind speed [m/s]
v = pf.FaceVariable(mesh, [0.0, 0.05, 0.05])
v.phiavalue = np.random.rand(Nr, Ntheta, Nphi + 1) * v_wind_max
v.thetavalue = np.random.rand(Nr, Ntheta + 1, Nphi) * v_wind_max

Mc = pf.convectionUpwindTerm(v)

# Time loop
t = 0
dt = 1.0
n_step = 50

for i in tqdm(range(n_step), bar_format="{desc}: {percentage:3.0f}% completed"):
    # Compose discretized terms for matrix equation
    Mt, RHSt = pf.transientTerm(c, dt, 1.0)
    M = Mt + Mbc + Mc
    RHS = RHSt + RHSbc
    c_new = pf.solveMatrixPDE(mesh, M, RHS)
    c.update_value(c_new)

print()

# In[17]:


# visualize the result
r = mesh.cellcenters.r
theta = mesh.cellcenters.theta
R, Theta = np.meshgrid(r, theta)
X = R * np.sin(Theta)
Y = R * np.cos(Theta)
Z = c.value[:, :, 1].transpose() # TO DO: make sure that you extract the values for a certain 'phi'...


if not SILENT:
    fig = plt.figure(4)
    plt.clf()
    ax = fig.add_subplot(111, polar=True)
    # set the color range between 0 and 1
    cax = ax.pcolormesh(Theta, R, Z, cmap="viridis")
    fig.colorbar(cax, label="Concentration [-]");


# In[18]:


print(c.value.min(), c.value.max())


# In[19]:


# Assuming c.value[5, :, :] is the data with size 16x16
data = c.value[1, :, :]  # Replace this with your actual data

# Generate theta and phi values
theta = np.linspace(0, np.pi, data.shape[0])
phi = np.linspace(0, 2 * np.pi, data.shape[1])
theta, phi = np.meshgrid(theta, phi)

# Convert spherical coordinates to Cartesian coordinates
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)


if not SILENT:
    # Create a 3D plot
    fig = plt.figure(5)
    plt.clf()
    ax = fig.add_subplot(111, projection="3d")
    
    # Plot the surface
    ax.plot_surface(
        x, y, z, facecolors=plt.cm.viridis(data), rstride=1, cstride=1, antialiased=False
    )
    
    # rotate the plot
    ax.view_init(elev=50, azim=80)




if not SILENT:
    # Show the plots
    plt.show()



# end test (if the script run until here, it should be OK)
# TODO: add more meaningful, result-oriented tests
successful_finish = True


# pytest
def test_success():
    assert successful_finish

