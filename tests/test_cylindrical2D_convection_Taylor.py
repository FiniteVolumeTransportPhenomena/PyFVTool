#!/usr/bin/env python
# coding: utf-8

# Adapted from the corresponding Jupyter Notebook
# MW 250530


# # Convection in 2D cylindrical geometry: Taylor dispersion
# 
# *MW, 230906, 240503*

# ## Introduction
# 
# This Notebook presents a first finite-volume modelisation of the dispersion of a solute in a fluid in a thin, long cylindrical tube undergoing Poiseuille flow. This dispersion is described, theoretically and experimentally, in the seminal paper by Taylor [1]. Further background can be found in that paper.
# 
# In the present Notebook, only the purely convective case is studied. A finite-volume solution of the corresponding partial differential equation is obtained using PyFVTool. This result is compared to the analytic expression obtained by Taylor [1].
# 
# ## To do
# 
# - Optimize numerical solution scheme and parameters
# - Try and compare different FV discretizations of the convective term
# 
# ## References
# [1] G. I. Taylor. 'Dispersion of Soluble Matter in Solvent Flowing Slowly through a Tube.', *Proc. Royal Soc. A* **1953**, *219*, 186–203. https://doi.org/10.1098/rspa.1953.0139

# ## Import modules & define utility functions

# In[1]:


import numpy as np
from typing import Any
from numpy.typing import NDArray # type hints need numpy >= 1.21


# In[2]:


# import matplotlib.pyplot as plt


# In[3]:


import pyfvtool as pf


# In[4]:


# visualization routine (imshow-based)
# def phi_visualize():
#     print(f't = {t:.1f} s')
#     # avoid ghost cells
#     plt.imshow(phi.value, origin = 'lower',
#                extent = [zz[0], zz[-1], rr[0]*rzoom, rr[-1]*rzoom])


# In[5]:


# calculate simple finite-volume integral over r
def integral_dr(phi0):
    v = phi0.cellvolume
    c = phi0.value
    return (v*c).sum(axis=0)


# ### Functions for evaluation of the analytic expression by Taylor ('A3')

# In[6]:


# analytic expression from Taylor 1953
def TaylorA3(x: float, t: float, 
             X: float, C_0: float, u_0: float) -> float:
    assert (t >= X/u_0), 't < X/u_0 not implemented'
    if (x >= 0) and (x < X):
        C_m = C_0 * x/(u_0*t)
    elif (x >= X) and (x < u_0*t):
        C_m = C_0 * X/(u_0*t)
    elif (x >= u_0*t) and (x < u_0*t + X):
        C_m = C_0*((X + u_0*t - x)/(u_0*t))
    else:
        C_m = 0.0
    return C_m
           
def TaylorA3_vec(xvec: NDArray[(Any,)], t: float, 
                  X: float, C_0: float, u_0: float) -> NDArray[(Any,)]:
    C_m_vec = np.zeros_like(xvec)
    for ix, x in enumerate(xvec):
        C_m_vec[ix] = TaylorA3(x, t, X, C_0, u_0)
    return C_m_vec  


# ## Finite-volume scheme with PyFVTool

# ### Define system & model parameters

# In[7]:


Lr = 7.5e-05 # [m] radius of cylinder
Lz = 0.3 # [m] length of cylinder
umax = 2*9.4314e-3 # [m s^-1] max flow velocity = 2 time average flow velocity


# In[8]:


# regular grid parameters
Nr = 40
Nz = 500


# In[9]:


# initial condition parameters (cell indices)
loadix0 = 20
loadix1 = 40


# In[10]:


# timestep parameters
deltat = 0.01 # [s] per time step


# In[11]:


# visualization parameters
rzoom = 1000


# ### PyFVTool finite-volume definition

# #### 2D cylindrical mesh

# In[12]:


msh = pf.CylindricalGrid2D(Nr, Nz, Lr, Lz)


# #### Set up Poiseuille flow velocity field

# In[13]:


rr = msh.cellcenters.r
zz = msh.facecenters.z


# In[14]:


uu =  umax*(1 - (rr**2)/(Lr**2)) # does not depend on zz


# In[15]:


u =  pf.FaceVariable(msh, 1.0)


# In[16]:


u.rvalue[:] = 0
u.zvalue[:] = uu[:, np.newaxis]


# In[17]:


# for i in [1, 10, -1]:
#     plt.plot(rr*1e6, u.zvalue[:, i])
# plt.xlabel('$r$ / µm') 
# plt.ylabel('$u_z(r)$ / m s$^{-1}$');


# #### Solution variable
# 
# *Standard 'no flux' boundary conditions. The convective flow field, however, will still transport matter out of the calculation domain.*

# In[18]:


phi = pf.CellVariable(msh, 0.0)


# #### Initial condition

# In[19]:


t=0.


# In[20]:

assert not phi.value.modified # test modified flag    

# initial condition
for i in range(loadix0, loadix1):
    phi.value[:, i] = 1.0

assert phi.value.modified # test modified flag

# In[21]:


# phi_visualize()


# In[22]:


initInt = phi.domainIntegral()
print(initInt)


# In[23]:


phiprofs = []
phiprofs.append((t, integral_dr(phi)))


# ### Solve the convection PDE with time-stepping

# In[24]:


def step_solver(Nstp):
    global t

    # convectionterm = pf.convectionTerm(u) # really ugly results?
    convectionterm = pf.convectionUpwindTerm(u) # numerical diffusion

    for i in range(Nstp):
        # Transient term needs to be re-evaluated at each time step
        transientterm = pf.transientTerm(phi, deltat, 1.0)
        eqnterms = [transientterm,
                    convectionterm]
        pf.solvePDE(phi, eqnterms)
        t += deltat


# In[25]:


step_solver(200)
phiprofs.append((t, integral_dr(phi)))


# In[26]:


print(t, initInt, phi.domainIntegral())


# In[27]:


# phi_visualize()


# In[28]:


step_solver(300)
phiprofs.append((t, integral_dr(phi)))


# In[29]:


step_solver(500)
phiprofs.append((t, integral_dr(phi)))


# In[30]:


print(t, initInt, phi.domainIntegral())


# In[31]:


# phi_visualize()


# ## Comparison between the finite-volume result and the analytic solution
# 
# Taylor [1] considers the radially averaged concentration profile along the tube as a function of time. We compare that to the radially integrated finite-volume result. (The ratio between the radial integral and radial average is simply constant).

# In[32]:


DX = phi.domain.facecenters.z[loadix0]
X = phi.domain.facecenters.z[loadix1] - DX
C_0 = phiprofs[0][1][(loadix0+loadix1)//2] # slot#0 contains initial condition


# In[33]:


zzz = np.linspace(0, Lz, 500)


# In[34]:


for ix, (tprof, phiprof) in enumerate(phiprofs):
    if ix == 2:
        lbl1 = 'FVM'
        lbl2 = 'analytic'
    else:
        lbl1 = None
        lbl2 = None
    # plt.plot(phi.domain.cellcenters.z, phiprof, 
    #          label=lbl1)
    # if tprof >= X/umax:
    #     plt.plot(zzz, TaylorA3_vec(zzz-DX, tprof, X, C_0, umax),
    #              'k:', label=lbl2)
# plt.xlabel('z / m')
# plt.legend();


# The agreement of the finite-volume solution with the analytic result is quite good. The FV calculations parameters have not been optimized. There is some obvious numerical diffusion in the FV result, and also some oscillatory artefact. These numerical artefacts may be reduced by using a different discretization scheme for the convective term. Any good advice in these matters is very welcome!

# ### Simple quantitative benchmark

# In[35]:


(tprof, phiprof) = phiprofs[-1]
z_num, c_num = phi.domain.cellcenters.z, phiprof
c_an_z_num = TaylorA3_vec(z_num-DX, tprof, X, C_0, umax)
norm_err = (c_an_z_num - c_num)/c_an_z_num.max()


# In[36]:


# plt.plot(z_num, norm_err)
# plt.ylabel('norm. error')
# plt.xlabel('$z$ / m');
# plt.ylim(-0.01,0.01)


#%% pytest

def test_success():
    # very basic benchmark for testing integrity of Notebook and calculations
    # checks if the normalized error is below a certain threshold (0.15% of max)
    # over a range of z (between 1/3 and 1/2 of full scale)
    assert np.all(np.abs(norm_err[Nz//3:Nz//2]) < 0.0015)
