"""
Testing cell volume calculation by evaluating the total domain volume
"""

import numpy as np
import pyfvtool as pf


# For now, bypass the asserts for spherical meshes, simply printing the 
# erroneous results, without failing.
BYPASS_SPHERICAL_ASSERTS = True


R = 1.0
Nr = 5   # lower values lead to bigger errors in cell volume calculation 

THETA = 2*np.pi
Ntheta = 5

Z = 1.0
Nz = 5

PHI = np.pi
Nphi = 5



##
## CylindricalGrid1D

msh = pf.CylindricalGrid1D(Nr,R)
u = pf.CellVariable(msh, 1.0)

Van = np.pi * R**2 * 1.0
mshsum = np.sum(msh.cellvolume)
uintg = u.domainIntegral()

print(Van, mshsum, uintg)

assert np.allclose(Van, mshsum), "CylindricalGrid1D volume error"
assert np.allclose(Van, uintg), "CylindricalGrid1D volume error"



##
## CylindricalGrid2D

msh = pf.CylindricalGrid2D(Nr, Nz, R, Z)
u = pf.CellVariable(msh, 1.0)

Van = np.pi * R**2 * Z
mshsum = np.sum(msh.cellvolume)
uintg = u.domainIntegral()

print(Van, mshsum, uintg)

assert np.allclose(Van, mshsum), "CylindricalGrid2D volume error"
assert np.allclose(Van, uintg), "CylindricalGrid2D volume error"



##
## PolarGrid2D

msh = pf.PolarGrid2D(Nr, Ntheta, R, THETA)
u = pf.CellVariable(msh, 1.0)

Van = np.pi * R**2 
mshsum = np.sum(msh.cellvolume)
uintg = u.domainIntegral()

print(Van, mshsum, uintg)

assert np.allclose(Van, mshsum), "PolarGrid2D volume error"
assert np.allclose(Van, uintg), "PolarGrid2D volume error"



##
## CylindricalGrid3D

msh = pf.CylindricalGrid3D(Nr, Ntheta, Nz, R, THETA, Z)
u = pf.CellVariable(msh, 1.0)

Van = np.pi * R**2 * Z
mshsum = np.sum(msh.cellvolume)
uintg = u.domainIntegral()

print(Van, mshsum, uintg)

assert np.allclose(Van, mshsum), "CylindricalGrid3D volume error"
assert np.allclose(Van, uintg), "CylindricalGrid3D volume error"



##
## SphericalGrid1D

msh = pf.SphericalGrid1D(Nr, R)
u = pf.CellVariable(msh, 1.0)

Van = 4/3 * np.pi * R**3
mshsum = np.sum(msh.cellvolume)
uintg = u.domainIntegral()

print(Van, mshsum, uintg)

if not BYPASS_SPHERICAL_ASSERTS:
    assert np.allclose(Van, mshsum), "SphericalGrid1D volume error"
    assert np.allclose(Van, uintg), "SphericalGrid1D volume error"



##
## SphericalGrid3D

# here, theta = 0...pi; phi = 0...2*pi
THETA = np.pi
Ntheta = 5

PHI = 2*np.pi
Nphi = 5

msh = pf.SphericalGrid3D(Nr, Ntheta, Nphi, R, THETA, PHI)
u = pf.CellVariable(msh, 1.0)

Van = 4/3 * np.pi * R**3
mshsum = np.sum(msh.cellvolume)
uintg = u.domainIntegral()

print(Van, mshsum, uintg)

if not BYPASS_SPHERICAL_ASSERTS:
    assert np.allclose(Van, mshsum), "SphericalGrid3D volume error"
    assert np.allclose(Van, uintg), "SphericalGrid3D volume error"


# remove BYPASS_SPHERICAL_ASSERTS once bug fixed
assert not BYPASS_SPHERICAL_ASSERTS, "Do not by-pass asserts for spherical meshes"
