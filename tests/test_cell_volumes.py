"""
Testing cell volume calculation by evaluating the total domain volume.
"""

import numpy as np
import pyfvtool as pf



# Global parameters
R = 1.0
Nr = 5
Ntheta = 5
Z = 1.0
Nz = 5
Nphi = 5

# Output params
fmtstr = "{0:.13s}    {1:.12f}  {2:.12f}  {3:.12f}"



def test_cylindrical_grid_1d():
    """Test that the total cell volume matches the analytical volume for CylindricalGrid1D."""
    msh = pf.CylindricalGrid1D(Nr, R)
    u = pf.CellVariable(msh, 1.0)
    expected_volume = np.pi * R**2 * 1.0
    mshsum = np.sum(msh.cellvolume)
    uintg = u.domainIntegral()
    print(fmtstr.format(msh.__repr__(), expected_volume, mshsum, uintg))
    assert np.allclose(expected_volume, mshsum), "CylindricalGrid1D volume error"
    assert np.allclose(expected_volume, uintg), "CylindricalGrid1D volume error"



def test_cylindrical_grid_2d():
    """Test that the total cell volume matches the analytical volume for CylindricalGrid2D."""
    msh = pf.CylindricalGrid2D(Nr, Nz, R, Z)
    u = pf.CellVariable(msh, 1.0)
    expected_volume = np.pi * R**2 * Z
    mshsum = np.sum(msh.cellvolume)
    uintg = u.domainIntegral()
    print(fmtstr.format(msh.__repr__(), expected_volume, mshsum, uintg))
    assert np.allclose(expected_volume, mshsum), "CylindricalGrid2D volume error"
    assert np.allclose(expected_volume, uintg), "CylindricalGrid2D volume error"



def test_polar_grid_2d():
    """Test that the total cell volume matches the analytical volume for PolarGrid2D."""
    THETA = 2 * np.pi
    msh = pf.PolarGrid2D(Nr, Ntheta, R, THETA)
    u = pf.CellVariable(msh, 1.0)
    expected_volume = np.pi * R**2
    mshsum = np.sum(msh.cellvolume)
    uintg = u.domainIntegral()
    print(fmtstr.format(msh.__repr__(), expected_volume, mshsum, uintg))
    assert np.allclose(expected_volume, mshsum), "PolarGrid2D volume error"
    assert np.allclose(expected_volume, uintg), "PolarGrid2D volume error"



def test_cylindrical_grid_3d():
    """Test that the total cell volume matches the analytical volume for CylindricalGrid3D."""
    THETA = 2 * np.pi
    msh = pf.CylindricalGrid3D(Nr, Ntheta, Nz, R, THETA, Z)
    u = pf.CellVariable(msh, 1.0)
    expected_volume = np.pi * R**2 * Z
    mshsum = np.sum(msh.cellvolume)
    uintg = u.domainIntegral()
    print(fmtstr.format(msh.__repr__(), expected_volume, mshsum, uintg))
    assert np.allclose(expected_volume, mshsum), "CylindricalGrid3D volume error"
    assert np.allclose(expected_volume, uintg), "CylindricalGrid3D volume error"



def test_spherical_grid_1d():
    """Test that the total cell volume matches the analytical volume for SphericalGrid1D."""
    msh = pf.SphericalGrid1D(Nr, R)
    u = pf.CellVariable(msh, 1.0)
    expected_volume = 4/3 * np.pi * R**3
    mshsum = np.sum(msh.cellvolume)
    uintg = u.domainIntegral()
    print(fmtstr.format(msh.__repr__(), expected_volume, mshsum, uintg))
    assert np.allclose(expected_volume, mshsum), "SphericalGrid1D volume error"
    assert np.allclose(expected_volume, uintg), "SphericalGrid1D volume error"



def test_spherical_grid_1d_uneven():
    """Test that the total cell volume matches the analytical volume for 
    SphericalGrid1D (unevenly spaced grid)"""
    facelocs = R * np.array([0.0, 0.1, 0.25, 0.5, 0.8, 1.0])
    msh = pf.SphericalGrid1D(facelocs)
    u = pf.CellVariable(msh, 1.0)
    expected_volume = 4/3 * np.pi * R**3
    mshsum = np.sum(msh.cellvolume)
    uintg = u.domainIntegral()
    print(fmtstr.format(msh.__repr__(), expected_volume, mshsum, uintg))
    assert np.allclose(expected_volume, mshsum), "SphericalGrid1D volume error"
    assert np.allclose(expected_volume, uintg), "SphericalGrid1D volume error"



def test_spherical_grid_3d():
    """Test that the total cell volume matches the analytical volume for SphericalGrid3D."""
    THETA = np.pi
    PHI = 2*np.pi
    msh = pf.SphericalGrid3D(Nr, Ntheta, Nphi, R, THETA, PHI)
    u = pf.CellVariable(msh, 1.0)
    expected_volume = 4/3 * np.pi * R**3
    mshsum = np.sum(msh.cellvolume)
    uintg = u.domainIntegral()
    print(fmtstr.format(msh.__repr__(), expected_volume, mshsum, uintg))
    assert np.allclose(expected_volume, mshsum), "SphericalGrid3D volume error"
    assert np.allclose(expected_volume, uintg), "SphericalGrid3D volume error"




if __name__ == '__main__':
    test_cylindrical_grid_1d()
    test_cylindrical_grid_2d()
    test_polar_grid_2d()
    test_cylindrical_grid_3d()
    test_spherical_grid_1d()
    test_spherical_grid_1d_uneven()
    test_spherical_grid_3d()
    