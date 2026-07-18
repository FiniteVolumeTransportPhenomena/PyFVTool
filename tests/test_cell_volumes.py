"""
Testing cell volume calculation by evaluating the total domain volume.
"""

import numpy as np
import pyfvtool as pf



# Global parameters
R = 1.0
Nr = 4
Ntheta = 4
Z = 1.0
Nz = 4
Nphi = 4

# Output params
fmtstr = "{0:.12s}  {1:15.12f} {2:15.12f} {3:15.12f}"


def test_grid_1d():
    facelocx = np.array([0.0, 1.0, 3.0, 10.0])
    msh = pf.Grid1D(facelocx)
    u = pf.CellVariable(msh, 1.0)
    expected_volume = facelocx[-1] - facelocx[0]
    mshsum = np.sum(msh.cellvolume)
    uintg = u.domainIntegral()
    print(fmtstr.format(msh.__repr__(), expected_volume, mshsum, uintg))
    assert np.allclose(expected_volume, mshsum), "Grid1D volume error"
    assert np.allclose(expected_volume, uintg), "Grid1D volume error"    



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



def test_grid_2d():
    facelocx = np.array([0.0, 1.0, 3.0, 10.0])
    facelocy = np.array([5.0, 8.0, 9.5, 10.0])
    msh = pf.Grid2D(facelocx, facelocy)
    u = pf.CellVariable(msh, 1.0)
    expected_volume =  (facelocx[-1]-facelocx[0])\
                      *(facelocy[-1]-facelocy[0])
    mshsum = np.sum(msh.cellvolume)
    uintg = u.domainIntegral()
    print(fmtstr.format(msh.__repr__(), expected_volume, mshsum, uintg))
    assert np.allclose(expected_volume, mshsum), "Grid2D volume error"
    assert np.allclose(expected_volume, uintg), "Grid2D volume error"    



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



def test_polar_grid_2d_slice():
    """Test that the total cell volume matches the analytical volume for 
    PolarGrid2D (only a quarter slice)."""
    THETA = (2.0/4) * np.pi
    msh = pf.PolarGrid2D(Nr, Ntheta, R, THETA)
    u = pf.CellVariable(msh, 1.0)
    expected_volume = (np.pi * R**2)/4
    mshsum = np.sum(msh.cellvolume)
    uintg = u.domainIntegral()
    print(fmtstr.format(msh.__repr__(), expected_volume, mshsum, uintg))
    assert np.allclose(expected_volume, mshsum), "PolarGrid2D volume error"
    assert np.allclose(expected_volume, uintg), "PolarGrid2D volume error"



def test_grid_3d():
    facelocx = np.array([0.0, 1.0, 3.0, 10.0])
    facelocy = np.array([5.0, 8.0, 9.5, 10.0])
    facelocz = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    msh = pf.Grid3D(facelocx, facelocy, facelocz)
    u = pf.CellVariable(msh, 1.0)
    expected_volume =  (facelocx[-1]-facelocx[0])\
                      *(facelocy[-1]-facelocy[0])\
                      *(facelocz[-1]-facelocz[0])    
    mshsum = np.sum(msh.cellvolume)
    uintg = u.domainIntegral()
    print(fmtstr.format(msh.__repr__(), expected_volume, mshsum, uintg))
    assert np.allclose(expected_volume, mshsum), "Grid3D volume error"
    assert np.allclose(expected_volume, uintg), "Grid3D volume error"    



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



def test_spherical_grid_3d_slice_uneven():
    """Test that the total cell volume matches the analytical volume for SphericalGrid3D.
    (Uneven slices) """
    Rloc = np.array([0.0, 0.2, 0.7, 1.0])
    THETAloc = np.pi*np.array([0.1, 0.25, 0.65, 0.9])
    PHIloc = 2*np.pi*np.array([0.5, 0.6, 0.8, 0.9, 1.0])
    msh = pf.SphericalGrid3D(Rloc, THETAloc, PHIloc)
    u = pf.CellVariable(msh, 1.0)
    expected_volume = (4/3 * np.pi * R**3)*0.8*0.5
    mshsum = np.sum(msh.cellvolume)
    uintg = u.domainIntegral()
    print(fmtstr.format(msh.__repr__(), expected_volume, mshsum, uintg))
    assert np.allclose(expected_volume, mshsum), "SphericalGrid3D volume error"
    assert np.allclose(expected_volume, uintg), "SphericalGrid3D volume error"



if __name__ == '__main__':
    test_grid_1d()
    test_cylindrical_grid_1d()
    test_spherical_grid_1d(); test_spherical_grid_1d_uneven()
    
    test_grid_2d()
    test_cylindrical_grid_2d()
    test_polar_grid_2d(); test_polar_grid_2d_slice()
    
    test_grid_3d()
    test_cylindrical_grid_3d()
    test_spherical_grid_3d(); test_spherical_grid_3d_slice_uneven()
    