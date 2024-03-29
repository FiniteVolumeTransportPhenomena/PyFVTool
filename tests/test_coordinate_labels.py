# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:41:50 2024

@author: werts-moltech
"""

# Test coordinate labeling in different mesh geometries


import numpy as np

from pyfvtool import Grid1D, CylindricalGrid1D, SphericalGrid1D
from pyfvtool import Grid2D, CylindricalGrid2D, PolarGrid2D 
from pyfvtool import Grid3D, CylindricalGrid3D, SphericalGrid3D


errors_expected = 0
errors_caught = 0

msh = Grid1D(10, 1.)
xx = msh.cellcenters.x
assert np.all(xx == msh.cellcenters._x)
errors_expected+=1
try:
    rr = msh.cellcenters.r
except AttributeError:
    errors_caught+=1
xx = msh.cellsize.x
assert np.all(xx == msh.cellsize._x)
errors_expected+=1
try:
    rr = msh.cellsize.r
except AttributeError:
    errors_caught+=1  
xx = msh.facecenters.x
assert np.all(xx == msh.facecenters._x)
errors_expected+=1
try:
    rr = msh.facecenters.r
except AttributeError:
    errors_caught+=1



msh = CylindricalGrid1D(10, 1.)
rr = msh.cellcenters.r
assert np.all(rr == msh.cellcenters._x)
errors_expected+=1
try:
    xx = msh.cellcenters.x
except AttributeError:
    errors_caught+=1
rr = msh.cellsize.r
assert np.all(rr == msh.cellsize._x)
errors_expected+=1
try:
    xx = msh.cellsize.x
except AttributeError:
    errors_caught+=1
rr = msh.facecenters.r
assert np.all(rr == msh.facecenters._x)
errors_expected+=1
try:
    xx = msh.facecenters.x
except AttributeError:
    errors_caught+=1  


    
print('Coordinate label errors expected: ', errors_expected,
      '  caught: ', errors_caught)


def test_success():
    assert errors_caught == errors_expected