# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:35:16 2024

@author: werts-moltech
"""

# Test some aspects of FaceVariable handling,
# in particular xvalue, rvalue etc. for different mesh types

import numpy as np

from pyfvtool import Grid1D, CylindricalGrid1D, SphericalGrid1D
from pyfvtool import Grid2D, CylindricalGrid2D, PolarGrid2D 
from pyfvtool import Grid3D, CylindricalGrid3D, SphericalGrid3D

from pyfvtool import FaceVariable






errors_expected = 0
errors_caught = 0



g1d = Grid1D(10, 1.0)
fv = FaceVariable(g1d, 1.0)
fv.xvalue[:] = 3.0
errors_expected += 1
try:
    fv.yvalue[:] = 3.0
except AttributeError:
    errors_caught +=1



g2d = Grid2D(10, 10, 1.0, 1.0)
fv = FaceVariable(g2d, 1.0)
fv.xvalue[:] = 3.0
fv.yvalue[:] = 4.0
errors_expected += 1
try:
    fv.zvalue[:] = 3.0
except AttributeError:
    errors_caught +=1 
errors_expected += 1
try:
    fv.rvalue[:] = 3.0
except AttributeError:
    errors_caught +=1
errors_expected += 1
try:
    fv.thetavalue[:] = 3.0
except AttributeError:
    errors_caught +=1



g3d = Grid3D(10, 10, 10, 1.0, 1.0, 1.0)
fv = FaceVariable(g3d, 1.0)
fv.xvalue[:] = 3.0
fv.yvalue[:] = 4.0
fv.zvalue[:] = 5.0
errors_expected += 1
try:
    fv.rvalue[:] = 7.0
except AttributeError:
    errors_caught +=1 


c1d = CylindricalGrid1D(10, 1.0)
fv = FaceVariable(c1d, 1.0)
fv.rvalue[:] = 3.0
print(fv.rvalue)
errors_expected += 1
try:
    fv.xvalue[:] = 3.0
except AttributeError:
    errors_caught +=1 
errors_expected += 1
try:
    fv.thetavalue[:] = 3.0
except AttributeError:
    errors_caught +=1    


s1d = SphericalGrid1D(10, 1.0)
fv = FaceVariable(s1d, 1.0)
errors_expected += 1
try:
    fv.xvalue[:] = 3.0
except NotImplementedError:
    errors_caught +=1 
    


c2d = CylindricalGrid2D(10, 10, 1.0, 1.0)   
fv = FaceVariable(c2d, 1.0)
errors_expected += 1
fv.zvalue[:] = 3.0
# peek inside (testing only)
assert np.all(fv._yvalue==fv.zvalue)
try:
    fv.yvalue[:] = 3.0
except AttributeError:
    errors_caught +=1 
    


s3d = SphericalGrid3D(10, 10, 10, 1.0, 2*np.pi, 2*np.pi)
fv = FaceVariable(s3d, 1.0)
fv.rvalue[:] = 3.0
fv.thetavalue[:] = 4.0
print(fv.thetavalue)
fv.phivalue[:] = 5.0
assert np.all(fv._xvalue == fv.rvalue)
assert np.all(fv._yvalue == fv.thetavalue)
assert np.all(fv._zvalue == fv.phivalue)


    
print('FaceVariable errors expected: ', errors_expected,
      '  caught: ', errors_caught)


def test_success():
    assert errors_caught == errors_expected
    
