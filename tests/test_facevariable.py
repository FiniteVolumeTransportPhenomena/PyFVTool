# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:35:16 2024

@author: werts-moltech
"""

# Test some aspects of FaceVariable handling,
# in particular xvalue, rvalue etc. for different mesh types

import numpy as np

import pyfvtool as pf





def test_facevariables():
    
    
    errors_expected = 0
    errors_caught = 0
    
    
    
    g1d = pf.Grid1D(10, 1.0)
    fv = pf.FaceVariable(g1d, 1.0)
    fv.xvalue[:] = 3.0
    errors_expected += 1
    try:
        fv.yvalue[:] = 3.0
    except AttributeError:
        errors_caught +=1
    
    
    
    g2d = pf.Grid2D(10, 10, 1.0, 1.0)
    fv = pf.FaceVariable(g2d, 1.0)
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
    
    
    
    g3d = pf.Grid3D(10, 10, 10, 1.0, 1.0, 1.0)
    fv = pf.FaceVariable(g3d, 1.0)
    fv.xvalue[:] = 3.0
    fv.yvalue[:] = 4.0
    fv.zvalue[:] = 5.0
    errors_expected += 1
    try:
        fv.rvalue[:] = 7.0
    except AttributeError:
        errors_caught +=1 
    
    
    c1d = pf.CylindricalGrid1D(10, 1.0)
    fv = pf.FaceVariable(c1d, 1.0)
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
    
    
    s1d = pf.SphericalGrid1D(10, 1.0)
    fv = pf.FaceVariable(s1d, 1.0)
    errors_expected += 1
    try:
        fv.xvalue[:] = 3.0
    except AttributeError:
        errors_caught +=1 
        
    
    
    c2d = pf.CylindricalGrid2D(10, 10, 1.0, 1.0)   
    fv = pf.FaceVariable(c2d, 1.0)
    errors_expected += 1
    fv.zvalue[:] = 3.0
    # peek inside (testing only)
    assert np.all(fv._yvalue == fv.zvalue)
    try:
        fv.yvalue[:] = 3.0
    except AttributeError:
        errors_caught +=1 
        
    
    
    s3d = pf.SphericalGrid3D(10, 10, 10, 1.0, np.pi, 2*np.pi)
    fv = pf.FaceVariable(s3d, 1.0)
    fv.rvalue[:] = 3.0
    fv.thetavalue[:] = 4.0
    print(fv.thetavalue)
    fv.phivalue[:] = 5.0
    assert np.all(fv._xvalue == fv.rvalue)
    assert np.all(fv._yvalue == fv.thetavalue)
    assert np.all(fv._zvalue == fv.phivalue)
    
    
    print('FaceVariable errors expected: ', errors_expected,
          '  caught: ', errors_caught)
    
    assert errors_caught == errors_expected
    
    
    
def test_facelocations3D():
    m = pf.Grid3D(3, 4, 5, 1.0, 10.0, 100.0)
    X,Y,Z = pf.faceLocations(m)
    assert X.xvalue.shape == X.yvalue.shape, "error in X faceLocations for Grid3D"
    assert X.xvalue.shape == X.zvalue.shape, "error in X faceLocations for Grid3D"
    assert Y.xvalue.shape == Y.yvalue.shape, "error in Y faceLocations for Grid3D"
    assert Y.xvalue.shape == Y.zvalue.shape, "error in Y faceLocations for Grid3D"
#    assert Z.xvalue.shape == Z.yvalue.shape, "error in Z faceLocations for Grid3D"
    assert Z.xvalue.shape == Z.zvalue.shape, "error in Z faceLocations for Grid3D"
    return m,X,Y,Z



if __name__ == '__main__':
    test_facevariables()
    m,X,Y,Z = test_facelocations3D()
    