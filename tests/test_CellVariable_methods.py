# -*- coding: utf-8 -*-
"""
Basic, incomplete testing of CellVariable arithmetic

"""

complete_success = False

import numpy as np
# import matplotlib.pyplot as plt

import pyfvtool as pf


msh = pf.Grid3D(20, 20, 20,
                1.0, 1.0, 1.0)

cv1 = pf.CellVariable(msh, 1.0)
cv2 = pf.CellVariable(msh, 2.0)

cv12 = cv1 + cv2
assert np.all(cv12.value == cv1.value\
                            + cv2.value)
    
cv3 = pf.CellVariable(msh, 3.0)
cv4 = pf.CellVariable(msh, 4.0)

cv34 = cv3 * cv4
assert np.all(cv34.value == cv3.value\
                            * cv4.value)

    
msh2 = pf.Grid2D(20, 20, 1.0, 1.0)

cv5 = pf.CellVariable(msh, 5.0)
cv6 = pf.CellVariable(msh, 2.0)


cv5.BCs.left.a[:] = 0.0
cv5.BCs.left.b[:] = 1.0
cv5.BCs.left.c[:] = 1.25

cv5.BCs.top.a[:] = 0.0
cv5.BCs.top.b[:] = 1.0
cv5.BCs.top.c[:] = -1.25

cv56 = cv5/cv6 # Warning! Ghost cells may lead to division by zero
               # This is why only inner cells should participate in arithmetic
assert np.all(cv56.value == cv5.value\
                            / cv6.value)
    
# test of boundary condition transfer
assert np.all(cv5.BCs.left.a == 0.0)
assert np.all(cv5.BCs.left.b == 1.0)
assert np.all(cv5.BCs.left.c == 1.25)


assert np.all(cv56.BCs.left.a == 0.0)
assert np.all(cv56.BCs.left.b == 1.0)
assert np.all(cv56.BCs.left.c == 1.25)


assert np.all(cv56.BCs.top.a == 0.0)
assert np.all(cv56.BCs.top.b == 1.0)
assert np.all(cv56.BCs.top.c == -1.25)


# We should probably test all arithmetic methods for CellVariables...
# Copilot?


# Quick, incomplete test of pf.funceval (and transfer of BCs)

def addfun(u0, u1):
    return u0 + u1

cvnew12 = pf.funceval(addfun, cv1, cv2)

assert np.all(cvnew12.value == cv12.value)


def divfun(u0, u1):
    return u0/u1

cvnew56 = pf.funceval(divfun, cv5, cv6)

assert np.all(cvnew56.value == cv56.value)


assert np.all(cvnew56.BCs.left.a == 0.0)
assert np.all(cvnew56.BCs.left.b == 1.0)
assert np.all(cvnew56.BCs.left.c == 1.25)


assert np.all(cvnew56.BCs.top.a == 0.0)
assert np.all(cvnew56.BCs.top.b == 1.0)
assert np.all(cvnew56.BCs.top.c == -1.25)





# End of test: all tests completed successfully

complete_success = True


# pytest
def test_success():
    assert complete_success