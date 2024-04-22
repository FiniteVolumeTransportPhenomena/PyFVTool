# -*- coding: utf-8 -*-
"""
Basic, incomplete testing of CellVariable arithmetic

"""

import numpy as np
import matplotlib.pyplot as plt

import pyfvtool as pf


msh = pf.Grid3D(20, 20, 20,
                1.0, 1.0, 1.0)

cv1 = pf.CellVariable(msh, 1.0)
cv2 = pf.CellVariable(msh, 2.0)

cv12 = cv1 + cv2
assert np.all(cv12.innerCellValues == cv1.innerCellValues\
                                      + cv2.innerCellValues)
    
cv3 = pf.CellVariable(msh, 3.0)
cv4 = pf.CellVariable(msh, 4.0)

cv34 = cv3 * cv4
assert np.all(cv34.innerCellValues == cv3.innerCellValues\
                                      * cv4.innerCellValues)

    
msh2 = pf.Grid2D(20, 20, 1.0, 1.0)

cv5 = pf.CellVariable(msh, 5.0)
cv6 = pf.CellVariable(msh, 2.0)


cv5.BCs.left.a[:] = 0.0
cv5.BCs.left.b[:] = 1.0
cv5.BCs.left.c[:] = 1.25

cv5.BCs.top.a[:] = 0.0
cv5.BCs.top.b[:] = 1.0
cv5.BCs.top.c[:] = -1.25

cv5.apply_BCs()

cv56 = cv5/cv6 # Warning! The ghost cells create division by zero
assert np.all(cv56.innerCellValues == cv5.innerCellValues\
                                      / cv6.innerCellValues)
    
# test of boundary condition transfer
assert np.all(cv5.BCs.left.a == 0.0)
assert np.all(cv5.BCs.left.b == 1.0)
assert np.all(cv5.BCs.left.c == 1.25)


# TODO: make the following work

# assert np.all(cv56.BCs.left.a == 0.0)
# assert np.all(cv56.BCs.left.b == 1.0)
# assert np.all(cv56.BCs.left.c == 1.25)




# # pytest
# def test_success():
#     assert 1 == 0, "to be implemented"