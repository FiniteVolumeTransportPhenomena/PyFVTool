"""
Simple tests of boundary condition functionality

At present, largely incomplete. We aim for physically meaningful tests, going
beyond regression testing. Yet, any test is better than no test at all.

The boundary conditions are more rigorously tested for certain meshes in 
specific test scripts where PyFVTool's numerical result is compared to a known
existing analytic solution or expected physical behaviour.
"""
import numpy as np
import pyfvtool as pf


def test_gaoflow_Grid3D():
    msh = pf.Grid3D(4, 4, 4, 1.0, 1.0, 1.0)
    BC = pf.BoundaryConditions(msh)
    BC.left.periodic = True
    BC.right.periodic = True
    BCterm = pf.boundaryConditionsTerm(BC)
    assert BCterm[0].nnz == 312,\
        "Unexpected number of stored sparse matrix elements for periodic BCs in Grid3D"


def test_gaoflow_CylindricalGrid3D():
    msh = pf.CylindricalGrid3D(4, 4, 4, 1.0, 2*np.pi, 1.0)
    BC = pf.BoundaryConditions(msh)
    BC.left.periodic = True
    BC.right.periodic = True
    failed = False
    try:
        BCterm = pf.boundaryConditionsTerm(BC)
    except ValueError:
        BCterm = None
        failed = True
    assert failed,\
        "Radial periodic boundary conditions should fail for CylindricalGrid3D."

 


if __name__=='__main__':
    test_gaoflow_Grid3D()
    test_gaoflow_CylindricalGrid3D()
