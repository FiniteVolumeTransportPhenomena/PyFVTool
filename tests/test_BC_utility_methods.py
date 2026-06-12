"""
Testing of utility methods for setting boundary conditions

set_FixedValue (Dirichlet)
set_FixedGradient (Neumann)
set_NewtonLaw(k, h, T_ext)
set_DefaultNoFlux() a = 1.0, b = 0.0, c = 0.0


"""

import numpy as np

import pyfvtool as pf

# system parameters
Nr = 100
Lr = 1.0
T_init = 298.15
T_ext = 200.00




def test_default_no_flux():
    mesh = pf.CylindricalGrid1D(Nr, Lr)
    Tcell = pf.CellVariable(mesh, T_init)
    
    for bc in [Tcell.BCs.left, Tcell.BCs.right]:
        assert bc.a[0] == 1.0, "default BC: expected a=1.0"
        assert bc.b[0] == 0.0, "default BC: expected b=0.0"
        assert bc.c[0] == 0.0, "default BC: expected c=0.0"






if __name__ == "__main__":
    test_default_no_flux()

    # WIP: next test (FixedValue), will become a test_ function when finished
    mesh = pf.CylindricalGrid1D(Nr, Lr)
    Tcell = pf.CellVariable(mesh, T_init)
    # Tcell.BCs.right.set_FixedValue(0.0)
    assert np.allclose(Tcell, T_init), "Cell does not have expected initial value"
    
    # solve steady-state diffusion and check that result is indeed
    # T_ext everywhere
    


    print("All tests passed.")
    
    
    