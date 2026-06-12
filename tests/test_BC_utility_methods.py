"""
Testing of utility methods for setting boundary conditions

Candidates:
fixedValue (Dirichlet)
fixedGradient (Neumann)
NewtonLaw(k, h, T_ext)
defaultNoFlux() a = 1.0, b = 0.0, c = 0.0


"""
import numpy as np

import pyfvtool as pf



# system parameters
Nr = 100
Lr = 1.0
T_init = 298.15
T_ext = 274.00
k = 0.1 
rhocp = 1.0
alpha = k / rhocp
S = 50.0 # TO DO: realistic values?



def test_default_no_flux():
    mesh = pf.CylindricalGrid1D(Nr, Lr)
    Tcell = pf.CellVariable(mesh, T_init)
    
    for bc in [Tcell.BCs.left, Tcell.BCs.right]:
        assert bc.a[0] == 1.0, "default BC: expected a=1.0"
        assert bc.b[0] == 0.0, "default BC: expected b=0.0"
        assert bc.c[0] == 0.0, "default BC: expected c=0.0"



def test_fixed_value():
    mesh = pf.CylindricalGrid1D(Nr, Lr)
    Tcell = pf.CellVariable(mesh, T_init) 
    
    Tcell.BCs.right.fixedValue(T_ext)
    
    for bc in [Tcell.BCs.right]:
        assert bc.a[0] == 0.0, "fixedValue BC error"
        assert bc.b[0] == 1.0, "fixedValue BC error"
        assert bc.c[0] == T_ext, "fixedValue BC error"
        
    assert np.allclose(Tcell, T_init), "Cell does not have expected initial value"
    
    # solve steady-state diffusion with source
    pf.solvePDE(Tcell, 
                [-pf.diffusionTerm(pf.FaceVariable(mesh, alpha)),
                  pf.constantSourceTerm(pf.CellVariable(mesh, S))])
    
    # check against analytic result
    Tanalytic = T_ext + S/(4*k) * (Lr**2 - Tcell.cellcenters.r**2)
    Terr = Tanalytic - Tcell.value
    
    assert np.all(abs(Terr) < 0.05) # small deviations persist in FVM solution
    


if __name__ == "__main__":
    test_default_no_flux()
    test_fixed_value()
    
    ###
    # WIP: next test developed interactively, will become a test_ function when finished
    ###

    ###
    ###
    

    print("All tests passed.")
    
