"""
Testing of utility methods for setting boundary conditions

- defaultNoFlux()
- fixedValue (Dirichlet)
- fixedGradient (Neumann)
- NewtonLaw(k, h, T_ext)

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
S = 50.0 # TO DO: use realistic values?

# additional parameters for Grid1D tests
Nx = 100
Lx = 1.0
kbc = 12.14



def test_default_bcs():
    mesh = pf.Grid1D(Nx, Lx)
    Tcell = pf.CellVariable(mesh, T_init)
    
    for bc in [Tcell.BCs.left, Tcell.BCs.right]:
        assert bc.a[0] == 1.0, "default BC: expected a=1.0"
        assert bc.b[0] == 0.0, "default BC: expected b=0.0"
        assert bc.c[0] == 0.0, "default BC: expected c=0.0"



def test_default_no_flux():
    # solve a simple 1D diffusion equation in closed system (no flux boundary conditions)
    mesh = pf.Grid1D(Nx, Lx)
    Ccell = pf.CellVariable(mesh, 0.0)
    Ccell.value[Ccell.cellcenters.x < Lx/2] = 1.0 # all initial concentration in the left half
    # switch BCs (just for testing subsequent restore)
    Ccell.BCs.right.fixedValue(10.0)
    # restore
    Ccell.BCs.left.defaultNoFlux()
    Ccell.BCs.right.defaultNoFlux()
    init_Ctot = Ccell.domainIntegral()
    # sweep to equilibrium: solve time-dependent PDE
    for i in range(25):
        pf.solvePDE(Ccell, [ pf.transientTerm(Ccell, 0.125),
                            -pf.diffusionTerm(pf.FaceVariable(mesh, 1.0))])
    final_Ctot = Ccell.domainIntegral()
    assert np.allclose(init_Ctot, final_Ctot), f"scalar not conserved: {init_Ctot} versus {final_Ctot}"
    assert np.allclose(Ccell.value, 0.5), "closed diffusion system did not reach equilibrium"



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
    


def test_fixed_gradient():
    # simple steady-state 1D
    mesh = pf.Grid1D(Nx, Lx)
    Tcell = pf.CellVariable(mesh, 0.0) 
        
    Tcell.BCs.right.fixedValue(T_ext)
    Tcell.BCs.left.fixedGradient(-kbc)
    
    for bc, cc in zip([Tcell.BCs.left], [-kbc]):
        assert bc.a[0] == 1.0, "fixedGradient BC error"
        assert bc.b[0] == 0.0, "fixedGradient BC error"
        assert bc.c[0] == cc, "fixedGradient BC error"

    pf.solvePDE(Tcell, [-pf.diffusionTerm(pf.FaceVariable(mesh, alpha))])

    # check gradient (will be same over full domain)
    xx, TT = Tcell.plotprofile() # get profile with outer face values
    steadygrad = (TT[-1]-TT[0])/(xx[-1]-xx[0])
    
    assert np.allclose(-kbc, steadygrad)



if __name__ == "__main__":
    test_default_bcs()
    test_default_no_flux()
    test_fixed_value()
    test_fixed_gradient()
    
    ###
    # WIP: next test developed interactively, will become a test_ function when finished
    ###


    # Newton cooling example: 
    #    http://olivier.granier.free.fr/MOOC-Anglais/Transferts/co/ex-CCP-6-transferts.html
    # Interestingly, in PyFVTool we should do Cylindrical2D(r, z) to use the actual Newton BC
    # In PyFVTool Grid1D, the cooling through the side wall would show up as a (linear) source term
    # 

    ###
    ###

    print("All tests passed.")
    
