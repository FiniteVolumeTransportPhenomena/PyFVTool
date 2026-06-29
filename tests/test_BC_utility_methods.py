"""
Testing of utility methods for setting boundary conditions

- defaultNoFlux
- fixedValue (Dirichlet)
- fixedGradient (Neumann)
- newtonCooling(k, h, T_ext)

"""

import numpy as np

import pyfvtool as pf



def test_default_bcs():
    Nx = 100
    Lx = 1.0
    T_init = 298.15
    
    mesh = pf.Grid1D(Nx, Lx)
    Tcell = pf.CellVariable(mesh, T_init)
    
    for bc in [Tcell.BCs.left, Tcell.BCs.right]:
        assert bc.a[0] == 1.0, "default BC: expected a=1.0"
        assert bc.b[0] == 0.0, "default BC: expected b=0.0"
        assert bc.c[0] == 0.0, "default BC: expected c=0.0"



def test_default_no_flux():
    Nx = 100
    Lx = 1.0
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
    Nr = 100
    Lr = 1.0
    T_init = 298.15
    T_ext = 274.00
    k = 0.1 
    rhocp = 1.0
    alpha = k / rhocp
    S = 50.0 # TO DO: use realistic values?
    
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
    Nx = 100
    Lx = 1.0
    kbc = 12.14
    T_ext = 274.00
    k = 0.1 
    rhocp = 1.0
    alpha = k / rhocp
    
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



def test_newton_cooling(silent=True):
    #
    # Newton cooling example: 
    #    http://olivier.granier.free.fr/MOOC-Anglais/Transferts/co/ex-CCP-6-transferts.html
    #
    # Interestingly, in PyFVTool we should do Cylindrical2D(r, z) to use the 
    # actual Newton BC.
    #
    # In PyFVTool Grid1D, the cooling through the side wall would show up as a 
    # (linear) source term. Interesting for a future exercise.
    # 

    Nr = 20
    Nz = 100
    # We use a very thin rod and high aspect ratio because the 1D analytic 
    # approximation (used for comparison)
    # is based on the assumption of a very high aspect ratio.
    # For a later modeling exercise, we can of course play with the aspect
    # ratio to investigate how the 1D analytic solution starts deviating in
    # cases of every lower aspect ratios.
    Lr = 0.075 
    Lz = 3.0
    
    k = 50.0  # [W m-1 K-1]
    h = 100.0 # [W m-2 K-1]
    rhocp = 3.3e6 # 
    alpha = k / rhocp
    
    T_source = 400.0
    T_ext = 280.0
    rixsel = Nr//2 # index of r position whose z profile is analysed
    Tdev_tol = 2.0 # acceptable deviation between FVM Cylindrical2D and analytic 1D model
    
    mesh = pf.CylindricalGrid2D(Nr, Nz, Lr, Lz)
    Tcell = pf.CellVariable(mesh, 0.0)
    
    # Finally, the 'top' boundary condition setting should be irrelevant
    # because the cylinder should long enough such that the extremity is 
    # already at T_ext. All BC types should give same result.
    # Tcell.BCs.top.fixedValue(T_ext)
    Tcell.BCs.bottom.fixedValue(T_source)

    Tcell.BCs.right.newtonCooling(k, h, T_ext)
    
    pf.solvePDE(Tcell, 
                [-pf.diffusionTerm(pf.FaceVariable(mesh, alpha))])
    
    rr, zz, Trrzz = Tcell.plotprofile()
    # in the future, convert to xarray.DataArray for easier processing
    
    # Take center of domain as representative 'radially averaged" temperature 
    # along the rod
    # Compare with analytic 1D solution
    # see:  http://olivier.granier.free.fr/MOOC-Anglais/Transferts/co/ex-CCP-6-transferts.html
    D = np.sqrt((k*Lr)/(2*h))
    Tan = (T_source-T_ext)*np.exp(-zz/D) + T_ext   

    if not silent:
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.clf()
        pf.visualizeCells(Tcell)
        # in the future, come up with more fancy plotting
    
        plt.figure(2)
        plt.clf()
        plt.plot(zz, Trrzz[rixsel, :])     
        plt.plot(zz, Tan)
        
        plt.show()
 
    Tdev = Tan - Trrzz[rixsel, :]
    assert np.all(abs(Tdev) < Tdev_tol),\
        "deviation beyond tolerance between Cylindrical2D FVM and analytical 1D models"



def test_newton_cooling2(silent=True):
    #
    # Model of the evolution of the temperature gradient in the side-wall
    # of a cup filled with hot chocolate milk
    #
    # Double Newton boundary conditions.
    #
    T_room = 293.15
    T_cup_init = T_room
    T_chocomilk = 355.0
    
    c_p_cup = 0.105e3 # J kg-1 K-1 porcelain specific heat capacity
    rho_cup = 2400.0 # kg m3 porcelain density
    k_cup = 1.5 # W m-1 K-1 porcelain thermal conductivity
    alpha_cup = k_cup / (rho_cup*c_p_cup)
    
    h_chocomilk_cup = 500.0 # W m-2 K-1 (if you stir, this can be much higher)
    h_cup_room = 20.0 # W m-2 K-1
    
    R_exterior = 0.040 # m 
    R_interior = 0.036 # m
    
    Nr = 60

    r = np.linspace(R_interior, R_exterior, Nr)
    mesh = pf.CylindricalGrid1D(r)
    
    T = pf.CellVariable(mesh, T_cup_init)
    
    T.BCs.left.newtonCooling(k_cup, h_chocomilk_cup, T_chocomilk)
    T.BCs.right.newtonCooling(k_cup, h_cup_room, T_room)
    
    dt = 0.025
    Nt = 400
    plotix = np.array(np.logspace(0, np.log10(Nt), 30).round(), dtype=int) - 1
    
    if not silent:
        import matplotlib.pyplot as plt
        plt.figure(10)
        plt.clf()
        plt.xlabel('distance from center line of cup / m')
        plt.ylabel('temperature / K')   
    
    for it in range(Nt):
        if it in plotix:
            if not silent:
                plt.plot(*T.plotprofile())
        pf.solvePDE(T, [ pf.transientTerm(T, dt),
                        -pf.diffusionTerm(pf.FaceVariable(mesh, alpha_cup))])
    
    assert np.all(T.value >= 0),\
        "Negative thermodynamic temperatures detected in the side-wall"\
        " of the cup... Something is wrong!"



if __name__ == "__main__":
    test_default_bcs()
    test_default_no_flux()
    test_fixed_value()
    test_fixed_gradient()
    test_newton_cooling(silent=False)
    test_newton_cooling2(silent=False)
    print("All tests passed.")
