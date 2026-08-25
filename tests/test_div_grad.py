# partial testing for gradient, divergence, divergenceTerm
#
# further tests are in
# - test_PyFVTool_basic_test.py
# - test_runs.py
#
# overall test coverage for this functionality is small
#
import numpy as np
import pyfvtool as pf


# Grid3D tests
def test_gradient_Grid3D():
    # very simple gradient test
    m = pf.Grid3D(20, 20, 20, 1.0, 1.0, 1.0)
    u = pf.CellVariable(m, 1.0)
    grad_u = pf.gradient(u)
    assert np.allclose(grad_u.xvalue, 0.0)
    assert np.allclose(grad_u.yvalue, 0.0)
    assert np.allclose(grad_u.zvalue, 0.0)



def test_divergence_Grid2D():
    # divergence of the gradient test
    m2 = pf.Grid2D(200, 200, 2*np.pi, 2*np.pi)
    u2 = pf.CellVariable(m2, 0)
    xx = u2.cellcenters.x
    yy = u2.cellcenters.y
    XX, YY = np.meshgrid(xx, yy)
    u2.value = np.sin(XX) * np.sin(YY)
    
    grad_u2 = pf.gradient(u2)
    div_grad_u2 = pf.divergence(grad_u2)
    
    analytic = -2 * u2.value
    err = div_grad_u2.value - analytic
    # never mind the edges
    assert np.max(np.abs(err[1:-1, 1:-1])) < 0.001
    # look at conservation of scalar, and analytic expectation
    np.testing.assert_allclose(
        np.sum(div_grad_u2.value[1:-1, 1:-1]), 0.0, atol=1e-8)



if __name__ == '__main__':
    test_gradient_Grid3D()
    test_divergence_Grid2D()
    