"""
utility functions come here
"""

import numpy as np

def int_range(a:int, b:int) -> np.ndarray:
    """
    returns a range of integer values from a to b
    """
    return np.linspace(a, b, b-a+1, dtype=int)



def fluxLimiter(flName: str, eps =2e-16):
    """
    returns a flux limiter function

    Parameters
    -----

    
    Notes
    -----
    This function returns a function handle to a flux limiter of user's
    choice.
    available flux limiters are: 'CHARM', 'HCUS', 'HQUICK', 'VanLeer',
    'VanAlbada1', 'VanAlbada2', 'MinMod', 'SUPERBEE', 'Sweby', 'Osher',
    'Koren', 'smart', 'MUSCL', 'QUICK', 'MC', and 'UMIST'.
    Default limiter is 'SUPERBEE'. See:
    <http://en.wikipedia.org/wiki/Flux_limiter>
    
    """
    if flName=="CHARM":
        def FL(r):
            return ((r>0.0)*r*(3.0*r+1.0)/(((r+1.0)**2.0)+eps*(r==-1.0)))
    elif flName=="HCUS":
        def FL(r):
            return (1.5*(r+np.abs(r))/(r+2.0))
    elif flName=="HQUICK":
        def FL(r):
            return (2.0*(r+np.abs(r))/((r+3.0)+eps*(r==-3.0)))
    elif flName=="ospre":
        def FL(r):
            return ((1.5*r*(r+1.0))/(r*(r+1.0)+1.0+eps*((r*(r+1.0)+1.0)==0.0)))
    elif flName=="VanLeer":
        def FL(r):
            return ((r+np.abs(r))/(1.0+np.abs(r)))
    elif flName=="VanAlbada1":
        def FL(r):
            return ((r+r*r)/(1.0+r*r))
    elif flName=="VanAlbada2":
        def FL(r):
            return (2.0*r/(1+r*r))
    elif flName=="MinMod":
        def FL(r):
            return ((r>0.0)*np.minimum(r,1.0))
    elif flName=="SUPERBEE":
        def FL(r):
            return (np.maximum(0.0, np.maximum(np.minimum(2.0*r,1.0), np.minimum(r,2.0))))
    elif flName=="Osher":
        def FL(r):
            b=1.5
            return (np.maximum(0.0, np.minimum(r,b)))
    elif flName=="Sweby":
        def FL(r):
            b=1.5
            return (np.maximum(0.0, np.maximum(np.minimum(b*r,1.0), np.minimum(r,b))))
    elif flName=="smart":
        def FL(r):
            return (np.maximum(0.0, np.minimum(4.0,np.minimum(0.25+0.75*r, 2.0*r))))
    elif flName=="Koren":
        def FL(r):
            return (np.maximum(0.0, np.minimum(2.0*r, np.minimum((1.0+2.0*r)/3.0, 2.0))))
    elif flName=="MUSCL":
        def FL(r):
            return (np.maximum(0.0, np.minimum(2.0*r, np.minimum(0.5*(1+r), 2.0))))
    elif flName=="QUICK":
        def FL(r):
            return (np.maximum(0.0, np.minimum(2.0, np.minimum(2.0*r, (3.0+r)/4.0))))
    elif flName=="UMIST":
        def FL(r):
            return (np.maximum(0.0, np.minimum(2.0, np.minimum(2.0*r, np.minimum((1.0+3.0*r)/4.0, (3.0+r)/4.0)))))
    else:
        print("The flux limiter of your choice is not available. The SUPERBEE flux limiter is used instead.")
        def FL(r):
            return (np.maximum(0.0, np.maximum(np.minimum(2.0*r,1.0), np.minimum(r,2.0))))
    return FL



class SignedTuple(tuple):
    """Just like a tuple, but supporting negation and unary positive operations
    
    Can only contain objects that support negation. It does nothing when
    the unary positive operator is applied.

    This class is intended for returning matrix equation terms (M and RHS)
    such that they can be used with solvePDE without reserve. At present, 
    no such signed tuples are needed, however. In the cases where tuples of
    M and RHS are returned, a negation operation makes no sense (e.g. on
    the boundaryConditionsTerm). All other cases, only use pure RHS or pure
    M, which already support negation.
    """
    def __init__(self, ii):
        # When this tuple-subclass object has been created (__new__),
        # it is already initialized with the tuple values.
        #
        # for xx in self:
        #      print(xx)
        #
        # So, here, in this __init__ we only need to check if the SignedTuple
        # object only contains 'negatable' objects. No need to call super()
        #
        for i in ii:
            if not hasattr(i,'__neg__'):
                raise ValueError('Object incompatible with SignedTuple')

    def __pos__(self):
        return self

    def __neg__(self):
        l = []
        for x in self:
            l.append(-x)
        return type(self)(l)



