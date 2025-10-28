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



class TrackedArray(np.ndarray):
    """
    A NumPy array subclass that tracks whether its values have been modified.
    
    The `modified` property is set to True whenever elements are changed via
    item assignment (e.g., arr[0] = 5). Modifications to views or slices also
    mark the base array as modified. Can be manually set or reset using direct
    assignment (e.g., arr.modified = False).
    
    **Limitation**: Only modifications through item assignment (arr[...] = value)
    are automatically tracked. Other in-place operations such as np.copyto(),
    np.put(), np.place(), arr.fill(), arr.sort(), and in-place ufuncs will NOT
    automatically set the modified flag. After using such operations, manually
    set arr.modified = True if tracking is needed.
    
    Attributes
    ----------
    modified : bool
        True if the array (or any view of it) has been modified since creation
        or since the last manual reset. Initially False. Can be set directly.
    
    Examples
    --------
    >>> arr = TrackedArray([1, 2, 3])
    >>> arr.modified
    False
    >>> arr[0] = 10
    >>> arr.modified
    True
    >>> arr.modified = False  # Reset the flag
    >>> arr[1:3][0] = 5  # Modification through a view
    >>> arr.modified
    True
    >>> # Manual tracking for operations that bypass __setitem__
    >>> np.copyto(arr, [7, 8, 9])
    >>> arr.modified = True  # Must set manually
    """
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        obj._modified = False
        return obj
    
    def __array_finalize__(self, obj):
        """Called whenever a new array instance is created."""
        if obj is None:
            return
        # Inherit _modified from parent, or initialize to False
        self._modified = getattr(obj, '_modified', False)
    
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        # Mark this array and its base (if it's a view) as modified
        self._modified = True
        if self.base is not None and isinstance(self.base, TrackedArray):
            self.base._modified = True
    
    @property
    def modified(self):
        """Return whether the array has been modified."""
        # Check both this array and its base
        if self.base is not None and isinstance(self.base, TrackedArray):
            return self._modified or self.base._modified
        return self._modified
    
    @modified.setter
    def modified(self, value):
        """Set the modification tracking flag."""
        self._modified = bool(value)
        # Also set on base if this is a view
        if self.base is not None and isinstance(self.base, TrackedArray):
            self.base._modified = bool(value)


