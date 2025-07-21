# coding: utf-8
from .pardiso_wrapper import PyPardisoSolver as OneMKLPardisoSolver
from .spsolve import spsolve as spsolve_oneMKL_pardiso
from .spsolve import pypardiso_solver as oneMKL_pardiso_solver_instance

# TODO: add test if OneMKLPardisoSolver actually works to detect missing (optional) dependencies upon import

__all__ = ['OneMKLPardisoSolver', 'spsolve_oneMKL', 'oneMKL_pardiso_solver_instance']
