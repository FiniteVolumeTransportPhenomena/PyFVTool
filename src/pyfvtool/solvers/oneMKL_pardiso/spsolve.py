# coding: utf-8

# Copyright (c) 2016, Adrian Haas and ETH Zürich
# All rights reserved.

# Redistribution and use in source and binary forms, with or without 
# modification, are permitted provided that the following conditions are met:

# Redistributions of source code must retain the above copyright notice, this 
# list of conditions and the following disclaimer. Redistributions in binary 
# form must reproduce the above copyright notice, this list of conditions and the 
# following disclaimer in the documentation and/or other materials provided 
# with the distribution.
# Neither the name of ETH Zürich nor the names of its contributors may be used 
# to endorse or promote products derived from this software without specific 
# prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE 
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import scipy.sparse as sp
from .pardiso_wrapper import PyPardisoSolver


# pypardiso_solver is used for the 'spsolve' and 'factorized' functions. Python crashes on windows if multiple
# instances of PyPardisoSolver make calls to the Pardiso library
pypardiso_solver = PyPardisoSolver()


def spsolve(A, b, factorize=True, squeeze=True, solver=pypardiso_solver, *args, **kwargs):
    """
    This function mimics scipy.sparse.linalg.spsolve, but uses the Pardiso solver instead of SuperLU/UMFPACK

        solve Ax=b for x

        --- Parameters ---
        A: sparse square CSR or CSC matrix (scipy.sparse.csr.csr_matrix)
        b: numpy ndarray
           right-hand side(s), b.shape[0] needs to be the same as A.shape[0]
        factorize: boolean, default True
                   matrix A is factorized by default, so the factorization can be reused
        squeeze: default True
                 strange quirk of scipy spsolve, which always returns x.squeeze(), this
                 feature in order to keep it compatible with implementations that rely on
                 this behaviour
        solver: instance of PyPardisoSolver, default pypardiso_solver
                you can supply your own instance of PyPardisoSolver, but using several instances
                of PyPardisoSolver in parallel can lead to errors

        --- Returns ---
        x: numpy ndarray
           solution of the system of linear equations, same shape as b (but returns shape (n,) if b has shape (n,1))

        --- Notes ---
        The computation time increases only minimally if the factorization and the solve phase are carried out
        in two steps, therefore it is factorized by default. Subsequent calls to spsolve with the same matrix A
        will be drastically faster. This makes the "factorized" method obsolete, but it is kept for compatibility.
    """
    if sp.issparse(A) and A.format == "csc":
        A = A.tocsr()  # fixes issue with brightway2 technosphere matrix

    solver._check_A(A)
    if factorize and not solver._is_already_factorized(A):
        solver.factorize(A)

    x = solver.solve(A, b)

    if squeeze:
        return x.squeeze()  # scipy spsolve always returns vectors with shape (n,) indstead of (n,1)
    else:
        return x

