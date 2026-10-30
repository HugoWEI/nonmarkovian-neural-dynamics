"""A non-normal two-dimensional linear system validates adjoint signs and FD."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from nonmarkovian_neural_dynamics.linear_adjoint import (
    adjoint_sensitivity, fixed_initial_fd
)

def test_non_normal_complex_sparse_adjoint():
    A=csr_matrix([[-.1+.03j,.5],[-.6,-.9+.1j]],dtype=complex)
    B=csr_matrix([[0,.13j],[.07,.08j]],dtype=complex)
    y0=np.array([.8+.1j,.2-.1j])
    c=np.array([1.,.3])
    deriv,_,_=adjoint_sensitivity(A,B,y0,c,T=2.,samples=101)
    h=1e-5
    numeric=fixed_initial_fd(A+h*B,A-h*B,y0,c,2.,h)
    assert abs(deriv-numeric)<1e-8

def test_dimension_validation():
    from pytest import raises
    A=csr_matrix(np.eye(2))
    with raises(ValueError):
        adjoint_sensitivity(A,A,np.ones(3),np.ones(2))


def test_shared_adjoint_two_perturbations():
    from nonmarkovian_neural_dynamics.linear_adjoint import (
        adjoint_trajectory,contract_sensitivity
    )
    A=csr_matrix([[-.3+.1j,.1],[-.2,-.4-.03j]],dtype=complex)
    x=np.array([.5+.2j,.1-.3j])
    c=np.array([1,0.4])
    Bs=[
        csr_matrix([[.1,0],[0,-.2j]],dtype=complex),
        csr_matrix([[0,.15j],[.08,0]],dtype=complex),
    ]
    shared=adjoint_trajectory(A,x,c,T=1.8,samples=101)
    for B in Bs:
        val=contract_sensitivity(B,shared)
        fd=fixed_initial_fd(A+1e-5*B,A-1e-5*B,x,c,1.8,1e-5)
        assert abs(val-fd)<2e-8
