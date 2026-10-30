"""Weak-coupling block memory feedback recovers quadratic leading order."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from nonmarkovian_neural_dynamics.memory_feedback import first_order_feedback

def test_weak_tier_feedback_matches_exact_to_leading_order():
    A=csr_matrix([[-.2]])
    B=csr_matrix([[.04]])
    C=csr_matrix([[.03]])
    D=csr_matrix([[-.7]])
    x0=np.array([1.+0.j])
    t=np.linspace(0,3,201)
    correction,_=first_order_feedback(A,B,C,D,x0,np.array([1.+0j]),t)
    H=csr_matrix([[-.2,.04],[.03,-.7]])
    actual=float((expm_multiply(H*3,np.array([1.,0.])))[0]-
                  (expm_multiply(A*3,np.array([1.])))[0])
    assert correction>0
    assert abs(correction-actual)<1e-5
