"""High-vs-low generator defect must reproduce exact terminal difference."""
import numpy as np
from scipy.sparse import csr_matrix
from nonmarkovian_neural_dynamics.linear_adjoint import adjoint_trajectory
from nonmarkovian_neural_dynamics.hierarchy_defect import (
    hierarchy_retained_indices,compute_defect,linear_observable_defect
)


def test_nested_hierarchy_error_with_missing_initial_tier():
    A=csr_matrix(np.array([[-.3+.1j,.2,.15],
                           [-.1,-.7,.3],
                           [.22,-.2,-1.2-.05j]],dtype=complex))
    low=A[:2,:2].tocsr()
    y0=np.array([.7+.1j,.2-.07j,.08+.04j])
    c=np.array([1,.4,0])
    t=np.linspace(0,2,201)
    indices=hierarchy_retained_indices([(0,),(1,)],
             [(0,),(1,),(2,)],1)
    approx,r,e0=compute_defect(low,A,y0,indices,t)
    _,exact,dual=adjoint_trajectory(A,y0,c,T=2,samples=len(t))
    initial,source,estimate=linear_observable_defect(dual,r,e0,t)
    true=float(np.real(np.vdot(c,approx[-1]-exact[-1])))
    assert abs(estimate-true)<3e-9
    assert abs(initial)>1e-5
    assert abs(source)>1e-5
    assert abs(estimate-(initial+source))<1e-12


def test_nested_label_validation():
    from pytest import raises
    with raises(ValueError):
        hierarchy_retained_indices([(1,)],[(0,)],2)
