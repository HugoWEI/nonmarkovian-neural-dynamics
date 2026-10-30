"""Construct analytic residuals, including a discontinuous piecewise PINN."""
import numpy as np
from scipy.linalg import expm
from nonmarkovian_neural_dynamics.goal_error import (
    terminal_defect,quadratic_occupation_error,HBAR
)

def check(with_jump):
    H=np.array([[-.5,.13],[.13,.2]])
    w0=np.array([1.,0.],dtype=complex)
    pert=np.array([.03+.01j,-.025+.02j])
    jump=np.array([.011-.005j,.014+.006j]) if with_jump else np.zeros(2,complex)
    T=2.0
    def true(t):
        return expm(1j*H*t/HBAR)@w0
    def error_smooth(t):
        return (t/T)**2*pert
    def residual(t,after):
        h=(t/T)**2
        dh=2*t/(T*T)
        return dh*pert-1j*(H@pert)*h/HBAR + (
            -1j*(H@jump)/HBAR if after else 0
        )
    def grid(a,b):
        t=np.linspace(a,b,1001)
        return (t,np.stack([residual(s,a>=1) for s in t]))
    if with_jump:
        segments=[grid(0,1),grid(1,2)]
        discontinuities=[(1.,jump)]
    else:
        segments=[grid(0,2)]
        discontinuities=[]
    e=terminal_defect(H,segments,initial_error=np.zeros(2),
                      jumps=discontinuities)
    actual=error_smooth(T)+jump
    np.testing.assert_allclose(e,actual,atol=2e-10,rtol=0)
    approx=true(T)+actual
    obs=quadratic_occupation_error(approx,e,np.array([0.,1.]))
    direct=(abs(approx[1])**2-abs(true(T)[1])**2).real
    np.testing.assert_allclose(obs[0],direct,atol=2e-10)
    np.testing.assert_allclose(obs[0],obs[1]-obs[2],atol=1e-14)

def test_smooth_identity():
    check(False)

def test_piecewise_interface_jump_identity():
    check(True)

def test_missing_jump_is_detectable():
    H=np.array([[-.5,.13],[.13,.2]])
    j=np.array([0.012,0.003],complex)
    s=[(np.linspace(0,1,101),np.zeros((101,2),complex)),
       (np.linspace(1,2,101),np.zeros((101,2),complex))]
    predicted=terminal_defect(H,s,jumps=[(1.,j)])
    omitted=terminal_defect(H,s)
    assert np.linalg.norm(predicted-omitted)>.01
