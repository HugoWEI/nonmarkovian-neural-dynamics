"""Numerical POD bounds on arbitrary complex sparse-adjoint snapshots."""
import numpy as np
from nonmarkovian_neural_dynamics.dual_compression import (
    CompressedDual,simpson_weights
)


def test_simpson_weights_integrate_quadratic():
    t=np.linspace(0,2,101)
    w=simpson_weights(t)
    assert abs(w@t**2-8/3)<1e-12


def test_rank_bound_and_full_recovery():
    gen=np.random.default_rng(17)
    t=np.linspace(0,2,21)
    dual=gen.normal(size=(21,8))+1j*gen.normal(size=(21,8))
    source=gen.normal(size=(21,8))+1j*gen.normal(size=(21,8))
    pod=CompressedDual(t,dual)
    errs=[]
    for rank in (1,3,8,21):
        data=pod.contract(source,rank)
        assert data["gradient_absolute_error"]<=data["certified_discrete_error_bound"]+1e-9
        errs.append(data["relative_dual_L2_error"])
    assert all(a>=b-1e-10 for a,b in zip(errs,errs[1:]))
    assert errs[-1]<1e-10


def test_low_rank_exact_time_span():
    t=np.linspace(0,1,41)
    mode=(1+.2j*t)[:,None]*np.array([1.,2.j,-.2])[None,:]
    source=np.ones_like(mode)
    pod=CompressedDual(t,mode)
    value=pod.contract(source,1)
    assert value["gradient_absolute_error"]<1e-11
    assert value["relative_dual_L2_error"]<1e-10
