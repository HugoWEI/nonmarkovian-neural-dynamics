"""Independent exact resonant-level numerical physics checks."""
import numpy as np
from nonmarkovian_neural_dynamics.exact_rlm import occupation,quadrature,reference_with_checks

def test_empty_dot_and_probability_bounds():
    t=np.linspace(0,6,61)
    for T in (.03,.3):
        p=occupation(t,T,n=36)
        assert abs(p[0])<1e-12
        assert np.all((p>=-1e-12)&(p<=1+1e-12))

def test_particle_hole_symmetric_thermal_bath():
    t=np.linspace(0,5,35)
    a=occupation(t,.3,n=25,eps=0.,dot_initial=0.)
    b=occupation(t,.3,n=25,eps=0.,dot_initial=1.)
    assert max(abs(a+b-1.))<5e-12

def test_spectral_reference_mesh_and_cutoff():
    t=np.linspace(0,6,61)
    for T in (.03,.3):
        ref,err=reference_with_checks(t,T)
        assert err["resolution_max"]<1e-6
        assert err["cutoff_max"]<1e-4
        assert np.all(np.isfinite(ref))
