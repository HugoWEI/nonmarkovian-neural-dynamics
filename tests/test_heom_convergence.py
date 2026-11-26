"""Pure convergence-metric regression tests (no HEOM library needed)."""
import numpy as np
from scripts.heom_convergence import compare


def test_identical_trajectories_zero_error():
    x={"times":np.array([0.,1.,2.]),"occupations":np.array([[0,0],[.1,.2],[.2,.3]])}
    m=compare(x,x)
    assert m["max_occupation_difference"]==0
    assert m["rms_occupation_difference"]==0


def test_nontrivial_difference_reported():
    t=np.array([0.,1.,2.])
    a={"times":t,"occupations":np.array([[0,0],[.1,.2],[.2,.3]])}
    b={"times":t,"occupations":np.array([[0,0],[.1,.25],[.21,.3]])}
    m=compare(a,b)
    assert abs(m["max_occupation_difference"]-.05) < 1e-12
    assert abs(m["terminal_occupation_difference"]-.01) < 1e-12
