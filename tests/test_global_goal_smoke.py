"""Regression test for global-terminal PINN objective variants."""
import numpy as np
from scripts.reduced_fermionic_pinn import train

def test_global_modes_short_training():
    for mode in ("global_projected","global_integrated"):
        x=train(mode=mode,temp=.03,seed=2,steps=20,nwindows=4)
        for key in ("occupation_rmse","terminal_occupation_abs_error",
                    "max_wave_norm_drift","runtime_seconds"):
            assert np.isfinite(x[key])
            assert x[key]>=0
