"""Train each neural objective briefly to catch runtime/shape regressions."""
import numpy as np
from scripts.reduced_fermionic_pinn import train

def test_objective_modes_train_and_report_metrics():
    for mode in ("uniform","projected","integrated"):
        r=train(mode,temp=.03,seed=1,steps=20,nwindows=4)
        for field in ("occupation_rmse","terminal_occupation_abs_error",
                      "max_wave_norm_drift","runtime_seconds"):
            assert np.isfinite(r[field])
            assert r[field]>=0
