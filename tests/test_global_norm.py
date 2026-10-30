"""Check that physical norm-penalty and parametrized loss execute."""
import numpy as np
from scripts.reduced_fermionic_pinn import train

def test_physical_norm_penalty_training_smoke():
    for mode in ("uniform","global_projected"):
        r=train(mode=mode,steps=20,nwindows=4,seed=2,
                goal_weight=.25,norm_weight=.1)
        assert np.isfinite(r["occupation_rmse"])
        assert np.isfinite(r["max_wave_norm_drift"])

def test_invalid_norm_weight_rejected():
    import pytest
    with pytest.raises(ValueError):
        train(steps=20,norm_weight=-1)
