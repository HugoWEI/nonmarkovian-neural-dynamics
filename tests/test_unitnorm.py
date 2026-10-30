"""Hard-normalization wavefunction outputs must remain norm 1."""
from scripts.reduced_fermionic_pinn import train

def test_unit_norm_exact_up_to_roundoff():
    for mode in ("uniform","global_projected"):
        r=train(mode=mode,temp=.03,seed=2,steps=20,nwindows=4,unit_norm=True)
        assert r["max_wave_norm_drift"]<1e-10
