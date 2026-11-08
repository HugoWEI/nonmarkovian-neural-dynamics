"""Pure feature-shape and physical zero-time restriction checks."""
import numpy as np
from scripts.interacting_neural_closure_train import feature_tensor

def test_feature_shapes_and_zero_time():
    params=np.array([[1.,.05],[.8,.1]])
    t=np.linspace(0,12,7)
    low=np.ones((2,7))*.5
    first=low+.01*(t[None,:]/12)**2
    features=feature_tensor(params,t,low,first)
    assert features.shape==(2,7,5)
    assert (features[:,0,0]==0).all()
    np.testing.assert_allclose(features[:,:,4],first-low)
