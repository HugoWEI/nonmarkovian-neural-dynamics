"""Run linear-memory adjoint identity checks across memory regimes."""
import json
from nonmarkovian_neural_dynamics.toy import check_adjoint_identity

if __name__ == "__main__":
    for lam in (2.0, 0.2, 0.02):
        result = check_adjoint_identity(lam=lam)
        print(json.dumps(result, indent=2))
        assert result["absolute_identity_error"] < 1e-7
