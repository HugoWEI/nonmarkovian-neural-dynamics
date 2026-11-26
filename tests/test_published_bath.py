"""Offline checks for parser of the original paper's source data."""
from scripts.upstream_bath_audit import analyze_text, parse_coefficients


def small_fixture():
    header = "2\n1\n1\n1\n2\n"
    values = []
    index = 0
    for sign in (1, 2):
        for j, (eta_re, eta_im, rate) in enumerate(
            ((0.25, 0.3, 5.0), (0.0, -0.025, 0.15)), start=1
        ):
            index += 1
            values.append(
                f"{index} {sign} 1 1 1 {j} {eta_re} {eta_im} 0 0 {-rate} 0 0"
            )
    return header + "\n".join(values)


def test_parse_coefficients():
    shape, groups = parse_coefficients(small_fixture())
    assert shape == (2, 1, 1, 1, 2)
    assert len(groups) == 2


def test_slow_mode_time():
    _, rows = analyze_text(small_fixture())
    assert all(abs(r["slowest_time_constant"] - 1 / 0.15) < 1e-12 for r in rows)
    assert all(r["integral_abs_C_0_20"] > 0 for r in rows)
