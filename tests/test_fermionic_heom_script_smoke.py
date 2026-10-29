"""Dependency-free structural smoke check. Not a real HEOM execution."""
import ast
from pathlib import Path


def test_heom_script_is_parseable():
    path = Path(__file__).resolve().parents[1] / "scripts" / "fermionic_heom_reference.py"
    ast.parse(path.read_text())


def test_script_explicitly_warns_about_approximate_bath_parameters():
    path = Path(__file__).resolve().parents[1] / "scripts" / "fermionic_heom_reference.py"
    content = path.read_text()
    assert "NOT an exact reproduction" in content
    assert "0.05" in content
