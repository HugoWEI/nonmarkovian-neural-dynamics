"""Regression test: background research sweep must run and write artifacts."""
import csv
import subprocess
import sys
from pathlib import Path


def test_background_sweep_runs_and_writes_artifacts(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "sweep_memory.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    out = tmp_path / "outputs" / "toy_sweep"
    for filename in ("memory_sweep.csv", "memory_sweep.json", "SUMMARY.md"):
        assert (out / filename).is_file(), f"missing artifact: {filename}"
    with (out / "memory_sweep.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert max(float(row["identity_error"]) for row in rows) < 1e-8
