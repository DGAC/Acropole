"""End-to-end tests for the `acropole` CLI (black-box, via subprocess)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "example_flight.csv"

QAR_FLAGS = [
    "--typecode",
    "FLPL_AIRC_TYPE",
    "--groundspeed",
    "GRND_SPD_KT",
    "--altitude",
    "ALTI_STD_FT",
    "--vertical-rate",
    "VERT_SPD_FTMN",
    "--airspeed",
    "TRUE_AIR_SPD_KT",
    "--mass",
    "MASS_KG",
    "--second",
    "FLIGHT_TIME",
]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "acropole.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.e2e
def test_help_lists_estimate() -> None:
    result = _run("--help")
    assert result.returncode == 0
    assert "estimate" in result.stdout


@pytest.mark.e2e
@pytest.mark.skipif(not EXAMPLE.exists(), reason="example_flight.csv not packaged")
def test_estimate_writes_default_output(tmp_path: Path) -> None:
    # copy the example into tmp so the default <flight>_fuel.csv lands there
    flight = tmp_path / "flight.csv"
    flight.write_text(EXAMPLE.read_text())
    result = _run("estimate", str(flight), *QAR_FLAGS)
    assert result.returncode == 0
    default_out = tmp_path / "flight_fuel.csv"
    assert default_out.exists()
    header = default_out.read_text().splitlines()[0]
    assert "fuel_flow_kgh" in header
    assert "fuel_cumsum" in header


@pytest.mark.e2e
@pytest.mark.skipif(not EXAMPLE.exists(), reason="example_flight.csv not packaged")
def test_estimate_writes_explicit_output(tmp_path: Path) -> None:
    out = tmp_path / "result.csv"
    result = _run("estimate", str(EXAMPLE), *QAR_FLAGS, "--out", str(out))
    assert result.returncode == 0
    assert out.exists()
    header = out.read_text().splitlines()[0]
    assert "fuel_flow_kgh" in header


@pytest.mark.e2e
def test_missing_file_errors() -> None:
    result = _run("estimate", "/does/not/exist.csv")
    assert result.returncode == 1
    assert "file not found" in result.stderr


@pytest.mark.e2e
@pytest.mark.skipif(not EXAMPLE.exists(), reason="example_flight.csv not packaged")
def test_missing_column_errors() -> None:
    # the file exists but its QAR columns don't match the default names
    result = _run("estimate", str(EXAMPLE))
    assert result.returncode == 1
    assert "Column" in result.stderr
    assert "not found" in result.stderr
