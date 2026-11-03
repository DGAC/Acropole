"""End-to-end tests for the `acropole` CLI (black-box, via subprocess)."""

from __future__ import annotations

import csv
import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "example_flight.csv"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_HAS_BACKEND = any(
    importlib.util.find_spec(name) is not None
    for name in ("polars", "pyarrow", "pandas")
)
_HAS_PARQUET = any(
    importlib.util.find_spec(name) is not None
    for name in ("polars", "pyarrow", "fastparquet")
)

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
@pytest.mark.skipif(not _HAS_BACKEND, reason="needs a frame backend")
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
@pytest.mark.skipif(not _HAS_BACKEND, reason="needs a frame backend")
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
@pytest.mark.skipif(not _HAS_BACKEND, reason="needs a frame backend")
def test_missing_column_errors() -> None:
    # the file exists but its QAR columns don't match the default names
    result = _run("estimate", str(EXAMPLE))
    assert result.returncode == 1
    assert "Column" in result.stderr
    assert "not found" in result.stderr


def _assert_csv_output(path: Path) -> None:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (FIXTURES / "flight.csv").open(newline="") as handle:
        expected_rows = list(csv.DictReader(handle))
    assert rows
    for row, expected in zip(rows, expected_rows, strict=True):
        assert row["typecode"] == expected["typecode"]
        for name in ("groundspeed", "altitude", "vertical_rate", "second"):
            assert float(row[name]) == pytest.approx(float(expected[name]))
        for name in ("fuel_flow", "fuel_flow_kgh", "fuel_cumsum"):
            assert math.isfinite(float(row[name]))


@pytest.mark.e2e
@pytest.mark.skipif(not _HAS_BACKEND, reason="needs a frame backend")
@pytest.mark.parametrize("direction", ["read", "write"])
def test_parquet_dependency_or_round_trip(tmp_path: Path, direction: str) -> None:
    """CSV and Parquet interoperate, or explain the missing optional engine."""
    source = FIXTURES / ("flight.parquet" if direction == "read" else "flight.csv")
    output = tmp_path / ("result.csv" if direction == "read" else "result.parquet")
    result = _run("estimate", str(source), "--out", str(output))
    if not _HAS_PARQUET:
        assert result.returncode == 1
        assert "Parquet" in result.stderr
        assert "acropole[cli]" in result.stderr
        assert "pyarrow" in result.stderr
        assert "Traceback" not in result.stderr
        assert not output.exists()
        return
    assert result.returncode == 0, result.stderr
    if direction == "write":
        restored = tmp_path / "restored.csv"
        result = _run("estimate", str(output), "--out", str(restored))
        assert result.returncode == 0, result.stderr
        _assert_csv_output(restored)
    else:
        _assert_csv_output(output)


@pytest.mark.e2e
@pytest.mark.skipif(_HAS_BACKEND, reason="only exercises the backend-free install")
def test_no_backend_errors(tmp_path: Path) -> None:
    output = tmp_path / "result.csv"
    result = _run("estimate", str(FIXTURES / "flight.csv"), "--out", str(output))
    assert result.returncode == 1
    assert "no dataframe backend installed" in result.stderr
    assert "acropole[cli]" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()
