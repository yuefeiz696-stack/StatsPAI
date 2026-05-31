"""Tests for the optional synthdid R backend."""

import subprocess

import numpy as np
import pytest

import statspai as sp
from statspai.synth.sdid import _find_rscript


def test_native_sdid_exposes_validation_boundary():
    result = sp.sdid(
        sp.datasets.california_prop99(),
        outcome="cigsale",
        unit="state",
        time="year",
        treated_unit="California",
        treatment_time=1989,
        backend="native",
        seed=42,
    )
    assert result.model_info["backend"] == "native"
    assert (
        result.model_info["validation_tier"]
        == "T4_native_regularisation_disclosure"
    )
    assert result.model_info["reference_backend"] == "synthdid"
    assert "regularisation/zeta convention disclosure" in result.model_info[
        "validation_note"
    ]


def _skip_unless_synthdid_available():
    try:
        rscript = _find_rscript()
    except RuntimeError:
        pytest.skip("Rscript is not installed")
    probe = subprocess.run(
        [
            rscript,
            "-e",
            (
                "quit(status = as.integer("
                "!requireNamespace('synthdid', quietly=TRUE) || "
                "!requireNamespace('jsonlite', quietly=TRUE)))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        pytest.skip("R packages synthdid/jsonlite are not installed")


def test_synthdid_backend_matches_reference_fixture():
    _skip_unless_synthdid_available()
    result = sp.sdid(
        sp.datasets.california_prop99(),
        outcome="cigsale",
        unit="state",
        time="year",
        treated_unit="California",
        treatment_time=1989,
        backend="synthdid",
        seed=42,
    )
    assert np.isclose(result.estimate, -15.94838884672099)
    assert np.isclose(result.se, 2.6266066920828113)
    assert result.model_info["backend"] == "synthdid"
    assert result.model_info["validation_tier"] == "reference_backend_bridge"
    assert "not counted as native Python parity evidence" in result.model_info[
        "validation_note"
    ]
    assert result.model_info["n_control"] == 38
    assert result.model_info["T_pre"] == 19


def test_sdid_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        sp.sdid(
            sp.datasets.california_prop99(),
            outcome="cigsale",
            unit="state",
            time="year",
            treated_unit="California",
            treatment_time=1989,
            backend="unknown",
        )
