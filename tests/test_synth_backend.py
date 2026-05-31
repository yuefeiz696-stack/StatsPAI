"""Tests for the optional Synth R backend."""

import subprocess

import numpy as np
import pytest

import statspai as sp
from statspai.synth.scm import _find_rscript


def test_native_synth_exposes_identification_boundary():
    result = sp.synth(
        sp.datasets.basque_terrorism(),
        outcome="gdppc",
        unit="region",
        time="year",
        treated_unit="Basque Country",
        treatment_time=1970,
        method="classic",
        backend="native",
        placebo=False,
    )
    assert result.model_info["backend"] == "native"
    assert result.model_info["validation_tier"] == "identification_dependent_native"
    assert result.model_info["reference_backend"] == "Synth"
    assert "T4 non-uniqueness disclosures" in result.model_info["validation_note"]


def _skip_unless_synth_available():
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
                "!requireNamespace('Synth', quietly=TRUE) || "
                "!requireNamespace('jsonlite', quietly=TRUE)))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        pytest.skip("R packages Synth/jsonlite are not installed")


def test_synth_backend_matches_reference_fixture():
    _skip_unless_synth_available()
    result = sp.synth(
        sp.datasets.basque_terrorism(),
        outcome="gdppc",
        unit="region",
        time="year",
        treated_unit="Basque Country",
        treatment_time=1970,
        method="classic",
        backend="synth",
    )
    assert np.isclose(result.estimate, -0.687789209705457)
    assert np.isclose(result.model_info["pre_treatment_rmse"], 0.0793738251107324)
    weights = result.model_info["weights"].set_index("unit")["weight"]
    assert np.isclose(weights.loc["Madrid"], 0.537261315298536)
    assert np.isclose(weights.loc["Cataluna"], 0.451608208273253)
    assert result.model_info["backend"] == "synth"
    assert result.model_info["validation_tier"] == "reference_backend_bridge"
    assert "not counted as native Python parity evidence" in result.model_info[
        "validation_note"
    ]


def test_synth_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        sp.synth(
            sp.datasets.basque_terrorism(),
            outcome="gdppc",
            unit="region",
            time="year",
            treated_unit="Basque Country",
            treatment_time=1970,
            method="classic",
            backend="unknown",
        )
