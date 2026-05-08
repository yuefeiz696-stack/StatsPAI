"""Panel negative-binomial regression guards."""

from __future__ import annotations

import numpy as np
import pandas as pd

import statspai as sp


def _panel_nb_data(seed: int = 20260507, n_g: int = 24, n_t: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gid = np.repeat(np.arange(n_g), n_t)
    group_effect = np.linspace(-0.8, 0.8, n_g)
    x = np.repeat(group_effect, n_t) + rng.normal(scale=0.35, size=n_g * n_t)
    eta = 0.15 + 0.35 * x + group_effect[gid]
    mu = np.exp(eta)
    alpha = 0.6
    size = 1.0 / alpha
    y = rng.negative_binomial(size, size / (size + mu))
    return pd.DataFrame({"y": y, "x": x, "id": gid, "t": np.tile(np.arange(n_t), n_g)})


def test_nbreg_formula_fixed_effects_are_explicit_not_ignored():
    df = _panel_nb_data()

    result = sp.nbreg("y ~ x | id", data=df, maxiter=60, tol=1e-6)

    assert result.model_info["fixed_effects"] == ["id"]
    assert result.model_info["n_fe_levels"] == {"id": df["id"].nunique()}
    assert result.model_info["n_fe_params"] == df["id"].nunique() - 1
    assert any(name.startswith("C(id)[") for name in result.params.index)


def test_xtnbreg_fe_wraps_nbreg_with_entity_metadata():
    df = _panel_nb_data()

    result = sp.xtnbreg(
        "y ~ x",
        data=df,
        entity="id",
        model="fe",
        maxiter=60,
        tol=1e-6,
    )

    assert result.model_info["panel_model"] == "fixed_effects"
    assert result.model_info["entity"] == "id"
    assert result.model_info["cluster"] == "id"
    assert result.model_info["fixed_effects"] == ["id"]


def test_xtnbreg_allows_formula_panel_part_without_entity_argument():
    df = _panel_nb_data(n_g=12, n_t=5)

    result = sp.xtnbreg("y ~ 1 | id", data=df, model="fe", maxiter=40, tol=1e-6)

    assert result.model_info["entity"] == "id"
    assert result.model_info["fixed_effects"] == ["id"]
