"""Local Projections (Jordà, AER 2005).

Estimates impulse responses by running horizon-by-horizon regressions:

    y_{t+h} - y_{t-1}  =  α_h  +  β_h · shock_t  +  γ_h · controls_t  +  ε_{t,h}

for h = 0, 1, …, H. The sequence ``{β_h}`` traces the impulse response
function. Inference uses Newey-West HAC standard errors (required because
the residuals are serially correlated by construction).

This estimator is much more flexible than a VAR-based IRF — no
finite-lag structure imposed — and is the default in modern empirical
macro (Ramey 2016; Plagborg-Møller & Wolf 2021).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import DataInsufficient


# --------------------------------------------------------------------- #
#  Newey-West HAC for a single regression
# --------------------------------------------------------------------- #

def _newey_west(X: np.ndarray, e: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West HAC covariance matrix for an OLS regression with
    residuals ``e``. Bartlett kernel weights up to ``lags``.
    """
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    # meat: sum_l w_l (Γ_l + Γ_l') where Γ_l = (1/n) sum_t X_t e_t e_{t-l} X_{t-l}'
    omega = np.zeros((k, k))
    u = X * e[:, None]                          # (n, k)
    for lag in range(lags + 1):
        if lag == 0:
            G = u.T @ u
        else:
            G = u[lag:].T @ u[:-lag]
            G = G + G.T
        w = 1.0 - lag / (lags + 1.0)
        omega += w * G
    V = XtX_inv @ omega @ XtX_inv
    return V


# --------------------------------------------------------------------- #
#  Result
# --------------------------------------------------------------------- #

@dataclass
class LocalProjectionsResult:
    horizons: np.ndarray              # (H+1,)
    irf: np.ndarray                   # (H+1,)
    se: np.ndarray                    # (H+1,)
    ci_lower: np.ndarray              # (H+1,)
    ci_upper: np.ndarray              # (H+1,)
    alpha: float
    shock_name: str
    outcome_name: str
    n_obs_per_horizon: np.ndarray     # (H+1,)  — usable sample per horizon

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "horizon": self.horizons,
            "irf": self.irf,
            "se": self.se,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "n": self.n_obs_per_horizon,
        })

    def plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        ax.plot(self.horizons, self.irf, "-o", color="C0",
                label=f"IRF of {self.outcome_name}")
        ax.fill_between(self.horizons, self.ci_lower, self.ci_upper,
                        alpha=0.25, color="C0",
                        label=f"{int((1 - self.alpha) * 100)}% CI")
        ax.axhline(0, color="grey", linewidth=0.6)
        ax.set_xlabel("Horizon h")
        ax.set_ylabel(f"Response to {self.shock_name}")
        ax.legend()
        return ax

    def summary(self) -> str:
        df = self.to_frame()
        lines = [
            f"Local Projections — {self.outcome_name} on {self.shock_name}",
            "-" * 55,
            df.to_string(index=False, float_format=lambda x: f"{x: .4f}"),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# --------------------------------------------------------------------- #
#  Public entry point
# --------------------------------------------------------------------- #

def local_projections(
    data: pd.DataFrame,
    outcome: str,
    shock: str,
    controls: Optional[List[str]] = None,
    horizons: int = 20,
    nw_lags: Optional[int] = None,
    alpha: float = 0.05,
    cumulative: bool = False,
    auto_lag: bool = True,
) -> LocalProjectionsResult:
    """Estimate impulse responses via Jordà (2005) local projections.

    Parameters
    ----------
    data : pd.DataFrame
        Time-ordered panel (single series for now — panel extension TBD).
    outcome : str
        Column name of the outcome variable y.
    shock : str
        Column name of the shock / treatment variable.
    controls : list of str, optional
        Additional regressors taken **verbatim** from ``data``: the
        column values at time t are used directly, without re-lagging.
        If you want the lag of a control, lag it yourself before
        passing it in (e.g. ``df["unemp_lag"] = df["unemp"].shift(1)``
        and then ``controls=["unemp_lag"]``). The pre-1.16 behaviour
        silently re-lagged controls a second time on top of an auto-
        added ``y_{t-1}``, producing collinear columns and surprising
        impulse responses; see ``MIGRATION.md`` for context.
    horizons : int, default 20
        Number of horizons h = 0, 1, …, H to estimate.
    nw_lags : int, optional
        Newey-West truncation lag. Defaults to ``round(1.5 * horizons)``
        per Kilian & Kim (2011) recommendation.
    alpha : float, default 0.05
        Significance level for the confidence band.
    cumulative : bool, default False
        If ``True``, return the cumulative response
        ``y_{t+h} - y_{t-1}``. Default (False) returns ``y_{t+h}``
        directly.
    auto_lag : bool, default True
        If ``True`` (the legacy default), also adds ``y_{t-1}`` and
        ``shock_{t-1}`` as automatic regressors. Set ``False`` for a
        bare ``y_{t+h} ~ const + shock_t + controls`` specification.
        These two auto-controls were silent in the pre-1.16 docstring.
    """
    if controls is None:
        controls = []
    df = data.copy().reset_index(drop=True)
    n = len(df)
    if nw_lags is None:
        nw_lags = max(1, int(round(1.5 * horizons)))

    y = df[outcome].to_numpy(dtype=float)
    s = df[shock].to_numpy(dtype=float)
    # Auto-added pre-treatment controls (y_{t-1}, shock_{t-1}) — only
    # used when auto_lag is True. User-supplied controls are taken at
    # face value (no re-lagging) so the design matrix matches what a
    # reader of the docstring would expect; see the docstring note
    # above and tests/r_parity/34_lp for the parity contract.
    lag_y = np.concatenate([[np.nan], y[:-1]])
    lag_s = np.concatenate([[np.nan], s[:-1]])
    extra_cols = []
    extra_names = []
    for c in controls:
        col = df[c].to_numpy(dtype=float)
        extra_cols.append(col)
        extra_names.append(c)

    irf = np.empty(horizons + 1)
    se = np.empty(horizons + 1)
    n_used = np.empty(horizons + 1, dtype=int)

    for h in range(horizons + 1):
        # LHS: y_{t+h} (or y_{t+h} - y_{t-1} if cumulative)
        if t_end := n - h:
            y_lhs = y[h:]
            lhs = y_lhs - lag_y[h:] if cumulative else y_lhs
        else:
            raise DataInsufficient("too few observations for horizon")
        shock_t = s[:t_end]
        extras_t = [arr[:t_end] for arr in extra_cols]

        cols = [np.ones(t_end), shock_t]
        if auto_lag:
            cols.append(lag_y[:t_end])
            cols.append(lag_s[:t_end])
        cols.extend(extras_t)
        X = np.column_stack(cols)
        valid = ~np.isnan(lhs) & ~np.any(np.isnan(X), axis=1)
        X_use = X[valid]
        lhs_use = lhs[valid]
        if len(X_use) <= X_use.shape[1] + 2:
            irf[h] = np.nan; se[h] = np.nan; n_used[h] = int(valid.sum())
            continue
        beta, *_ = np.linalg.lstsq(X_use, lhs_use, rcond=None)
        e = lhs_use - X_use @ beta
        V = _newey_west(X_use, e, lags=max(h, nw_lags))
        irf[h] = float(beta[1])                 # coefficient on shock
        se[h] = float(np.sqrt(max(V[1, 1], 0)))
        n_used[h] = int(valid.sum())

    z_crit = stats.norm.ppf(1 - alpha / 2)
    _result = LocalProjectionsResult(
        horizons=np.arange(horizons + 1),
        irf=irf,
        se=se,
        ci_lower=irf - z_crit * se,
        ci_upper=irf + z_crit * se,
        alpha=alpha,
        shock_name=shock,
        outcome_name=outcome,
        n_obs_per_horizon=n_used,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            _result,
            function="sp.timeseries.local_projections",
            params={
                "outcome": outcome, "shock": shock,
                "controls": list(controls) if controls else None,
                "horizons": horizons,
                "nw_lags": nw_lags,
                "alpha": alpha, "cumulative": cumulative,
                "auto_lag": auto_lag,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result
