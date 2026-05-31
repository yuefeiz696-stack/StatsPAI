"""
Augmented Synthetic Control Method (ASCM).

Combines the synthetic control estimator with an outcome model (ridge
regression) to correct for imperfect pre-treatment fit — reducing bias
when the standard SCM cannot perfectly match the treated unit.

Model
-----
τ̂_ascm = τ̂_scm + (Y₀_post - X₀_post β̂) ⊤ γ̂

where γ̂ are the SCM weights, β̂ is a ridge estimator on pre-treatment
donor data, and the correction term adjusts for remaining pre-treatment
imbalance.

References
----------
Ben-Michael, E., Feller, A. and Rothstein, J. (2021).
"The Augmented Synthetic Control Method."
*Journal of the American Statistical Association*, 116(536), 1789-1803. [@benmichael2021augmented]

Ben-Michael, E., Feller, A. and Rothstein, J. (2022).
"Synthetic Controls with Staggered Adoption."
*Journal of the Royal Statistical Society: Series B*, 84(2), 351-381. [@benmichael2022synthetic]
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.optimize import minimize

from ..core.results import CausalResult


def augsynth(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit,
    treatment_time,
    covariates: Optional[List[str]] = None,
    ridge_lambda: Optional[float] = None,
    placebo: bool = True,
    alpha: float = 0.05,
    backend: str = "native",
    **kwargs,
) -> CausalResult:
    """
    Augmented Synthetic Control Method (Ben-Michael, Feller & Rothstein 2021).

    Fits a standard SCM then adds a ridge-outcome-model bias correction.
    Per-period correction is

        bias(t) = m̂_t(X1_pre) − Σ_j γ_j m̂_t(X_j,pre),

    where m̂_t is a ridge regression of donor post-period outcomes on
    donor pre-period outcomes. Collapses to standard SCM when
    ``ridge_lambda → ∞`` and to pure outcome-model imputation when
    ``ridge_lambda → 0``.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    outcome, unit, time : str
        Column names.
    treated_unit, treatment_time : scalar
        Treated-unit identifier and first treatment period.
    covariates : list of str, optional
        Additional predictors (currently informational; main adjustment
        comes from pre-treatment outcomes).
    ridge_lambda : float, optional
        Ridge penalty. When ``None``, selected by leave-one-donor-out CV.
    placebo : bool, default True
        Run in-space placebo permutation tests for SE / p-value.
    alpha : float, default 0.05
        Significance level.
    backend : {'native', 'augsynth', 'r'}, default 'native'
        ``'native'`` uses StatsPAI's Python ridge-augmented SCM
        implementation. ``'augsynth'``/``'r'`` delegates the point
        estimate and pre-period RMSPE to the R ``augsynth`` package
        through ``Rscript`` using ``progfunc='Ridge'`` and ``scm=TRUE``.
        The R backend is intended for exact reference-package parity;
        the native path remains the dependency-light default.
    **kwargs
        Ignored — accepted for dispatcher compatibility.

    Returns
    -------
    CausalResult
        ``detail`` has one row per post-treatment period with columns
        ``time, treated, counterfactual, effect``. ``model_info`` includes
        ``pre_rmspe, post_rmspe, weights, ridge_lambda, n_donors,
        n_pre_periods, n_post_periods, placebo_distribution``.

    References
    ----------
    Ben-Michael, E., Feller, A. and Rothstein, J. (2021). "The Augmented
    Synthetic Control Method." *JASA*, 116(536), 1789-1803. [@benmichael2021augmented]

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.synth.california_tobacco()
    >>> result = sp.augsynth(df, outcome='cigsale', unit='state', time='year',
    ...                       treated_unit='California', treatment_time=1989)
    >>> print(result.summary())
    """
    backend_norm = backend.lower().replace("-", "_")
    if backend_norm in {"augsynth", "r", "augsynth_r"}:
        if covariates:
            raise NotImplementedError(
                "The augsynth R reference backend currently supports the "
                "outcome/treatment specification used in the parity harness; "
                "use backend='native' for covariates."
            )
        if ridge_lambda is not None:
            raise NotImplementedError(
                "The augsynth R reference backend uses augsynth::augsynth's "
                "own Ridge regularisation convention; use backend='native' "
                "for an explicit ridge_lambda."
            )
        return _augsynth_r_backend(
            data=data,
            outcome=outcome,
            unit=unit,
            time=time,
            treated_unit=treated_unit,
            treatment_time=treatment_time,
            alpha=alpha,
        )
    if backend_norm != "native":
        raise ValueError(
            "Unknown backend. Use 'native', 'augsynth', or 'r'."
        )

    # --- Input validation (unified with classic SCM contract) ---
    required_cols = [outcome, unit, time]
    if covariates:
        required_cols = required_cols + list(covariates)
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data")
    if treated_unit not in data[unit].values:
        raise ValueError(
            f"Treated unit '{treated_unit}' not found in '{unit}' column"
        )

    # --- Reshape to wide format ---
    panel = data.pivot_table(index=unit, columns=time, values=outcome)
    all_times = sorted(panel.columns)
    pre_times = [t for t in all_times if t < treatment_time]
    post_times = [t for t in all_times if t >= treatment_time]

    if len(pre_times) < 2:
        from statspai.exceptions import DataInsufficient
        raise DataInsufficient(
            "Need at least 2 pre-treatment periods",
            recovery_hint=(
                "Augmented SCM fits an outcome bridge on pre-periods — "
                "needs at least 2. Use sp.did if you only have 1 pre period."
            ),
            diagnostics={"n_pre_periods": int(len(pre_times))},
            alternative_functions=["sp.did"],
        )
    if len(post_times) < 1:
        from statspai.exceptions import DataInsufficient
        raise DataInsufficient(
            "Need at least 1 post-treatment period",
            recovery_hint=(
                "Verify treatment_time is within the panel window."
            ),
            diagnostics={"n_post_periods": int(len(post_times))},
            alternative_functions=[],
        )

    # Treated and donor matrices
    Y1_pre = panel.loc[treated_unit, pre_times].values.astype(np.float64)
    Y1_post = panel.loc[treated_unit, post_times].values.astype(np.float64)

    donors = [u for u in panel.index if u != treated_unit]
    Y0_pre = panel.loc[donors, pre_times].values.astype(np.float64)  # (J, T0)
    Y0_post = panel.loc[donors, post_times].values.astype(np.float64)  # (J, T1)

    J = len(donors)
    T0 = len(pre_times)
    T1 = len(post_times)

    # --- Step 1: Standard SCM weights ---
    gamma = _scm_weights(Y1_pre, Y0_pre)

    # --- Step 2: Ridge outcome model ---
    # Fit ridge of donor post-outcomes on donor pre-outcomes:
    #   Y0_post = X β + ε,  X = Y0_pre (J × T0),  β ∈ R^{T0 × T1}
    # Closed-form:  β = (X'X + λI)^{-1} X' Y0_post.
    if ridge_lambda is None:
        ridge_lambda = _cv_ridge_lambda_bias(Y0_pre, Y0_post)

    beta = _ridge_post_coef(Y0_pre, Y0_post, ridge_lambda)   # (T0, T1)

    # --- Step 3: Augmented estimate (Ben-Michael et al. 2021 Eq. 3) ---
    Y1_hat_scm_pre = Y0_pre.T @ gamma    # (T0,)
    Y1_hat_scm_post = Y0_post.T @ gamma  # (T1,)

    pre_residual_scm = Y1_pre - Y1_hat_scm_pre           # (T0,)
    pre_rmspe = float(np.sqrt(np.mean(pre_residual_scm ** 2)))

    # Per-period bias correction:
    #   bias(t) = m̂_t(X1_pre) − Σ_j γ_j m̂_t(X_j,pre)
    #          = (Y1_pre − Y0_pre'γ) @ β_t
    #          = pre_residual_scm @ β[:, t]
    bias_per_period = pre_residual_scm @ beta            # (T1,)
    Y1_hat_aug_post = Y1_hat_scm_post + bias_per_period  # (T1,)

    effects = Y1_post - Y1_hat_aug_post
    att = float(np.mean(effects))

    # --- Inference via placebo permutation ---
    if placebo:
        placebo_effects = []
        for j in range(J):
            other_idx = [i for i in range(J) if i != j]
            Y_plac_pre = Y0_pre[j]
            Y_plac_post = Y0_post[j]
            Y_others_pre = Y0_pre[other_idx]
            Y_others_post = Y0_post[other_idx]

            g_plac = _scm_weights(Y_plac_pre, Y_others_pre)
            plac_pre_hat = Y_others_pre.T @ g_plac
            plac_post_hat = Y_others_post.T @ g_plac

            beta_plac = _ridge_post_coef(
                Y_others_pre, Y_others_post, ridge_lambda
            )
            plac_residual = Y_plac_pre - plac_pre_hat
            plac_bias = plac_residual @ beta_plac

            plac_eff = float(np.mean(
                Y_plac_post - plac_post_hat - plac_bias
            ))
            placebo_effects.append(plac_eff)

        placebo_effects = np.array(placebo_effects)
        se = float(np.std(placebo_effects, ddof=1))
        pvalue = float(np.mean(np.abs(placebo_effects) >= abs(att)))
        pvalue = max(pvalue, 1 / (J + 1))

        t_crit = sp_stats.norm.ppf(1 - alpha / 2)
        ci = (att - t_crit * se, att + t_crit * se)
    else:
        placebo_effects = np.array([])
        se = float("nan")
        pvalue = float("nan")
        ci = (float("nan"), float("nan"))

    # Build period-level results
    effects_df = pd.DataFrame({
        "time": post_times,
        "treated": Y1_post,
        "counterfactual": Y1_hat_aug_post,
        "effect": effects,
    })

    # Unified gap table (full trajectory, matches classic SCM contract)
    all_times_arr = np.array(pre_times + post_times)
    Y1_all = np.concatenate([Y1_pre, Y1_post])
    Y1_hat_all = np.concatenate([Y1_hat_scm_pre, Y1_hat_aug_post])
    gap_all = Y1_all - Y1_hat_all
    gap_df = pd.DataFrame({
        "time": all_times_arr,
        "treated": Y1_all,
        "synthetic": Y1_hat_all,
        "gap": gap_all,
        "post_treatment": np.concatenate([
            np.zeros(T0, dtype=bool), np.ones(T1, dtype=bool),
        ]),
    })

    weight_df = (
        pd.DataFrame({"unit": donors, "weight": gamma})
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )
    weight_df = weight_df[weight_df["weight"] > 1e-6]

    return CausalResult(
        method="Augmented Synthetic Control (ASCM)",
        estimand="ATT",
        estimate=att,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=len(data),
        detail=effects_df,
        model_info={
            "model_type": "Synthetic Control (Augmented)",
            "pre_rmspe": pre_rmspe,
            "pre_treatment_rmse": pre_rmspe,
            "pre_treatment_mspe": pre_rmspe ** 2,
            "post_rmspe": float(np.sqrt(np.mean(effects ** 2))),
            "weights": weight_df,
            "weights_dict": dict(zip(donors, gamma)),
            "n_donors": J,
            "n_pre_periods": T0,
            "n_post_periods": T1,
            "treatment_time": treatment_time,
            "treated_unit": treated_unit,
            "ridge_lambda": ridge_lambda,
            "effects_by_period": effects_df,
            "gap_table": gap_df,
            "times": all_times_arr,
            "Y_synth": Y1_hat_all,
            "Y_treated": Y1_all,
            "placebo_distribution": placebo_effects,
            "n_placebos": len(placebo_effects),
        },
    )


def _find_rscript() -> str:
    """Return a usable Rscript executable, including common macOS paths."""
    candidates = [
        shutil.which("Rscript"),
        "/Library/Frameworks/R.framework/Resources/bin/Rscript",
        "/usr/local/bin/Rscript",
        "/opt/homebrew/bin/Rscript",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise RuntimeError(
        "The augsynth backend requires Rscript plus the R packages "
        "'augsynth' and 'jsonlite'. Install R or use backend='native'."
    )


def _augsynth_r_backend(
    *,
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    alpha: float,
) -> CausalResult:
    """Delegate ASCM estimation to R ``augsynth`` for parity."""
    if not pd.api.types.is_numeric_dtype(data[time]):
        raise TypeError(
            "The augsynth R backend requires a numeric time column so it "
            "can reproduce the reference package's pre/post split."
        )

    unit_col = "statspai_unit"
    time_col = "statspai_time"
    outcome_col = "statspai_outcome"
    treated_col = "statspai_treated"
    target_col = "statspai_target"
    panel_df = pd.DataFrame(
        {
            unit_col: data[unit],
            time_col: data[time],
            outcome_col: data[outcome],
        }
    )
    panel_df[target_col] = (data[unit] == treated_unit).astype(int)
    panel_df[treated_col] = (
        (data[unit] == treated_unit) & (data[time] >= treatment_time)
    ).astype(int)
    panel_df = panel_df.sort_values([unit_col, time_col]).reset_index(drop=True)

    r_script = r'''
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("expected 3 arguments: input output treatment_time")
}
input_path <- args[[1]]
output_path <- args[[2]]
treatment_time <- as.numeric(args[[3]])

suppressPackageStartupMessages({
  library(augsynth)
  library(jsonlite)
})

df <- read.csv(input_path, stringsAsFactors = FALSE, check.names = FALSE)
fit <- augsynth::augsynth(
  form = statspai_outcome ~ statspai_treated,
  unit = statspai_unit,
  time = statspai_time,
  data = df,
  progfunc = "Ridge",
  scm = TRUE
)

sm <- summary(fit)
att_est <- as.numeric(sm$average_att$Estimate)

synth_traj <- predict(fit)
yrs <- as.numeric(names(synth_traj))
treated_rows <- df[df$statspai_target == 1, ]
treated_y <- treated_rows$statspai_outcome[match(yrs, treated_rows$statspai_time)]
pre_idx <- yrs < treatment_time
pre_residuals <- treated_y[pre_idx] - synth_traj[pre_idx]
pre_rmspe <- sqrt(mean(pre_residuals^2))

payload <- list(
  estimate = att_est,
  pre_rmspe = as.numeric(pre_rmspe),
  n_obs = nrow(df),
  n_units = length(unique(df$statspai_unit)),
  n_pre_periods = sum(sort(unique(df$statspai_time)) < treatment_time),
  n_post_periods = sum(sort(unique(df$statspai_time)) >= treatment_time)
)
jsonlite::write_json(
  payload,
  output_path,
  auto_unbox = TRUE,
  null = "null",
  na = "null",
  digits = 16
)
'''

    rscript = _find_rscript()
    with tempfile.TemporaryDirectory(prefix="statspai_augsynth_") as tmp:
        tmp_path = Path(tmp)
        data_path = tmp_path / "panel.csv"
        script_path = tmp_path / "augsynth_backend.R"
        out_path = tmp_path / "result.json"
        panel_df.to_csv(data_path, index=False)
        script_path.write_text(r_script, encoding="utf-8")

        proc = subprocess.run(
            [
                rscript,
                str(script_path),
                str(data_path),
                str(out_path),
                str(float(treatment_time)),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "R augsynth backend failed. stderr:\n"
                f"{proc.stderr.strip()}"
            )
        payload = json.loads(out_path.read_text(encoding="utf-8"))

    pre_rmspe = float(payload["pre_rmspe"])
    ci = (float("nan"), float("nan"))
    model_info = {
        "model_type": "Synthetic Control (Augmented)",
        "backend": "augsynth",
        "r_package": "augsynth",
        "progfunc": "Ridge",
        "scm": True,
        "pre_rmspe": pre_rmspe,
        "pre_treatment_rmse": pre_rmspe,
        "pre_treatment_mspe": pre_rmspe ** 2,
        "n_donors": int(payload["n_units"]) - 1,
        "n_pre_periods": int(payload["n_pre_periods"]),
        "n_post_periods": int(payload["n_post_periods"]),
        "treatment_time": treatment_time,
        "treated_unit": treated_unit,
        "ridge_lambda": None,
    }

    return CausalResult(
        method="Augmented Synthetic Control (R augsynth reference backend)",
        estimand="ATT",
        estimate=float(payload["estimate"]),
        se=float("nan"),
        pvalue=float("nan"),
        ci=ci,
        alpha=alpha,
        n_obs=int(payload["n_obs"]),
        detail=None,
        model_info=model_info,
    )


# ====================================================================== #
#  Internal helpers
# ====================================================================== #

def _scm_weights(Y1_pre: np.ndarray, Y0_pre: np.ndarray) -> np.ndarray:
    """
    Standard SCM: find non-negative weights that minimise
    ||Y1_pre - Y0_pre' γ||² subject to sum(γ) = 1, γ >= 0.
    """
    from ._core import solve_simplex_weights
    return solve_simplex_weights(Y1_pre, Y0_pre.T)


def _ridge_post_coef(
    Y0_pre: np.ndarray,
    Y0_post: np.ndarray,
    lam: float,
) -> np.ndarray:
    """
    Ridge regression coefficients for the outcome model used in the
    Ben-Michael et al. (2021) ASCM bias correction.

    Fits β so that Y0_post ≈ Y0_pre @ β using Tikhonov-regularised OLS,
    with X ≡ Y0_pre treated as a (J, T0) design matrix and Y ≡ Y0_post
    treated as a (J, T1) multi-output target.

    Closed-form: β = (X'X + λ I_{T0})^{-1} X' Y0_post.

    Parameters
    ----------
    Y0_pre : (J, T0) donor pre-treatment outcomes.
    Y0_post : (J, T1) donor post-treatment outcomes.
    lam : non-negative ridge penalty.

    Returns
    -------
    beta : (T0, T1) coefficient matrix.
    """
    T0 = Y0_pre.shape[1]
    A = Y0_pre.T @ Y0_pre + lam * np.eye(T0)
    rhs = Y0_pre.T @ Y0_post
    return np.linalg.solve(A, rhs)


def _cv_ridge_lambda_bias(
    Y0_pre: np.ndarray,
    Y0_post: np.ndarray,
    lambdas: Optional[np.ndarray] = None,
) -> float:
    """
    Leave-one-donor-out CV to pick the ridge penalty for the ASCM
    outcome model m̂: Y0_pre → Y0_post.
    """
    if lambdas is None:
        lambdas = np.logspace(-3, 3, 20)

    J = Y0_pre.shape[0]
    best_lam = 1.0
    best_mse = np.inf

    for lam in lambdas:
        mse = 0.0
        for j in range(J):
            idx = [i for i in range(J) if i != j]
            X_tr = Y0_pre[idx]
            Y_tr = Y0_post[idx]
            try:
                beta = _ridge_post_coef(X_tr, Y_tr, lam)
                pred = Y0_pre[j] @ beta      # (T1,)
                mse += float(np.mean((Y0_post[j] - pred) ** 2))
            except np.linalg.LinAlgError:
                mse += 1e10
        mse /= J
        if mse < best_mse:
            best_mse = mse
            best_lam = float(lam)

    return best_lam
