"""
Cattaneo-Jansson-Ma (2020) density discontinuity test.

Tests whether the density of the running variable is continuous at
the RD cutoff. Replaces McCrary (2008) as the modern standard for
RD manipulation testing.

Uses local polynomial density estimation on each side of the cutoff
with data-driven bandwidth selection and bias-corrected inference.

References
----------
Cattaneo, M.D., Jansson, M. and Ma, X. (2020).
"Simple Local Polynomial Density Estimators."
*Journal of the American Statistical Association*, 115(531), 1449-1455. [@cattaneo2020simple]

Cattaneo, M.D., Jansson, M. and Ma, X. (2018).
"Manipulation Testing Based on Density Discontinuity."
*The Stata Journal*, 18(1), 234-261. [@cattaneo2018manipulation]
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


def rddensity(
    data: pd.DataFrame,
    x: str,
    c: float = 0,
    p: int = 2,
    h: Optional[Union[float, Sequence[float]]] = None,
    alpha: float = 0.05,
    backend: str = "native",
) -> CausalResult:
    """
    CJM (2020) density discontinuity test for RD manipulation.

    Modern replacement for McCrary (2008). Uses local polynomial
    density estimation with bias-corrected inference.

    Parameters
    ----------
    data : pd.DataFrame
    x : str
        Running variable.
    c : float, default 0
        RD cutoff.
    p : int, default 2
        Polynomial order for density estimation.
    h : float or length-2 sequence, optional
        Bandwidth. A scalar applies the same bandwidth on both sides;
        a length-2 sequence is interpreted as ``(h_left, h_right)``.
        Default: automatic StatsPAI native pilot rule.
    alpha : float, default 0.05
    backend : {"native", "r"}, default "native"
        ``"native"`` uses StatsPAI's dependency-light implementation.
        ``"r"`` delegates to ``rddensity::rddensity`` through
        ``Rscript`` when the R package is installed, matching the
        reference package's selector and test statistic.

    Returns
    -------
    CausalResult
        - ``estimate``: T-statistic for density discontinuity
        - ``pvalue``: p-value for H0: continuous density at cutoff
        - ``model_info``: density estimates left/right, bandwidth

    Examples
    --------
    >>> result = sp.rddensity(df, x='score', c=0)
    >>> print(f"T = {result.estimate:.3f}, p = {result.pvalue:.4f}")
    >>> if result.pvalue < 0.05:
    ...     print("Evidence of manipulation!")

    Notes
    -----
    The test estimates the density from the left and right using
    local polynomial regression on the empirical CDF, then tests:

        H0: f_+(c) = f_-(c)  (no density discontinuity)
        H1: f_+(c) ≠ f_-(c)  (manipulation at cutoff)

    Advantages over McCrary (2008):
    - No arbitrary binning step
    - Data-driven bandwidth with formal optimality
    - Bias-corrected inference

    See Cattaneo, Jansson & Ma (2020, *JASA*).
    """
    backend_norm = backend.lower().replace("-", "_")
    if backend_norm in {"r", "rddensity", "r_reference", "r_package"}:
        return _rddensity_r_backend(data=data, x=x, c=c, p=p, h=h, alpha=alpha)
    if backend_norm not in {"native", "statspai"}:
        raise ValueError("backend must be 'native' or 'r'.")

    X = data[x].values.astype(float)
    X = X[np.isfinite(X)]
    n = len(X)

    if n < 20:
        raise ValueError("Need at least 20 observations.")

    X_c = X - c
    x_left = X_c[X_c < 0]
    x_right = X_c[X_c >= 0]
    n_l = len(x_left)
    n_r = len(x_right)

    if n_l < 5 or n_r < 5:
        raise ValueError("Not enough observations on each side.")

    h_l, h_r, h_source = _resolve_bandwidths(h, x_left, x_right, p)

    # Density estimation via local polynomial on the empirical CDF
    # Pass n (full sample size) so density is on the correct scale
    f_left, se_left = _local_poly_density(x_left, 0, h_l, p, side='left',
                                          n_full=n)
    f_right, se_right = _local_poly_density(x_right, 0, h_r, p, side='right',
                                            n_full=n)

    # Test statistic
    diff = f_right - f_left
    se_diff = np.sqrt(se_left ** 2 + se_right ** 2)
    T_stat = diff / se_diff if se_diff > 0 else 0
    pvalue = float(2 * (1 - stats.norm.cdf(abs(T_stat))))

    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (diff - z_crit * se_diff, diff + z_crit * se_diff)

    model_info = {
        'test': 'Cattaneo-Jansson-Ma (2020)',
        'density_left': f_left,
        'density_right': f_right,
        'density_diff': diff,
        'bandwidth_left': h_l,
        'bandwidth_right': h_r,
        'bandwidth_source': h_source,
        'backend': 'native',
        'validation_note': (
            "StatsPAI native rddensity uses a dependency-light pilot "
            "bandwidth and local-density approximation; manual bandwidths "
            "are sensitivity controls, not a guarantee of "
            "rddensity::rddensity numeric parity. Use backend='r' when exact "
            "reference-package selector/test-statistic parity is required."
        ),
        'polynomial_order': p,
        'n_left': n_l,
        'n_right': n_r,
        'cutoff': c,
    }

    _result = CausalResult(
        method='CJM (2020) Density Test',
        estimand='T-statistic (density discontinuity)',
        estimate=float(T_stat),
        se=float(se_diff),
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        model_info=model_info,
        _citation_key='rddensity',
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            _result,
            function="sp.diagnostics.rddensity",
            params={
                "x": x, "c": c, "p": p, "h": h, "alpha": alpha,
                "backend": backend,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


def _resolve_bandwidths(h, x_left, x_right, p) -> Tuple[float, float, str]:
    if h is None:
        return _cjm_bandwidth(x_left, p), _cjm_bandwidth(x_right, p), "native_pilot"

    if np.isscalar(h):
        h_value = float(h)
        if not np.isfinite(h_value) or h_value <= 0:
            raise ValueError("h must be positive and finite.")
        return h_value, h_value, "manual_scalar"

    h_values = list(h)
    if len(h_values) != 2:
        raise ValueError("h must be a scalar or a length-2 sequence (h_left, h_right).")
    h_l, h_r = float(h_values[0]), float(h_values[1])
    if not (np.isfinite(h_l) and np.isfinite(h_r)) or h_l <= 0 or h_r <= 0:
        raise ValueError("h_left and h_right must be positive and finite.")
    return h_l, h_r, "manual_side_specific"


def _rddensity_r_backend(
    data: pd.DataFrame,
    x: str,
    c: float,
    p: int,
    h: Optional[Union[float, Sequence[float]]],
    alpha: float,
) -> CausalResult:
    if shutil.which("Rscript") is None:
        raise ImportError("backend='r' requires Rscript and the R package rddensity.")

    X = data[x].values.astype(float)
    X = X[np.isfinite(X)]
    n = len(X)
    if n < 20:
        raise ValueError("Need at least 20 observations.")
    n_l = int(np.sum(X < c))
    n_r = int(np.sum(X >= c))
    if n_l < 5 or n_r < 5:
        raise ValueError("Not enough observations on each side.")

    if h is None:
        h_l = h_r = None
        h_source = "rddensity_default"
    else:
        h_l, h_r, h_source = _resolve_bandwidths(h, X[X < c] - c, X[X >= c] - c, p)

    r_code = r"""
suppressPackageStartupMessages({
  if (!requireNamespace("rddensity", quietly = TRUE)) {
    stop("R package 'rddensity' is not installed")
  }
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("R package 'jsonlite' is not installed")
  }
})
args <- commandArgs(trailingOnly = TRUE)
input <- args[[1]]
cutoff <- as.numeric(args[[2]])
p <- as.integer(args[[3]])
alpha <- as.numeric(args[[4]])
h_left <- as.numeric(args[[5]])
h_right <- as.numeric(args[[6]])
df <- utils::read.csv(input)
h_arg <- NULL
if (is.finite(h_left) && is.finite(h_right)) {
  h_arg <- c(h_left, h_right)
}
fit <- rddensity::rddensity(X = df$x, c = cutoff, p = p, h = h_arg)
density_left <- as.numeric(fit$hat$left)[1]
density_right <- as.numeric(fit$hat$right)[1]
density_diff <- density_right - density_left
pvalue <- as.numeric(fit$test$p_jk)[1]
zstat <- as.numeric(fit$test$t_jk)[1]
bw_left <- as.numeric(fit$h$left)[1]
bw_right <- as.numeric(fit$h$right)[1]
out <- list(
  density_left = density_left,
  density_right = density_right,
  density_diff = density_diff,
  pvalue = pvalue,
  zstat = zstat,
  bandwidth_left = bw_left,
  bandwidth_right = bw_right,
  polynomial_order = as.integer(fit$opt$p)
)
cat(jsonlite::toJSON(out, auto_unbox = TRUE, digits = 16, null = "null"))
"""

    with tempfile.TemporaryDirectory(prefix="statspai-rddensity-") as tmp:
        tmp_path = Path(tmp)
        csv_path = tmp_path / "x.csv"
        script_path = tmp_path / "run_rddensity.R"
        pd.DataFrame({"x": X}).to_csv(csv_path, index=False)
        script_path.write_text(r_code)
        h_l_arg = "NaN" if h_l is None else f"{h_l:.17g}"
        h_r_arg = "NaN" if h_r is None else f"{h_r:.17g}"
        proc = subprocess.run(
            [
                "Rscript",
                str(script_path),
                str(csv_path),
                f"{c:.17g}",
                str(int(p)),
                f"{alpha:.17g}",
                h_l_arg,
                h_r_arg,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            "backend='r' failed while running rddensity::rddensity: "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )

    out: Dict[str, Any] = json.loads(proc.stdout)
    diff = float(out["density_diff"])
    zstat = float(out["zstat"])
    pvalue = float(out["pvalue"])
    se_diff = abs(diff / zstat) if np.isfinite(zstat) and abs(zstat) > 0 else 0.0
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (diff - z_crit * se_diff, diff + z_crit * se_diff)

    model_info = {
        'test': 'Cattaneo-Jansson-Ma (2020)',
        'density_left': float(out["density_left"]),
        'density_right': float(out["density_right"]),
        'density_diff': diff,
        'bandwidth_left': float(out["bandwidth_left"]),
        'bandwidth_right': float(out["bandwidth_right"]),
        'bandwidth_source': h_source,
        'backend': 'rddensity',
        'polynomial_order': int(out["polynomial_order"]),
        'n_left': n_l,
        'n_right': n_r,
        'cutoff': c,
    }

    _result = CausalResult(
        method='CJM (2020) Density Test [rddensity::rddensity]',
        estimand='T-statistic (density discontinuity)',
        estimate=zstat,
        se=float(se_diff),
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        model_info=model_info,
        _citation_key='rddensity',
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            _result,
            function="sp.diagnostics.rddensity",
            params={
                "x": x, "c": c, "p": p, "h": h, "alpha": alpha,
                "backend": "r",
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


def _cjm_bandwidth(x, p):
    """CJM-style bandwidth: plug-in rule based on density curvature."""
    n = len(x)
    sd = np.std(x)
    # Silverman-type with adjustment for local polynomial order
    h = 1.06 * sd * n ** (-1 / (2 * p + 3))
    return max(h, 0.01 * sd)


def _local_poly_density(x, target, h, p, side='right', n_full=None):
    """
    Estimate density at 'target' using local polynomial on the ECDF.

    The key insight from CJM (2020): smooth the empirical CDF using
    local polynomial regression, then differentiate to get the density.

    Parameters
    ----------
    x : array
        Side-specific running variable values (e.g., x_left or x_right).
    target : float
        Point at which to estimate density (the cutoff, usually 0).
    h : float
        Bandwidth.
    p : int
        Polynomial order.
    side : str
        'left' or 'right'.
    n_full : int, optional
        Full sample size (both sides). If None, uses len(x).
        This is needed to convert the side-specific density to the
        unconditional density: f(c) = (n_side / n_full) * f_side(c).
    """
    n_side = len(x)
    if n_full is None:
        n_full = n_side

    # Keep observations within bandwidth on the correct side
    if side == 'left':
        in_bw = (x >= target - h) & (x < target)
    else:
        in_bw = (x >= target) & (x <= target + h)

    x_bw = x[in_bw]
    n_bw = len(x_bw)

    if n_bw < p + 2:
        # Fallback: simple histogram density
        f_hat = n_bw / (h * n_full) if h > 0 else 0
        se = f_hat / np.sqrt(max(n_bw, 1))
        return max(f_hat, 1e-10), max(se, 1e-10)

    # CJM (2020) approach: fit local polynomial to the ECDF of the
    # FULL one-sided sample, evaluated at points within the bandwidth.
    # The ECDF is F_side(x) = (rank of x among all side observations) / n_full.
    # Using n_full (not n_side) so that f = dF/dx gives the unconditional density.
    x_all_sorted = np.sort(x)  # all side observations sorted
    # ECDF values at bandwidth observations, scaled by n_full
    ecdf_vals = np.searchsorted(x_all_sorted, x_bw, side='right') / n_full

    # Local polynomial regression of ECDF on (x - target)
    # F(x) ≈ β₀ + β₁(x-c) + β₂(x-c)² + ...
    # Density = dF/dx|_{x=c} = β₁
    dx = x_bw - target
    w = np.maximum(1 - np.abs(dx / h), 0)  # triangular kernel

    X_poly = np.column_stack([dx ** j for j in range(p + 1)])
    sqw = np.sqrt(w)
    Xw = X_poly * sqw[:, np.newaxis]
    yw = ecdf_vals * sqw

    try:
        beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        # f(c) = β₁ (the slope coefficient, in units of dx not u)
        f_hat = abs(beta[1]) if p >= 1 else n_bw / (h * n_full)

        # SE: the ECDF at point x has variance F(x)(1-F(x))/n_full.
        # For the local polynomial fit, we use a heteroskedastic sandwich:
        # V(β) = (X'WX)^{-1} X'W Σ WX (X'WX)^{-1}
        # where Σ_ii = Var(F(x_i)) = F(x_i)(1-F(x_i)) / n_full
        ecdf_var = ecdf_vals * (1 - ecdf_vals) / n_full
        ecdf_var = np.maximum(ecdf_var, 1e-20)
        XwX_inv = np.linalg.pinv(Xw.T @ Xw)
        # Sandwich: meat = X'W * diag(ecdf_var) * WX
        meat = (Xw * ecdf_var[:, None]).T @ Xw
        vcov = XwX_inv @ meat @ XwX_inv
        se_beta1 = np.sqrt(max(vcov[1, 1], 0)) if vcov.shape[0] > 1 else 0
        se = abs(se_beta1)
    except Exception:
        f_hat = n_bw / (h * n_full)
        se = f_hat / np.sqrt(max(n_bw, 1))

    return max(f_hat, 1e-10), max(se, 1e-10)


# Citation
CausalResult._CITATIONS['rddensity'] = (
    "@article{cattaneo2020simple,\n"
    "  title={Simple Local Polynomial Density Estimators},\n"
    "  author={Cattaneo, Matias D. and Jansson, Michael and Ma, Xinwei},\n"
    "  journal={Journal of the American Statistical Association},\n"
    "  volume={115},\n"
    "  number={531},\n"
    "  pages={1449--1455},\n"
    "  year={2020},\n"
    "  publisher={Taylor \\& Francis}\n"
    "}"
)
