"""
GRF-style inference add-ons for the StatsPAI CausalForest.

Implements:

- :func:`test_calibration`: Chernozhukov-Demirer-Duflo-Fernández-Val
  (2020, CDDF henceforth) "Best Linear Predictor of CATE" calibration
  test. Regresses the orthogonal pseudo-outcome on the predicted CATE
  (the "mean forest prediction") and a demeaned version of it (the
  "differential forest prediction"); tests the joint/individual
  significance. Under correct ATE calibration the first coefficient is
  1; under non-trivial CATE heterogeneity the second is >0.

- :func:`rate`: Rank-Average Treatment Effect (Yadlowsky, Fleming,
  Shah, Tibshirani, Wager 2023). Integrates the difference between the
  ATE on the top-``q`` predicted-CATE fraction and the overall ATE,
  producing an AUTOC (area-under the TOC curve) statistic with a
  valid normal-approximation confidence interval built from half-
  sample splits.

- :func:`honest_variance`: subsample-splitting variance estimate for
  aggregate quantities (ATE / GATE) computed on a forest with honest
  sample splits. Uses the half-sample delta-method bootstrap of Wager-
  Athey (2018).
- :func:`average_treatment_effect`: GRF-style ATE/ATT/ATC/ATO aggregation
  of CATE predictions with effective sample size and normal CIs.
- :func:`forest_diagnostics`: overlap and CATE-distribution diagnostics.

These functions are stateless and take a fitted :class:`CausalForest`
object (plus, for some, outcome / treatment / feature arrays). They
never mutate the forest.

References
----------
Chernozhukov, V., Demirer, M., Duflo, E., Fernández-Val, I. (2020).
"Generic Machine Learning Inference on Heterogeneous Treatment Effects
in Randomized Experiments, with an Application to Immunization in India."
NBER WP 24678. [@chernozhukov2020generic]

Yadlowsky, S., Fleming, S., Shah, N., Brunskill, E., Wager, S. (2021).
"Evaluating Treatment Prioritization Rules via Rank-Weighted Average
Treatment Effects." arXiv:2111.07966.

Athey, S., Tibshirani, J., Wager, S. (2019). "Generalized Random
Forests." Annals of Statistics, 47(2), 1148-1178. [@athey2019surrogate]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from .causal_forest import CausalForest


# ======================================================================
# test_calibration (Chernozhukov-Demirer-Duflo-Fernández-Val 2020)
# ======================================================================


def calibration_test(
    forest: "CausalForest",
    X: Optional[np.ndarray] = None,
    Y: Optional[np.ndarray] = None,
    T: Optional[np.ndarray] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """BLP-of-CATE calibration test (Chernozhukov-Demirer-Duflo-Fernandez-Val 2020).

    Pseudo-outcome regression:

        Ψ_i = α + β₁ · τ̂(X_i) + β₂ · (τ̂(X_i) - Eτ̂) + ε_i

    where ``Ψ_i`` is the orthogonal AIPW pseudo-outcome built from the
    forest's own propensity/outcome-model predictions. Hypothesis:

        H₀^{(1)}: β₁ = 1   (well-calibrated mean forest prediction)
        H₀^{(2)}: β₂ = 0   (no systematic CATE heterogeneity)

    Rejecting H₀^{(2)} is the headline finding — it demonstrates that
    the forest captures *real* heterogeneity rather than noise.

    Parameters
    ----------
    forest : fitted CausalForest
    X, Y, T : optional arrays
        If not given, the forest's stored training arrays are used.
    alpha : float
        Significance level for reported CIs.

    Returns
    -------
    DataFrame
        Rows ``beta_mean`` and ``beta_differential`` with ``coef``,
        ``se``, ``t``, ``p``, ``ci_low``, ``ci_high``.
    """
    if not forest.fitted_:
        raise ValueError("Forest must be fitted before calibration testing.")

    X_ = np.asarray(X if X is not None else forest._X_original, dtype=np.float64)
    Y_ = np.asarray(
        Y if Y is not None else getattr(forest, "_Y_original", None),
        dtype=np.float64,
    ).ravel()
    T_ = np.asarray(
        T if T is not None else getattr(forest, "_T_original", None),
        dtype=np.float64,
    ).ravel()

    tau_hat = np.asarray(forest.effect(X_), dtype=np.float64).ravel()
    tau_bar = float(tau_hat.mean())
    tau_dem = tau_hat - tau_bar

    # Pseudo-outcome: AIPW-style. Built from the forest's stashed
    # cross-fitted nuisance predictions when available; falls back to
    # Horvitz-Thompson otherwise.
    m_hat, e_hat = _get_nuisances(forest, X_, Y_, T_)
    e_hat = np.clip(e_hat, 0.02, 0.98)
    psi = _construct_pseudo_outcome(Y_, T_, e_hat, m_hat, forest, X_)

    # CDDF (2020) canonical regression: pseudo-outcome on mean forest
    # prediction and differential forest prediction. Intercept is
    # omitted because ``tau_dem`` already has zero mean — including it
    # would introduce a rank-1 collinearity with the ``tau_hat`` column.
    # β_1 tests H0: perfect calibration (β_1 = 1); β_2 tests H0: no
    # heterogeneity in the forest's predicted CATE (β_2 = 0).
    n = len(psi)
    D = np.column_stack([tau_hat, tau_dem])
    DtD = D.T @ D
    # Guard against a near-singular design (e.g. constant τ̂): add a
    # tiny ridge and note it in the output.
    ridge = 1e-10 * np.trace(DtD) / 2
    DtD_inv = np.linalg.inv(DtD + ridge * np.eye(2))
    beta = DtD_inv @ (D.T @ psi)
    resid = psi - D @ beta

    # HC1 robust SE (White 1980) with degrees-of-freedom correction.
    k = D.shape[1]
    scores = D * resid[:, None]
    meat = scores.T @ scores
    V_hc1 = (n / max(n - k, 1)) * DtD_inv @ meat @ DtD_inv
    se = np.sqrt(np.maximum(np.diag(V_hc1), 0.0))

    z = stats.norm.ppf(1 - alpha / 2)
    # Calibration: β_1 tests against 1; Heterogeneity: β_2 tests against 0.
    names = ["mean_forest_prediction", "differential_forest_prediction"]
    null_values = [1.0, 0.0]
    out = pd.DataFrame(
        {
            "coef": beta,
            "se": se,
            "null": null_values,
        },
        index=names,
    )
    out["t"] = (out["coef"] - out["null"]) / out["se"].replace(0, np.nan)
    out["p"] = 2 * (1 - stats.norm.cdf(np.abs(out["t"].fillna(0))))
    out["ci_low"] = out["coef"] - z * out["se"]
    out["ci_high"] = out["coef"] + z * out["se"]
    return out


def _construct_pseudo_outcome(
    Y: np.ndarray,
    T: np.ndarray,
    e_hat: np.ndarray,
    m_hat: np.ndarray,
    forest: "CausalForest",
    X: np.ndarray,
) -> np.ndarray:
    """AIPW-style CATE pseudo-outcome. Uses forest-estimated nuisances if
    available; falls back to Horvitz-Thompson otherwise. Returns Ψ_i
    interpretable as an unbiased signal for τ(X_i).
    """
    mu1 = getattr(forest, "_mu1_insample", None)
    mu0 = getattr(forest, "_mu0_insample", None)
    if mu1 is not None and mu0 is not None:
        mu1 = np.asarray(mu1).ravel()
        mu0 = np.asarray(mu0).ravel()
        mu_T = T * mu1 + (1 - T) * mu0
        return (mu1 - mu0) + (T - e_hat) * (Y - mu_T) / (e_hat * (1 - e_hat))
    # Horvitz-Thompson signal (centered on e_hat-adjusted residual)
    return T * (Y - m_hat) / e_hat - (1 - T) * (Y - m_hat) / (1 - e_hat)


def _get_nuisances(
    forest: "CausalForest",
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the fitted nuisance predictions from the CausalForest, with
    fallbacks if they are unavailable.
    """
    m_hat = getattr(forest, "_m_insample", None)
    e_hat = getattr(forest, "_e_insample", None)
    if m_hat is None:
        # Fall back to sample mean of Y
        m_hat = np.full_like(Y, Y.mean())
    if e_hat is None:
        e_hat = np.full_like(T, T.mean())
    return np.asarray(m_hat, dtype=np.float64).ravel(), np.asarray(e_hat, dtype=np.float64).ravel()


# ======================================================================
# RATE (Yadlowsky-Fleming-Shah-Tibshirani-Wager 2023)
# ======================================================================


def rate(
    forest: "CausalForest",
    X: Optional[np.ndarray] = None,
    Y: Optional[np.ndarray] = None,
    T: Optional[np.ndarray] = None,
    target: str = "AUTOC",
    q_grid: int = 100,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Rank-Average Treatment Effect (Yadlowsky et al. 2023).

    Let ``S(x) = τ̂(x)`` denote a prioritisation score (higher = higher
    priority). Define the TOC (targeting operator characteristic) curve:

        TOC(q) = E[τ(X) | S(X) ≥ Q_{1-q}(S)] - E[τ(X)]

    i.e. the expected CATE among the top-``q`` fraction minus the
    population ATE. Two scalar summaries are supported:

    - **AUTOC**: ``∫₀¹ TOC(q) dq`` — the *unweighted* area under the
      TOC curve. Emphasises prioritisation performance uniformly
      across the quantile range.
    - **QINI**:  ``∫₀¹ q · TOC(q) dq`` — down-weights narrow top
      fractions; closer to the classical uplift / Qini coefficient.

    Estimation
    ----------
    The DR-RATE estimator from Yadlowsky et al. uses an AIPW pseudo-
    outcome ``Ψ_i`` (computed from the forest's own cross-fitted
    nuisance predictions) and reduces AUTOC / Qini to a weighted sum:

        AUTOC_hat = (1/n) Σ_i Ψ_i · w_{AUTOC}(R_i / n) - Ψ̄
        QINI_hat  = (1/n) Σ_i Ψ_i · w_{QINI}(R_i / n)  - (1/2) Ψ̄

    where ``R_i`` is the descending rank of ``S(X_i)`` and the weights
    are closed-form rank kernels. This representation makes the
    estimator a sample mean of per-observation contributions φ_i, so
    the variance admits the standard influence-function form

        Var(AUTOC_hat) = (1/(n(n-1))) Σ_i (φ_i - φ̄)²

    which replaces the conservative half-sample estimator used in the
    earlier draft of this function.

    Parameters
    ----------
    forest : fitted CausalForest
    X, Y, T : arrays, optional
        If omitted, falls back to the forest's stored training arrays.
    target : {'AUTOC', 'QINI'}
    q_grid : int
        Number of quantile grid points used to report the TOC curve.
        Does not affect the point estimate or SE (those are computed
        from ranks exactly).
    alpha : float
    seed : int, optional
        Ignored; kept for API backwards compatibility.

    Returns
    -------
    dict with keys ``estimate``, ``se``, ``ci_low``, ``ci_high``,
    ``target``, ``toc_curve`` (``(q_grid, 2)``), ``n``, ``method``.

    References
    ----------
    Yadlowsky, S., Fleming, S., Shah, N., Brunskill, E., Wager, S.
    (2021). "Evaluating Treatment Prioritization Rules via Rank-
    Weighted Average Treatment Effects." arXiv:2111.07966.
    """
    if not forest.fitted_:
        raise ValueError("Forest must be fitted.")
    if target not in ("AUTOC", "QINI"):
        raise ValueError("target must be 'AUTOC' or 'QINI'")

    X_ = np.asarray(X if X is not None else forest._X_original, dtype=np.float64)
    Y_ = np.asarray(
        Y if Y is not None else getattr(forest, "_Y_original", None),
        dtype=np.float64,
    ).ravel()
    T_ = np.asarray(
        T if T is not None else getattr(forest, "_T_original", None),
        dtype=np.float64,
    ).ravel()
    n = len(Y_)

    m_hat, e_hat = _get_nuisances(forest, X_, Y_, T_)
    e_hat = np.clip(e_hat, 0.02, 0.98)
    psi = _construct_pseudo_outcome(Y_, T_, e_hat, m_hat, forest, X_)
    tau_hat = np.asarray(forest.effect(X_)).ravel()
    psi_bar = float(psi.mean())

    # Descending rank by τ̂ (rank 1 = highest priority).  Ties: stable
    # ordering is fine; any rank permutation among ties leaves the
    # weighted sum invariant because the kernel only depends on the
    # fractional rank.
    desc_order = np.argsort(-tau_hat, kind="mergesort")
    rank = np.empty(n, dtype=np.float64)
    rank[desc_order] = np.arange(1, n + 1)  # 1-based rank
    u = rank / n                             # fractional rank ∈ (0, 1]

    # Closed-form rank kernels. Derivation (AUTOC): AUTOC_hat rewrites
    # as the mean over k ∈ {1,..,n} of the top-k mean of Ψ minus Ψ̄.
    # Reordering the double sum gives the weight on observation i as
    # w_i = (1/n) · Σ_{k=R_i}^{n} (1/k) = (H_n - H_{R_i - 1}) / n.
    # The harmonic-number formula is exact for the empirical estimator.
    H = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, n + 1))])
    # For observation i with rank R_i, weight = H_n - H_{R_i - 1}.
    R_int = rank.astype(np.int64)
    w_autoc = H[n] - H[R_int - 1]  # shape (n,)
    # Qini kernel: w_{QINI}(u) = 1 - u (weights decrease linearly with
    # the descending rank). Using the fractional rank keeps the sample
    # average well-calibrated to the continuous population integral.
    w_qini = 1.0 - u

    # Rewrite both estimators as a single sample mean
    #     θ̂ = (1/n) Σ_i Ψ_i · (w_i - c)
    # with c = 1 for AUTOC (because AUTOC_hat = mean(Ψ·w) - Ψ̄) and
    # c = 1/2 for Qini. The per-observation contribution
    #     φ_i = Ψ_i · (w_i - c)
    # depends only on obs i (treating the ranks as conditioning), so
    # its sample variance over n gives the correct influence-function
    # variance of θ̂ — no whole-sample subtraction is needed.
    if target == "AUTOC":
        phi = psi * (w_autoc - 1.0)
    else:  # QINI
        phi = psi * (w_qini - 0.5)
    estimate = float(phi.mean())

    # Influence-function variance: Var_hat(φ) / n, using the n-1
    # denominator for the centred sum of squares.
    phi_centered = phi - phi.mean()
    var_est = float(phi_centered @ phi_centered) / (n * (n - 1)) if n > 1 else float("nan")
    se = float(np.sqrt(max(var_est, 0.0)))
    z = stats.norm.ppf(1 - alpha / 2)
    ci = (estimate - z * se, estimate + z * se)

    # TOC curve for diagnostics (not used in the IF variance path).
    psi_sorted = psi[desc_order]
    cum = np.cumsum(psi_sorted) / np.arange(1, n + 1)
    toc_all = cum - psi_bar  # length-n array, evaluated at q_k = k/n
    q_targets = np.linspace(1.0 / q_grid, 1.0, q_grid)
    idx_sel = np.clip((q_targets * n).astype(np.int64) - 1, 0, n - 1)
    toc_grid = np.column_stack([q_targets, toc_all[idx_sel]])

    return {
        "estimate": estimate,
        "se": se,
        "ci_low": ci[0],
        "ci_high": ci[1],
        "target": target,
        "toc_curve": toc_grid,
        "n": n,
        "method": "Influence-function SE (Yadlowsky et al. 2023)",
    }


# ======================================================================
# Honest subsample variance for aggregate quantities
# ======================================================================


def honest_variance(
    forest: "CausalForest",
    X: Optional[np.ndarray] = None,
    n_splits: int = 25,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Half-sample bootstrap variance of the ATE/GATE estimate.

    Repeatedly partition the sample into two halves, compute the mean
    predicted CATE on each, and aggregate. Returns the sample variance
    of the per-split means divided by the number of splits — a crude
    but robust uncertainty quantifier when the forest's internal
    variance estimator is unavailable.

    Parameters
    ----------
    forest : fitted CausalForest
    X : ndarray, optional
    n_splits : int
        Number of random half-sample draws.
    seed : int, optional

    Returns
    -------
    dict with ``ate``, ``se``, ``ci_low``, ``ci_high`` (95 %).
    """
    if not forest.fitted_:
        raise ValueError("Forest must be fitted.")
    X_ = np.asarray(X if X is not None else forest._X_original, dtype=np.float64)
    tau = np.asarray(forest.effect(X_)).ravel()
    n = len(tau)
    rng = np.random.default_rng(seed)

    means = np.empty(n_splits)
    for s in range(n_splits):
        perm = rng.permutation(n)
        half = perm[: n // 2]
        means[s] = float(tau[half].mean())

    ate = float(tau.mean())
    se = float(np.std(means, ddof=1) / np.sqrt(n_splits))
    z = stats.norm.ppf(0.975)
    return {
        "ate": ate,
        "se": se,
        "ci_low": ate - z * se,
        "ci_high": ate + z * se,
    }


def average_treatment_effect(
    forest: "CausalForest",
    X: Optional[np.ndarray] = None,
    T: Optional[np.ndarray] = None,
    target_sample: str = "all",
    alpha: float = 0.05,
    clip: float = 0.01,
) -> Dict[str, float]:
    """Aggregate CATE predictions into ATE/ATT/ATC/ATO targets.

    This mirrors the most-used ``grf::average_treatment_effect`` targets:
    ``"all"`` (ATE), ``"treated"`` (ATT), ``"control"`` (ATC), and
    ``"overlap"`` (ATO, weighted by ``e(X)(1-e(X))``).

    The estimate is the **doubly-robust AIPW influence-function mean**
    (the estimator grf reports), not a plug-in average of the CATE
    predictions.  Using the forest's own cross-fitted nuisances
    :math:`\\hat m(X)=\\hat E[Y\\mid X]` and :math:`\\hat e(X)=\\hat
    E[T\\mid X]`, the ATE score is

    .. math::
        \\Gamma_i = \\hat\\tau(X_i)
            + \\frac{T_i-\\hat e(X_i)}{\\hat e(X_i)(1-\\hat e(X_i))}
              \\bigl(Y_i-\\hat m(X_i)-(T_i-\\hat e(X_i))\\hat\\tau(X_i)\\bigr),

    and the ATT/ATC scores use the analogous Robins doubly-robust
    weighting.  ``se`` is the influence-function standard error
    :math:`\\mathrm{sd}(\\Gamma)/\\sqrt n`.  When the score cannot be
    formed (out-of-sample ``X`` with no stored nuisances) the function
    falls back to the plug-in CATE average and sets ``method='plug_in'``.

    Parameters
    ----------
    clip : float, default 0.01
        Propensity scores are clipped to ``[clip, 1-clip]`` before the
        inverse-propensity term to stabilise the score under near-overlap
        violations.
    """
    if not forest.fitted_:
        raise ValueError("Forest must be fitted.")

    target = target_sample.lower().strip()
    aliases = {
        "ate": "all",
        "all": "all",
        "att": "treated",
        "treated": "treated",
        "atc": "control",
        "control": "control",
        "ato": "overlap",
        "overlap": "overlap",
    }
    target = aliases.get(target)
    if target is None:
        raise ValueError(
            "target_sample must be one of 'all', 'treated', 'control', "
            "or 'overlap'"
        )

    use_insample = X is None and T is None
    X_ = np.asarray(X if X is not None else forest._X_original, dtype=np.float64)
    tau = np.asarray(forest.effect(X_), dtype=np.float64).ravel()
    T_ = np.asarray(
        T if T is not None else getattr(forest, "_T_original", None),
        dtype=np.float64,
    ).ravel()
    Y_ = np.asarray(getattr(forest, "_Y_original", None), dtype=np.float64).ravel()
    if len(T_) != len(tau):
        raise ValueError("T must have the same length as X/effect predictions.")

    # Outcome and propensity nuisances.  When aggregating on the training
    # sample we reuse the forest's own cross-fitted (cv=3) out-of-fold
    # nuisances m̂ = Ê[Y|X] and ê = Ê[T|X] -- the same quantities grf
    # uses for ``average_treatment_effect`` -- so the ATE/ATT scores are
    # honest doubly-robust influence functions rather than a plug-in mean
    # of the (regularisation-shrunk) CATE predictions.
    m_insample = getattr(forest, "_m_insample", None)
    e_insample = getattr(forest, "_e_insample", None)
    if use_insample and m_insample is not None and e_insample is not None:
        m_hat = np.asarray(m_insample, dtype=np.float64).ravel()
        e_hat = np.asarray(e_insample, dtype=np.float64).ravel()
    else:
        m_hat, e_hat = _get_nuisances(forest, X_, Y_, T_)
        m_hat = np.asarray(m_hat, dtype=np.float64).ravel()
        e_hat = np.asarray(e_hat, dtype=np.float64).ravel()
    e_hat = np.clip(e_hat, clip, 1.0 - clip)
    if len(e_hat) != len(tau) or len(m_hat) != len(tau) or len(Y_) != len(tau):
        # AIPW score is unavailable (out-of-sample without nuisances or a
        # length mismatch); fall back to the plug-in CATE average and flag it.
        return _plug_in_average(tau, T_, e_hat, target, alpha)

    n = int(len(tau))
    z = float(stats.norm.ppf(1 - alpha / 2))
    # Reconstruct the per-arm outcome regressions from (m̂, ê, τ̂):
    #   m = e·μ1 + (1-e)·μ0,  μ1 - μ0 = τ  ⇒  μ0 = m - e·τ,  μ1 = m + (1-e)·τ.
    mu0 = m_hat - e_hat * tau
    mu1 = m_hat + (1.0 - e_hat) * tau
    m_full = m_hat + (T_ - e_hat) * tau  # = E[Y|X,T] under the model

    if target == "all":
        estimand = "ATE"
        psi = tau + (T_ - e_hat) / (e_hat * (1.0 - e_hat)) * (Y_ - m_full)
        estimate = float(psi.mean())
        se = float(psi.std(ddof=1) / np.sqrt(n))
        ess = float(n)
    elif target == "treated":
        estimand = "ATT"
        p1 = float(max(T_.mean(), 1e-8))
        psi = (T_ * (Y_ - mu0) - (1.0 - T_) * (e_hat / (1.0 - e_hat)) * (Y_ - mu0)) / p1
        estimate = float(psi.mean())
        se = float(psi.std(ddof=1) / np.sqrt(n))
        ess = float(T_.sum())
    elif target == "control":
        estimand = "ATC"
        p0 = float(max((1.0 - T_).mean(), 1e-8))
        psi = ((1.0 - T_) * (mu1 - Y_) - T_ * ((1.0 - e_hat) / e_hat) * (mu1 - Y_)) / p0
        estimate = float(psi.mean())
        se = float(psi.std(ddof=1) / np.sqrt(n))
        ess = float((1.0 - T_).sum())
    else:  # overlap (ATO): overlap-weighted average of the AIPW pointwise scores
        estimand = "ATO"
        w = e_hat * (1.0 - e_hat)
        if float(w.sum()) <= 0:
            raise ValueError(
                f"No observations contribute to target_sample={target_sample!r}."
            )
        psi_pt = tau + (T_ - e_hat) / (e_hat * (1.0 - e_hat)) * (Y_ - m_full)
        estimate = float(np.average(psi_pt, weights=w))
        norm_w = w / w.sum()
        se = float(np.sqrt(np.sum((norm_w ** 2) * (psi_pt - estimate) ** 2)))
        ess = float((w.sum() ** 2) / np.sum(w ** 2))

    return {
        "estimate": estimate,
        "se": se,
        "ci_low": estimate - z * se,
        "ci_high": estimate + z * se,
        "target_sample": target,
        "estimand": estimand,
        "method": "aipw",
        "effective_sample_size": ess,
        "n": n,
        "alpha": float(alpha),
        "pscore_min": float(e_hat.min()),
        "pscore_max": float(e_hat.max()),
    }


def _plug_in_average(
    tau: np.ndarray,
    T_: np.ndarray,
    e_hat: np.ndarray,
    target: str,
    alpha: float,
) -> Dict[str, float]:
    """Fallback weighted average of CATE predictions (no AIPW score).

    Used only when the doubly-robust influence function cannot be formed
    (out-of-sample ``X`` with no stored nuisances, or a length mismatch).
    """
    if target == "all":
        weights = np.ones_like(tau)
        estimand = "ATE"
    elif target == "treated":
        weights = (T_ == 1).astype(float)
        estimand = "ATT"
    elif target == "control":
        weights = (T_ == 0).astype(float)
        estimand = "ATC"
    else:
        weights = e_hat * (1.0 - e_hat)
        estimand = "ATO"
    if float(weights.sum()) <= 0:
        raise ValueError("No observations contribute to the requested target_sample.")
    estimate = float(np.average(tau, weights=weights))
    norm_w = weights / weights.sum()
    se = float(np.sqrt(np.sum((norm_w ** 2) * (tau - estimate) ** 2)))
    z = float(stats.norm.ppf(1 - alpha / 2))
    ess = float((weights.sum() ** 2) / np.sum(weights ** 2))
    return {
        "estimate": estimate,
        "se": se,
        "ci_low": estimate - z * se,
        "ci_high": estimate + z * se,
        "target_sample": target,
        "estimand": estimand,
        "method": "plug_in",
        "effective_sample_size": ess,
        "n": int(len(tau)),
        "alpha": float(alpha),
        "pscore_min": float(e_hat.min()) if len(e_hat) else float("nan"),
        "pscore_max": float(e_hat.max()) if len(e_hat) else float("nan"),
    }


def forest_diagnostics(
    forest: "CausalForest",
    X: Optional[np.ndarray] = None,
    T: Optional[np.ndarray] = None,
    propensity_bounds: Tuple[float, float] = (0.05, 0.95),
) -> Dict[str, object]:
    """Return overlap and CATE-distribution diagnostics for a fitted forest."""
    if not forest.fitted_:
        raise ValueError("Forest must be fitted.")

    low, high = propensity_bounds
    if not 0 <= low < high <= 1:
        raise ValueError("propensity_bounds must satisfy 0 <= low < high <= 1.")

    X_ = np.asarray(X if X is not None else forest._X_original, dtype=np.float64)
    tau = np.asarray(forest.effect(X_), dtype=np.float64).ravel()
    T_ = np.asarray(
        T if T is not None else getattr(forest, "_T_original", np.zeros(len(tau))),
        dtype=np.float64,
    ).ravel()
    Y_ = np.asarray(
        getattr(forest, "_Y_original", np.zeros(len(tau))),
        dtype=np.float64,
    ).ravel()
    _m_hat, e_hat = _get_nuisances(forest, X_, Y_, T_)
    e_hat = np.clip(np.asarray(e_hat, dtype=np.float64).ravel(), 0.0, 1.0)
    if len(e_hat) != len(tau):
        e_hat = np.full(len(tau), float(np.mean(T_)))

    overlap = (e_hat >= low) & (e_hat <= high)
    warnings = []
    if e_hat.min() < low or e_hat.max() > high:
        warnings.append(
            "propensity scores outside requested overlap bounds; report "
            "ATE/ATT with caution or use target_sample='overlap'"
        )
    if float(np.std(tau)) < 1e-8:
        warnings.append("predicted CATE is nearly constant; heterogeneity is weak")
    if not getattr(forest, "honest", True):
        warnings.append("forest was fitted with honest=False")

    return {
        "n": int(len(tau)),
        "n_treated": int(np.sum(T_ == 1)),
        "n_control": int(np.sum(T_ == 0)),
        "cate_mean": float(np.mean(tau)),
        "cate_sd": float(np.std(tau, ddof=1)) if len(tau) > 1 else 0.0,
        "cate_min": float(np.min(tau)),
        "cate_max": float(np.max(tau)),
        "cate_iqr": float(np.subtract(*np.percentile(tau, [75, 25]))),
        "pscore_min": float(np.min(e_hat)),
        "pscore_max": float(np.max(e_hat)),
        "overlap_low": float(low),
        "overlap_high": float(high),
        "overlap_share": float(np.mean(overlap)),
        "n_low_pscore": int(np.sum(e_hat < low)),
        "n_high_pscore": int(np.sum(e_hat > high)),
        "warnings": warnings,
    }


# GRF-compatible alias: the R ``grf`` package exposes this test as
# ``test_calibration``. We keep ``test_calibration`` as an alias so
# users familiar with GRF can reach for the same name, while the
# canonical Python name avoids pytest's ``test_*`` auto-discovery.
test_calibration = calibration_test


__all__ = [
    "calibration_test",
    "test_calibration",
    "rate",
    "honest_variance",
    "average_treatment_effect",
    "forest_diagnostics",
]
