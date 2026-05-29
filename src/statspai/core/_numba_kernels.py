"""
Numba-accelerated computational kernels for StatsPAI.

Provides JIT-compiled hot-path routines for OLS estimation, sandwich
covariance matrices, and clustered standard errors.  When Numba is not
installed the module exposes pure-NumPy fallbacks so the rest of the
package never breaks.

Usage inside StatsPAI
---------------------
>>> from statspai.core._numba_kernels import ols_fit, sandwich_hc, cluster_meat
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
#  Numba availability detection
# --------------------------------------------------------------------------- #

try:
    from numba import njit, prange  # type: ignore[import-untyped]

    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False

    # Transparent no-op decorator so the pure-Python definitions below
    # compile without any change in call signature.
    def njit(*args, **kwargs):  # type: ignore[misc]
        """Identity decorator when numba is absent."""
        def _wrap(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return _wrap

    class _FakePrange:
        """Mimic ``numba.prange`` as a plain ``range``."""
        def __new__(cls, *a, **kw):
            return range(*a, **kw)

    prange = _FakePrange  # type: ignore[assignment,misc]


_NUMBA_CACHE = HAS_NUMBA and Path(__file__).exists()


# --------------------------------------------------------------------------- #
#  Optional Rust HDFE backend — provides cluster_meat with Rayon parallelism
#  over clusters. Falls through to the numba kernel below if the wheel is not
#  built (in which case numerics are bit-identical to the previous release).
# --------------------------------------------------------------------------- #
try:
    import statspai_hdfe as _rust_hdfe  # type: ignore

    _HAS_RUST_CLUSTER = hasattr(_rust_hdfe, "cluster_meat")
except ImportError:  # pragma: no cover
    _rust_hdfe = None  # type: ignore[assignment]
    _HAS_RUST_CLUSTER = False


# --------------------------------------------------------------------------- #
#  Core OLS kernel
# --------------------------------------------------------------------------- #

@njit(cache=_NUMBA_CACHE)
def _xtx(X: np.ndarray) -> np.ndarray:
    """Compute X'X using explicit loops (cache-friendly for tall X)."""
    n, k = X.shape
    out = np.zeros((k, k))
    for i in range(n):
        for j in range(k):
            for m in range(j, k):
                out[j, m] += X[i, j] * X[i, m]
    # Symmetrise
    for j in range(k):
        for m in range(j + 1, k):
            out[m, j] = out[j, m]
    return out


@njit(cache=_NUMBA_CACHE)
def _xty(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute X'y."""
    n, k = X.shape
    out = np.zeros(k)
    for i in range(n):
        for j in range(k):
            out[j] += X[i, j] * y[i]
    return out


def ols_fit(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast OLS: beta = (X'X)^{-1} X'y.

    Returns
    -------
    params : ndarray (k,)
    fitted : ndarray (n,)
    residuals : ndarray (n,)
    """
    XtX = _xtx(X)
    Xty = _xty(X, y)
    try:
        params = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        params = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
    fitted = X @ params
    residuals = y - fitted
    return params, fitted, residuals


# --------------------------------------------------------------------------- #
#  Sandwich (HC) covariance helpers
# --------------------------------------------------------------------------- #

@njit(cache=_NUMBA_CACHE)
def _sandwich_meat_hc0(X: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """Meat of HC0 sandwich: X' diag(e^2) X."""
    n, k = X.shape
    out = np.zeros((k, k))
    for i in range(n):
        e2 = residuals[i] * residuals[i]
        for j in range(k):
            xj = X[i, j] * e2
            for m in range(j, k):
                out[j, m] += xj * X[i, m]
    for j in range(k):
        for m in range(j + 1, k):
            out[m, j] = out[j, m]
    return out


def sandwich_hc(
    X: np.ndarray,
    residuals: np.ndarray,
    XtX_inv: np.ndarray,
    hc_type: str = "hc1",
) -> np.ndarray:
    """
    Heteroskedasticity-robust covariance matrix (HC0–HC3).

    Parameters
    ----------
    X : (n, k) array
    residuals : (n,) array
    XtX_inv : (k, k) array – precomputed (X'X)^{-1}
    hc_type : str
        One of 'hc0', 'hc1', 'hc2', 'hc3'.

    Returns
    -------
    (k, k) covariance matrix
    """
    n, k = X.shape
    hc_type = hc_type.lower()

    if hc_type == "hc0":
        meat = _sandwich_meat_hc0(X, residuals)
    elif hc_type == "hc1":
        meat = _sandwich_meat_hc0(X, residuals) * (n / (n - k))
    elif hc_type in ("hc2", "hc3"):
        h = np.sum(X * (X @ XtX_inv), axis=1)  # diagonal of hat matrix
        if hc_type == "hc2":
            w = residuals ** 2 / (1 - h)
        else:
            w = residuals ** 2 / (1 - h) ** 2
        meat = (X * w[:, None]).T @ X
    else:
        raise ValueError(f"Unknown hc_type: {hc_type}")

    return XtX_inv @ meat @ XtX_inv


# --------------------------------------------------------------------------- #
#  Clustered standard errors (fast path)
# --------------------------------------------------------------------------- #

@njit(cache=_NUMBA_CACHE)
def _cluster_meat_sorted(
    X: np.ndarray,
    residuals: np.ndarray,
    cluster_starts: np.ndarray,
    cluster_ends: np.ndarray,
) -> np.ndarray:
    """
    Compute cluster-robust meat matrix when data is pre-sorted by cluster.

    Parameters
    ----------
    cluster_starts, cluster_ends : 1-d int arrays of length G
        cluster_starts[g] is the first row of cluster g;
        cluster_ends[g] is one-past-the-last row.
    """
    k = X.shape[1]
    G = cluster_starts.shape[0]
    meat = np.zeros((k, k))
    score_g = np.zeros(k)

    for g in range(G):
        # zero out
        for j in range(k):
            score_g[j] = 0.0
        for i in range(cluster_starts[g], cluster_ends[g]):
            for j in range(k):
                score_g[j] += X[i, j] * residuals[i]
        for j in range(k):
            for m in range(j, k):
                meat[j, m] += score_g[j] * score_g[m]
    # symmetrise
    for j in range(k):
        for m in range(j + 1, k):
            meat[m, j] = meat[j, m]
    return meat


def cluster_meat(
    X: np.ndarray,
    residuals: np.ndarray,
    cluster_ids: np.ndarray,
) -> np.ndarray:
    """
    Cluster-robust meat matrix.

    Sorts data by *cluster_ids* once, then delegates to either the
    Rust+Rayon kernel (preferred when ``statspai_hdfe`` is built) or
    the numba JIT kernel (always-on fallback).

    Parameters
    ----------
    X : (n, k)
    residuals : (n,)
    cluster_ids : (n,) integer or object cluster labels

    Returns
    -------
    meat : (k, k) array
    """
    order = np.argsort(cluster_ids, kind="mergesort")
    X_s = np.ascontiguousarray(X[order], dtype=np.float64)
    r_s = np.ascontiguousarray(residuals[order], dtype=np.float64)
    c_s = cluster_ids[order]

    # Identify cluster boundaries
    change = np.empty(len(c_s), dtype=np.bool_)
    change[0] = True
    change[1:] = c_s[1:] != c_s[:-1]
    starts = np.where(change)[0].astype(np.intp)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = len(c_s)

    if _HAS_RUST_CLUSTER:
        # Rust kernel takes int64 cluster boundaries.
        return _rust_hdfe.cluster_meat(
            X_s,
            r_s,
            starts.astype(np.int64),
            ends.astype(np.int64),
        )
    return _cluster_meat_sorted(X_s, r_s, starts, ends)


# --------------------------------------------------------------------------- #
#  HAC (Newey-West) kernel — vectorised NumPy, no Numba needed
# --------------------------------------------------------------------------- #

def hac_meat(
    X: np.ndarray,
    residuals: np.ndarray,
    max_lags: int | None = None,
) -> np.ndarray:
    """
    Newey-West HAC meat matrix with Bartlett kernel.

    Parameters
    ----------
    X : (n, k)
    residuals : (n,)
    max_lags : int or None
        If None, uses floor(4*(n/100)^{2/9}).

    Returns
    -------
    (k, k) meat matrix
    """
    n = X.shape[0]
    if max_lags is None:
        max_lags = int(np.floor(4 * (n / 100) ** (2 / 9)))

    # The Newey-West HAC meat is the unnormalised autocovariance sum
    # S = sum_t m_t m_t' + sum_{j=1..L} w_j * (sum_t m_t m_{t-j}'
    # + sum_t m_{t-j} m_t'), so the sandwich V = (X'X)^{-1} S (X'X)^{-1}
    # is on the same scale as OLS/HC. The pre-fix kernel divided Gamma_j
    # by n, which made V too small by a factor of n and SE too small by
    # sqrt(n) — on n = 200 the parity test against R sandwich::NeweyWest
    # and Stata `newey` showed sp HAC SE ~14x smaller than the textbook
    # answer (parity finding #13, 2026-05-28). Same family of bug as
    # the sp.qreg Powell sandwich (finding #8).
    moments = X * residuals[:, None]
    gamma0 = moments.T @ moments
    total = gamma0.copy()

    for j in range(1, max_lags + 1):
        gamma_j = moments[j:].T @ moments[:-j]
        w = 1 - j / (max_lags + 1)
        total += w * (gamma_j + gamma_j.T)

    return total
