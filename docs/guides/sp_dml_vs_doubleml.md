# `sp.dml` and the DoubleML reference implementation

`sp.dml` is StatsPAI's implementation of the Double/Debiased Machine
Learning (DML) framework of Chernozhukov et al. (2018). The canonical
reference implementations are the [DoubleML][doubleml] R package
(Bach, Kurz, Chernozhukov, Spindler & Klaassen, JSS 108(3), 2024)
and the [doubleml-for-py][doubleml-py] Python package
(Bach, Chernozhukov, Kurz & Spindler, JMLR 23(53), 2022).

This guide covers (1) how the `sp.dml` API maps onto DoubleML, (2)
where the numbers agree, and (3) what the differences are and why
they exist.

[doubleml]: https://github.com/DoubleML/doubleml-for-r
[doubleml-py]: https://github.com/DoubleML/doubleml-for-py

## API mapping

`sp.dml` is a single dispatcher; the `model=` argument selects the
estimator and matches one of DoubleML's classes one-to-one.

| Model | `sp.dml(...)` | `doubleml.DoubleML*(...)` | DoubleML R |
| --- | --- | --- | --- |
| Partially linear regression | `sp.dml(model='plr', ml_g=..., ml_m=...)` | `DoubleMLPLR(ml_l=..., ml_m=...)` | `DoubleMLPLR$new(...)` |
| Interactive regression (AIPW ATE) | `sp.dml(model='irm', ml_g=..., ml_m=...)` | `DoubleMLIRM(ml_g=..., ml_m=...)` | `DoubleMLIRM$new(...)` |
| Partially linear IV | `sp.dml(model='pliv', instrument=..., ml_g=..., ml_m=..., ml_r=...)` | `DoubleMLPLIV(ml_l=..., ml_m=..., ml_r=...)` | `DoubleMLPLIV$new(...)` |
| Interactive IV (LATE) | `sp.dml(model='iivm', instrument=..., ml_g=..., ml_m=..., ml_r=...)` | `DoubleMLIIVM(ml_g=..., ml_m=..., ml_r=...)` | `DoubleMLIIVM$new(...)` |

Anything that takes a scikit-learn estimator on the DoubleML side also
works in `sp.dml`. StatsPAI additionally accepts string shortcuts
(`'rf'`, `'gbm'`, `'lasso'`, `'ridge'`, `'linear'`, `'logistic'`,
`'xgb'`, `'lgbm'`) for the common nuisance learners.

## Same-DGP, same-seed numerical agreement

The fixture in `tests/reference_parity/_fixtures/dml_data.csv` is a
seed-42 DGP with `n=1000`, `p=10`, true treatment effect `θ=0.5`. The
external parity test
[`tests/external_parity/test_dml_python_parity.py`](../../tests/external_parity/test_dml_python_parity.py)
runs `sp.dml` and `doubleml-for-py` on this fixture with identical
scikit-learn learners (`LassoCV(cv=5)` for regression,
`LogisticRegressionCV(cv=5)` for binary propensity) under a fixed
seed.

| Model | `sp.dml` (StatsPAI 1.16.0) | `doubleml-for-py` 0.11.3 | `DoubleML` R 1.0.2 (cv.glmnet) |
| --- | --- | --- | --- |
| **PLR** (continuous d) | **+0.5590 ± 0.0331** | **+0.5590 ± 0.0331** | +0.5368 ± 0.0335 |
| **IRM** (binary d, AIPW) | -0.0191 ± 0.0766 | -0.0267 ± 0.0742 | +0.0066 ± 0.0744 |

- **PLR**: `sp.dml` and `doubleml-for-py` agree to 4 decimal places on
  both the point estimate and the standard error. This is exact
  numerical equivalence under shared scikit-learn folds — both
  implementations use the same Neyman-orthogonal score and the same
  CV-fold partition under a fixed seed. The slight deviation from the
  R reference (~4%) reflects glmnet's penalty path differing
  fractionally from scikit-learn's `LassoCV`; the R reference is
  pinned by [`tests/reference_parity/test_dml_parity.py`](../../tests/reference_parity/test_dml_parity.py)
  to within 7% relative.

- **IRM**: All three implementations land statistically at zero (the
  truth for this DGP). `sp.dml` and `doubleml-for-py` differ by
  0.0076 absolute — well within one standard error — because the two
  AIPW score implementations make different choices on propensity
  trimming and score normalization. The external parity test
  tolerates 0.05 absolute deviation, which is roughly two-thirds of
  one SE on this fixture.

## When to expect divergence

`sp.dml` deviates from `doubleml-for-py` only in implementation
details that the original Chernozhukov et al. (2018) score leaves
unspecified:

1. **Propensity trimming**: `sp.dml` clips propensities to
   `[0.01, 0.99]` by default; `doubleml-for-py` uses `[0, 1]` unless
   the user supplies `trimming_threshold`. Setting matching
   thresholds eliminates the IRM gap.
2. **Repeated cross-fitting aggregation**: `n_rep > 1` aggregates by
   median in both. With `n_rep=1` the seed fully determines folds.
3. **Convenience defaults**: `sp.dml`'s string aliases
   (`ml_g='rf'`, etc.) map to specific scikit-learn configurations
   (e.g. `RandomForestRegressor(n_estimators=200)`). Passing an
   explicit sklearn estimator removes this layer.

For audit-grade numerical equivalence, supply the same
`sklearn`-compatible estimators to both libraries (as the external
parity test does), and PLR / PLIV / IIVM agree to machine precision
under a fixed seed. IRM agreement is up to AIPW trimming choices.

## Running the parity tests yourself

```bash
pip install -e ".[dev]"
pip install doubleml          # not a runtime dependency of StatsPAI

# Python-side parity (sp.dml vs doubleml-for-py)
pytest tests/external_parity/test_dml_python_parity.py -v

# R-side parity (requires R + DoubleML + mlr3 installed locally)
pytest tests/reference_parity/test_dml_parity.py -v
```

The R-side fixture (`dml_R.json`) was generated once on R 4.5.2 with
`DoubleML` 1.0.2 + `mlr3learners` 0.14.0 + `cv_glmnet`; rerun
`_generate_dml.R` only when the DGP itself changes.

## References

- **Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E.,
  Hansen, C., Newey, W. & Robins, J. (2018).** Double/debiased
  machine learning for treatment and structural parameters. *The
  Econometrics Journal*, 21(1), C1–C68. [@chernozhukov2018double]
- **Bach, P., Chernozhukov, V., Kurz, M.S. & Spindler, M. (2022).**
  DoubleML — An Object-Oriented Implementation of Double Machine
  Learning in Python. *Journal of Machine Learning Research*,
  23(53), 1–6. [@bach2022doubleml]
- **Bach, P., Kurz, M.S., Chernozhukov, V., Spindler, M. & Klaassen,
  S. (2024).** DoubleML — An Object-Oriented Implementation of
  Double Machine Learning in R. *Journal of Statistical Software*,
  108(3), 1–56. DOI: [`10.18637/jss.v108.i03`](https://doi.org/10.18637/jss.v108.i03). [@bach2024doubleml]
