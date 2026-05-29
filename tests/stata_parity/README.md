# `tests/stata_parity/` — cross-language parity harness against Stata

This directory is the **StatsPAI ↔ Stata** sibling of
[`tests/r_parity/`](../r_parity/): each module pair runs the same
calibrated replica on both sides, dumps a full-precision JSON
result, and `tests/r_parity/compare.py` joins the three sides
(StatsPAI, R, Stata) into a single 3-way Track A parity table for
the JSS Appendix B.

The harness is read by the same `compare.py` that drives the R
side — there is **one** comparator and **one** tolerance budget
(`compare.py::TOLERANCES`). Parity is a property of the estimator,
not of the reference language, so we deliberately do not register
a separate budget for the Stata comparison. Known Stata convention
gaps that exceed the shared headline budget are explicitly enumerated
in `compare.py::STATA_HEADLINE_GAP_EXCEPTIONS`.

## What's here

```
tests/stata_parity/
├── README.md
├── _common.do            # shared scaffolding: JSON writer (file-based, survives `mata clear`)
├── _quick_compare.py     # ad-hoc 3-way comparator while developing modules
├── NN_<method>.do        # one .do per module
├── logs/                 # Stata's per-run .smcl/.log + the JSON-row tmp files
└── results/
    └── NN_<method>_Stata.json   # full-precision results, joined by compare.py
```

Each `.do` file imports `../r_parity/data/NN_<name>.csv` (the same
bytes the R side reads), runs the canonical Stata reference, and
writes one row per parity statistic to
`results/NN_<name>_Stata.json` via the helpers in `_common.do`.

## Modules covered (44 of 51)

| # | Method                       | StatsPAI                       | Stata reference                                              |
| --- | --- | --- | --- |
| 01 | OLS + HC1 SE                  | `sp.regress`                   | `regress, vce(robust)`                                       |
| 02 | 2SLS + HC1 SE                 | `sp.iv`                        | `ivregress 2sls, vce(robust) small`                          |
| 03 | HDFE 2-way FE                 | `sp.fast.feols`                | `reghdfe, absorb(...) vce(unadjusted)`                       |
| 04 | CS-DiD simple ATT             | `sp.callaway_santanna`         | `csdid + estat simple, method(reg)`                          |
| 05 | Sun-Abraham event study       | `sp.sun_abraham`               | `eventstudyinteract`                                         |
| 06 | RD CCT bias-corrected         | `sp.rdrobust`                  | `rdrobust`                                                   |
| 07 | Classical SCM                 | `sp.synth(method="classic")`   | `synth ..., trunit(...) trperiod(...) nested`                |
| 09 | RD density (CJM)              | `sp.rddensity`                 | `rddensity`                                                  |
| 10 | Honest DiD bounds (FLCI)      | `sp.honest_did`                | `honestdid, b(...) vcov(...) numpre(...) mvec(...) delta(sd)`|
| 11 | PSM 1:1 NN                    | `sp.psm`                       | `teffects psmatch, atet nneighbor(1)`                        |
| 12 | Synthetic DiD                 | `sp.synth(method="sdid")`      | `sdid ..., vce(placebo)`                                     |
| 14 | OLS + cluster (CR1)           | `sp.regress(robust="cluster")` | `regress, vce(cluster ...)`                                  |
| 15 | HDFE + cluster                | `sp.fast.feols(vcov="cluster")`| `reghdfe, absorb(...) vce(cluster ...)`                      |
| 16 | BJS imputation                | `sp.bjs_pretrend_joint`        | `did_imputation, autosample`                                 |
| 17 | Wooldridge ETWFE              | `sp.wooldridge_did`            | `jwdid + estat simple`                                       |
| 20 | Goodman-Bacon decomposition   | `sp.bacon_decomposition`       | `bacondecomp, ddetail`                                       |
| 21 | Honest-DiD relative-mags      | `sp.honest_did(restriction="relative_magnitudes")` | `honestdid, ... delta(rm)` |
| 22 | sensemakr robustness          | `sp.sensemakr`                 | `sensemakr depvar regs, treat(...) benchmark(...) kd(1) ky(1)` |
| 23 | E-value                       | `sp.evalue`                    | `evalue rr`                                                 |
| 24 | Cox proportional hazards      | `sp.survival.cox`              | `stcox`                                                     |
| 25 | Linear mixed model            | `sp.mixed`                     | `mixed ..., reml`                                            |
| 26 | GLMM logit (Laplace)          | `sp.melogit`                   | `melogit ..., intmethod(laplace)`                           |
| 27 | GLMM AGHQ (n=8)               | `sp.melogit(nAGQ=8)`           | `melogit ..., intpoints(8)`                                  |
| 28 | Stochastic frontier (cross-sec) | `sp.frontier`                | `frontier, distribution(hnormal)`                            |
| 29 | Panel SFA Pitt-Lee            | `sp.xtfrontier`                | `xtfrontier, ti`                                             |
| 30 | Blinder-Oaxaca decomposition  | `sp.oaxaca_blinder`            | `oaxaca`                                                     |
| 33 | VAR                           | `sp.var`                       | `var`                                                        |
| 34 | Local projections             | `sp.local_projections`         | horizon-by-horizon `regress`; `lpirf` recorded in extras     |
| 35 | Panel FE/RE/Hausman           | `sp.panel`                     | `xtreg, fe/re` + `hausman`                                   |
| 36 | Causal mediation              | `sp.mediation`                 | `paramed`                                                    |
| 37 | PPML + HDFE                   | `sp.ppmlhdfe`                  | `ppmlhdfe`                                                   |
| 39 | ARIMA(2,0,0)                  | `sp.arima`                     | `arima`                                                      |
| 40 | Quantile reg (median)         | `sp.qreg`                      | `qreg`                                                       |
| 41 | Tobit (left-censored)         | `sp.tobit`                     | `tobit, ll(0)`                                               |
| 42 | Negative binomial             | `sp.nbreg`                     | `nbreg`                                                      |
| 43 | Heckman 2-step                | `sp.heckman`                   | `heckman, twostep`                                           |
| 44 | Multinomial logit             | `sp.mlogit`                    | `mlogit`                                                     |
| 45 | Ordered logit                 | `sp.ologit`                    | `ologit`                                                     |
| 46 | Conditional logit             | `sp.clogit`                    | `clogit, group(...)`                                         |
| 47 | PPML + 3-way HDFE             | `sp.ppmlhdfe`                  | `ppmlhdfe, absorb(origin dest year)`                         |
| 48 | Binary probit                 | `sp.probit`                    | `probit`                                                     |
| 49 | Ordered probit                | `sp.oprobit`                   | `oprobit`                                                    |
| 50 | Arellano-Bond GMM             | `sp.xtabond`                   | `xtabond`                                                    |
| 51 | Newey-West HAC OLS            | `sp.regress(robust="hac")`     | `newey`                                                      |

### Modules **without** a Stata sibling

These have no authoritative Stata port we can compare against
without fabricating one — `compare.py::STATA_SKIP_REASON` records
the reason and the 3-way table prints it explicitly:

- **08 DML PLR** — Stata has `ddml`, but we do not treat it as a
  canonical reference for the published DoubleML R algorithm.
- **13 causal forest** — no Stata port of `grf`.
- **18 augsynth** — no Stata port of the augmented SCM.
- **19 gsynth** — no Stata port of the generalised SCM.
- **31 DFL reweighting** — no canonical Stata port selected.
- **32 RIF / UQR** — `rifhdreg` requires a GitHub install and is not
  part of the portable Stata parity baseline.
- **38 DR-DID** — Stata `drdid` is Ferman's different DR formula, not
  the Sant'Anna-Zhao estimator used by the R reference.

## Running

End-to-end run for a single module (assumes the matching
`tests/r_parity/NN_<name>.py` has already produced the CSV in
`tests/r_parity/data/`):

```bash
cd tests/stata_parity
/Applications/Stata/StataMP.app/Contents/MacOS/stata-mp -b -q do 11_psm.do
python3 ../r_parity/compare.py
```

Run everything:

```bash
cd tests/stata_parity
for dofile in [0-9][0-9]_*.do; do
  /Applications/Stata/StataMP.app/Contents/MacOS/stata-mp -b -q do "${dofile}"
done
python3 ../r_parity/compare.py
```

The same critical Stata smoke path is available through pytest:

```bash
pytest tests/test_parity_runtime.py -m external_parity_runtime --no-cov
```

## Stata environment

- **Edition tested**: Stata 18 BE (Basic Edition; matrix max = 800).
  None of the 44 modules trip the BE matrix limit.
- **`set type double`** is forced in `_common.do` so
  `import delimited` reads the CSV bytes at full IEEE-754 precision;
  without it, Stata's float default would cost 4-5 orders of
  magnitude in parity (1e-12 → 1e-8 on OLS).
- **JSON writer**: file-based (under `logs/<module>.rows.tmp`) rather
  than Mata-resident, because several Stata commands (`rdrobust`,
  `csdid`, `sdid`, others) call `mata mata clear` internally and
  would wipe a Mata accumulator mid-run.

## Required SSC / community packages

```stata
ssc install ivreg2 ranktest csdid drdid did_imputation eventstudyinteract \
    jwdid hdfe synth rdrobust rddensity honestdid bacondecomp \
    sfcross sfpanel sensemakr avar ppmlhdfe paramed evalue
```

`reghdfe`, `sdid`, `psmatch2`, and `oaxaca` were already on the test
machine; `mixed`, `melogit`, `xtfrontier`, `frontier`, `regress`,
`ivregress`, `teffects psmatch`, `xtreg`, `var`, `arima`, `qreg`,
`tobit`, `nbreg`, `heckman`, `mlogit`, `ologit`, `clogit`, `probit`,
`oprobit`, `xtabond`, and `newey` are Stata built-ins.

## How the JSS paper uses this

[`Paper-JSS/manuscript/sections/appendix.tex`](../../Paper-JSS/manuscript/sections/appendix.tex)
`\input`s [`manuscript/tables/appendix_b_parity.tex`](../../Paper-JSS/manuscript/tables/appendix_b_parity.tex),
which is a copy of `tests/r_parity/results/parity_table_3way.tex`
refreshed by `compare.py`. Re-running `compare.py` after any module
change is sufficient to keep the appendix in sync.
