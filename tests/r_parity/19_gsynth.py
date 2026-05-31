"""StatsPAI Generalized SCM parity (Python side) -- Module 19.

Runs the **native Python** generalized SCM (``sp.gsynth(backend='native')``)
on the Basque-Country replica (Xu 2017). The companion 19_gsynth.R uses
``gsynth::gsynth`` on the same CSV.

Tier: T4 documented identification gap.  The interactive-fixed-effects
factor model is identified only up to a rotation, and the number of
factors is chosen by cross-validation that can differ across
implementations; the native Python optimum differs from gsynth's by
rel ~ 0.59 on this replica.  The optional ``backend='gsynth'`` R bridge
is a convenience feature, NOT used here as a parity comparator
(comparing the bridge to R would be circular).
"""
from __future__ import annotations

import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results


MODULE = "19_gsynth"


def main() -> None:
    df = sp.datasets.basque_terrorism()
    df = df.copy()
    df["treated_indicator"] = (
        (df["region"] == "Basque Country") & (df["year"] >= 1970)
    ).astype(int)
    dump_csv(df, MODULE)

    fit = sp.gsynth(
        df,
        outcome="gdppc",
        unit="region",
        time="year",
        treated_unit="Basque Country",
        treatment_time=1970,
        backend="native",
        seed=PARITY_SEED,
    )

    rows: list[ParityRecord] = [
        ParityRecord(
            module=MODULE, side="py", statistic="att_gsynth",
            estimate=float(fit.estimate),
            se=float(fit.se) if fit.se is not None else None,
            n=int(len(df)),
        ),
    ]
    _nf = fit.model_info.get("n_factors")
    if _nf is not None:
        rows.append(ParityRecord(
            module=MODULE, side="py", statistic="n_factors",
            estimate=float(_nf), n=int(len(df))))
    _pre = fit.model_info.get("pre_treatment_rmse",
                              fit.model_info.get("pre_rmse"))
    if _pre is not None:
        rows.append(ParityRecord(
            module=MODULE, side="py", statistic="pre_rmse",
            estimate=float(_pre), n=int(len(df))))

    write_results(
        MODULE, "py", rows,
        extra={
            "method": "gsynth (Xu 2017)",
            "backend": fit.model_info.get("backend", "native"),
            "n_donors": int(fit.model_info.get("n_donors", 0)),
            "tier": "T4",
            "native_note": (
                "Headline row is the NATIVE Python generalized SCM "
                "(backend='native'). The interactive-FE factor model is "
                "identified only up to rotation and the CV factor count can "
                "differ from gsynth's, so the residual gap is graded T4, not "
                "a parity pass. The optional backend='gsynth' R bridge is a "
                "convenience feature, not a parity comparator."
            ),
        },
    )


if __name__ == "__main__":
    main()
