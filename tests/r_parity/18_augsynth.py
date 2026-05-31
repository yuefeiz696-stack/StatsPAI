"""StatsPAI Augmented SCM parity (Python side) -- Module 18.

Runs the **native Python** augmented SCM (``sp.augsynth(backend='native')``)
on the Basque-Country replica (Ben-Michael, Feller & Rothstein 2021).
The companion 18_augsynth.R uses ``augsynth::augsynth`` on the same CSV.

Tier: T4 documented convention gap.  The ridge-augmented SCM estimand
is identified but the ridge penalty and outcome-model conventions are
not uniquely pinned across implementations; the native Python optimum
differs from augsynth's by rel ~ 0.34 on this replica.  The optional
``backend='augsynth'`` R bridge is a convenience feature, NOT used here
as a parity comparator (comparing the bridge to R would be circular).
"""
from __future__ import annotations

import statspai as sp

from _common import ParityRecord, dump_csv, write_results


MODULE = "18_augsynth"


def main() -> None:
    df = sp.datasets.basque_terrorism()
    dump_csv(df, MODULE)

    fit = sp.augsynth(
        df,
        outcome="gdppc",
        unit="region",
        time="year",
        treated_unit="Basque Country",
        treatment_time=1970,
        backend="native",
    )

    rows: list[ParityRecord] = [
        ParityRecord(
            module=MODULE, side="py", statistic="att_augmented",
            estimate=float(fit.estimate),
            se=float(fit.se),
            n=int(len(df)),
        ),
    ]
    _pre = fit.model_info.get("pre_treatment_rmse",
                              fit.model_info.get("pre_rmspe"))
    if _pre is not None:
        rows.append(
            ParityRecord(
                module=MODULE, side="py", statistic="pre_rmspe",
                estimate=float(_pre),
                n=int(len(df)),
            )
        )

    write_results(
        MODULE, "py", rows,
        extra={
            "method": "augmented",
            "backend": fit.model_info.get("backend", "native"),
            "n_donors": int(fit.model_info.get("n_donors", 0)),
            "tier": "T4",
            "native_note": (
                "Headline row is the NATIVE Python augmented SCM "
                "(backend='native'). The residual gap vs augsynth is a "
                "documented ridge/outcome-model convention, graded T4, not "
                "a parity pass. The optional backend='augsynth' R bridge is "
                "a convenience feature, not a parity comparator."
            ),
        },
    )


if __name__ == "__main__":
    main()
