"""StatsPAI Synthetic DID parity (Python side) -- Module 12.

Runs the **native Python** synthetic-DID estimator
(``sp.sdid(backend='native')``) on the california_prop99 replica
(Arkhangelsky et al. 2021). The companion 12_sdid.R uses
``synthdid::synthdid_estimate`` on the same CSV.

Tier: T4 documented convention gap.  The SDID estimand is identified
but the regularisation parameter zeta and the unit/time weight optima
are not uniquely pinned across implementations, so the native Python
optimum differs from synthdid's by a documented regularisation
convention (rel ~ 0.08 on this replica).  StatsPAI also ships an
optional ``backend='synthdid'`` R bridge for users who need the exact R
numbers; it is a convenience feature, NOT used here as a parity
comparator (comparing the bridge to R would be circular).
"""
from __future__ import annotations

import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results


MODULE = "12_sdid"


def main() -> None:
    df = sp.datasets.california_prop99()
    dump_csv(df, MODULE)

    fit = sp.sdid(
        df,
        outcome="cigsale",
        unit="state",
        time="year",
        treated_unit="California",
        treatment_time=1989,
        backend="native",
        seed=PARITY_SEED,
    )

    rows: list[ParityRecord] = [
        ParityRecord(
            module=MODULE, side="py", statistic="att_sdid",
            estimate=float(fit.estimate),
            se=float(fit.se),
            ci_lo=float(fit.ci[0]) if fit.ci is not None else None,
            ci_hi=float(fit.ci[1]) if fit.ci is not None else None,
            n=int(len(df)),
        )
    ]

    write_results(
        MODULE, "py", rows,
        extra={
            "method": "sdid",
            "n_treated": int(fit.model_info["n_treated"]),
            "n_control": int(fit.model_info["n_control"]),
            "T_pre": int(fit.model_info["T_pre"]),
            "T_post": int(fit.model_info["T_post"]),
            "se_method": fit.model_info["se_method"],
            "backend": fit.model_info.get("backend", "native"),
            "validation_tier": fit.model_info.get("validation_tier"),
            "reference_backend": fit.model_info.get("reference_backend"),
            "tier": "T4",
            "native_note": (
                "Headline row is the NATIVE Python SDID (backend='native'). "
                "The residual gap vs synthdid is a documented regularisation "
                "(zeta) convention, graded T4, not a parity pass. The "
                "optional backend='synthdid' R bridge is a convenience "
                "feature, not a parity comparator."
            ),
        },
    )


if __name__ == "__main__":
    main()
