"""StatsPAI RD density manipulation parity (Python side) -- Module 09.

Runs the **native Python** sp.rddensity (backend="native") on the Lee
2008 senate replica and emits the left/right density estimates and the
density difference at the cutoff. The companion 09_rddensity.R uses
rddensity::rddensity with identical defaults.

Tier: T4 common-conclusion gap.  The native CJM local-polynomial
density estimator uses a dependency-light bandwidth selector that
differs from rddensity's, so the density difference at the cutoff
differs by a small absolute amount (abs ~ 0.05 on this replica) while
the substantive manipulation-test conclusion is identical (both fail to
reject).  The optional backend='r' bridge shells out to rddensity for
users who need the exact reference number; it is a convenience feature,
NOT used here as a parity comparator (comparing the bridge to R would
be circular).
"""
from __future__ import annotations

import statspai as sp

from _common import ParityRecord, dump_csv, write_results


MODULE = "09_rddensity"


def main() -> None:
    df = sp.datasets.lee_2008_senate()
    dump_csv(df, MODULE)

    fit = sp.rddensity(df, x="margin", c=0.0, backend="native")
    mi = fit.model_info

    rows: list[ParityRecord] = [
        ParityRecord(
            module=MODULE, side="py", statistic="density_diff",
            estimate=float(mi["density_diff"]),
            n=int(len(df)),
        ),
        ParityRecord(
            module=MODULE, side="py", statistic="density_left",
            estimate=float(mi["density_left"]), n=int(len(df)),
        ),
        ParityRecord(
            module=MODULE, side="py", statistic="density_right",
            estimate=float(mi["density_right"]), n=int(len(df)),
        ),
        ParityRecord(
            module=MODULE, side="py", statistic="bandwidth_left",
            estimate=float(mi["bandwidth_left"]), n=int(len(df)),
        ),
        ParityRecord(
            module=MODULE, side="py", statistic="bandwidth_right",
            estimate=float(mi["bandwidth_right"]), n=int(len(df)),
        ),
    ]
    _pv = getattr(fit, "pvalue", None)
    if _pv is not None:
        rows.append(ParityRecord(
            module=MODULE, side="py", statistic="test_pvalue",
            estimate=float(_pv), n=int(len(df))))

    write_results(
        MODULE, "py", rows,
        extra={
            "polynomial_order": int(mi.get("polynomial_order", 2)),
            "backend": mi.get("backend", "native"),
            "validation_tier": mi.get("validation_tier"),
            "reference_backend": mi.get("reference_backend"),
            "test_kind": "Cattaneo-Jansson-Ma (2020)",
            "tier": "T4",
            "native_note": (
                "Headline row is the NATIVE Python CJM density test "
                "(backend='native'). Its dependency-light bandwidth selector "
                "differs from rddensity's, so the density difference differs "
                "by a small absolute amount while the manipulation-test "
                "conclusion is identical (both fail to reject) -- graded T4. "
                "The optional backend='r' bridge is a convenience feature, "
                "not a parity comparator."
            ),
        },
    )


if __name__ == "__main__":
    main()
