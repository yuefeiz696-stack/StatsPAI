"""StatsPAI original-data parity (Python side) -- Module 05.

Runs sp.rdrobust on the *original* rdrobust::rdrobust_RDsenate
extract (Lee 2008 House-elections sharp RD on the Senate
vote-share margin).
"""
from __future__ import annotations

import statspai as sp

from _common import OrigRecord, read_csv, write_results


MODULE = "05_lee_original"


def _scalar(v):
    """Coerce an sp.rdrobust output cell (DataFrame / Series / scalar) to float."""
    if hasattr(v, "iloc"):
        return float(v.iloc[0])
    return float(v)


def main() -> None:
    df = read_csv(MODULE)
    n = len(df)

    fit = sp.rdrobust(df, y="y", x="x", c=0.0,
                      kernel="triangular", bwselect="mserd")

    # The sp.rdrobust result object exposes .estimate / .se for the
    # bias-corrected robust headline; older fixtures may store the
    # conventional column under .coef. We pull both where available.
    rows = [
        OrigRecord(
            module=MODULE, side="py", statistic="rd_jump_robust",
            estimate=_scalar(fit.estimate), se=_scalar(fit.se),
            n=n, published=None,
            citation="rdrobust::rdrobust bias-corrected robust point and SE",
        ),
    ]

    # Conventional point if exposed.
    if hasattr(fit, "coef") and getattr(fit, "coef") is not None:
        try:
            conv_beta = _scalar(fit.coef.loc["Conventional", "Coeff"])
            conv_se = _scalar(fit.se_table.loc["Conventional", "Std. Err."]) \
                if hasattr(fit, "se_table") and fit.se_table is not None \
                else None
            rows.insert(0, OrigRecord(
                module=MODULE, side="py", statistic="rd_jump_conventional",
                estimate=conv_beta, se=conv_se,
                n=n, published=7.99,
                citation="Lee (2008) Table 1; CCT (2014) Table 4 conventional sharp-RD jump",
            ))
        except Exception:
            pass

    write_results(MODULE, "py", rows,
                  extra={"data_source": "rdrobust::rdrobust_RDsenate",
                         "n_obs": n})


if __name__ == "__main__":
    main()
