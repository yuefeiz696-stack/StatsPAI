"""StatsPAI classical SCM parity (Python side) -- Module 07.

Runs the native Python classical SCM
(``sp.synth(method='classic', backend='native')``) on the Basque-Country
replica and emits the average post-treatment gap, the pre-treatment RMSE,
and the donor weights. The companion 07_scm.R uses ``Synth::synth`` on
the same CSV with the standard pre-treatment-outcomes-only predictor
spec.

Tier: documented convention gap. The Basque donor-weight vector is not
unique under the default outcomes-only convention, so the native Python
QP path can land on a different equally plausible donor vector than
``Synth``. Native correctness is separately certified on the uniquely
identified SCM DGP in module 52_scm_unique; users who need exact R
numbers can call the optional ``backend='synth'`` bridge.
"""
from __future__ import annotations

import statspai as sp

from _common import ParityRecord, dump_csv, write_results


MODULE = "07_scm"


def main() -> None:
    df = sp.datasets.basque_terrorism()
    dump_csv(df, MODULE)

    # Match R Synth's exact ADH(2010) specification: every pre-treatment
    # year as its own outcome special-predictor with nested V-W
    # optimisation. Under this common specification the native Python
    # solver tracks Synth to rel ~ 0.024 on the replica (and to rel
    # ~ 2e-4 on the original Synth::basque extract); the residual is the
    # documented local-optimum non-uniqueness of the outer V program.
    pre_years = list(range(1955, 1970))
    fit = sp.synth(
        df,
        outcome="gdppc",
        unit="region",
        time="year",
        treated_unit="Basque Country",
        treatment_time=1970,
        method="classic",
        backend="native",
        special_predictors=[("gdppc", yr, "mean") for yr in pre_years],
        v_method="nested",
        placebo=False,
    )

    rows: list[ParityRecord] = [
        ParityRecord(
            module=MODULE, side="py", statistic="avg_post_gap",
            estimate=float(fit.estimate),
            se=float(fit.se),
            n=int(len(df)),
        ),
        ParityRecord(
            module=MODULE, side="py", statistic="pre_treatment_rmse",
            estimate=float(fit.model_info["pre_treatment_rmse"]),
            n=int(len(df)),
        ),
    ]

    # weights is a DataFrame with columns ['unit', 'weight'] listing
    # the active donors only; absent donors carry implicit weight 0.
    # Emit a row for every donor in the donor pool so the comparator
    # can match against the R-side weight vector.
    weights_df = fit.model_info["weights"]
    donor_pool = sorted(
        df.loc[df["region"] != "Basque Country", "region"].unique().tolist()
    )
    weight_map = dict(zip(weights_df["unit"], weights_df["weight"]))
    for unit in donor_pool:
        rows.append(
            ParityRecord(
                module=MODULE, side="py",
                statistic=f"weight_{unit}",
                estimate=float(weight_map.get(unit, 0.0)),
                n=int(len(df)),
            )
        )

    write_results(
        MODULE, "py", rows,
        extra={
            "method": "classic",
            "backend": fit.model_info.get("backend", "native"),
            "validation_tier": fit.model_info.get("validation_tier"),
            "reference_backend": fit.model_info.get("reference_backend"),
            "treatment_time": 1970,
            "treated_unit": "Basque Country",
            "n_donors": int(fit.model_info["n_donors"]),
            "placebo": False,
            "tier": "T4",
            "native_note": (
                "Headline row uses backend='native'. The Basque donor-weight "
                "solution is not unique under outcomes-only predictors, so "
                "the native QP path can differ from Synth's nested-V optimum. "
                "Native correctness is separately certified on a uniquely "
                "identified DGP in module 52_scm_unique; backend='synth' is "
                "available when exact R Synth numbers are required."
            ),
        },
    )


if __name__ == "__main__":
    main()
