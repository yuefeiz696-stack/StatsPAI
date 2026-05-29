# Stability and Validation Tiers

StatsPAI now separates **API lifecycle** from **numerical validation evidence**. This is the main correction to the older catalogue: `stability='stable'` no longer means "R/Stata parity-grade" by itself.

## Three Fields

| Field | Scope | Meaning |
| --- | --- | --- |
| `stability` | whole function API | `stable`, `experimental`, or `deprecated` |
| `validation_status` | evidence for numerical output | `certified`, `validated`, `api_stable`, `experimental`, or `deprecated` |
| `limitations` | parameter/variant gaps | documented unsupported variants inside an otherwise usable function |

Use `stability` when you care about public API compatibility. Use `validation_status` when you care about publication-grade numerical evidence.

## Stability

- `stable`: public signature is locked under SemVer minor releases.
- `experimental`: method/API may shift across minor versions.
- `deprecated`: scheduled for removal; replacement should be documented in `MIGRATION.md`.

## Validation

- `certified`: cross-language or published-reference parity evidence exists, usually from `tests/r_parity/`, `tests/stata_parity/`, or published-replication fixtures.
- `validated`: analytic/reference parity tests exist in `tests/reference_parity/` or `tests/external_parity/`, but the function is not in the main Track A R/Stata harness.
- `api_stable`: stable public API, but no machine-readable parity evidence has been attached yet.
- `experimental`: mirrors `stability='experimental'`.
- `deprecated`: mirrors `stability='deprecated'`.

## Filtering

```python
import statspai as sp

sp.list_functions()                              # all registered functions
sp.list_functions(stability="stable")            # stable API
sp.list_functions(validation_status="certified") # parity-backed functions
sp.agent_cards(validation_status="certified")    # parity-backed agent cards

spec = sp.describe_function("regress")
spec["stability"]          # "stable"
spec["validation_status"]  # "certified"
spec["validation_notes"]   # parity artifact / reference notes
```

```bash
statspai list --stability experimental
statspai list --validation certified
statspai describe rdrobust
```

`sp.help()` prints both `STABILITY` and `VALIDATION` count blocks. Per-function help shows `Stability:`, `Validation:`, `Evidence:`, and `Known limitations` when available.

## Promotion Path

1. Promote `experimental` to `stable` when the public API is ready for SemVer compatibility.
2. Promote `api_stable` to `validated` when analytic/reference parity tests exist.
3. Promote `validated` to `certified` when the function enters the cross-language or published-reference parity harness.
4. Remove a `limitation` only when the unsupported variant lands with its own test.

## Current Limitation Hotspots

These are machine-readable through `sp.describe_function(name)["limitations"]` and should be treated as the priority backlog for production hardening:

- `callaway_santanna`: repeated cross-sections currently support only `estimator="reg"` with `control_group="nevertreated"`.
- `rdrobust`: observation-level weights are reserved and raise `NotImplementedError`.
- `hal_tmle`: `variant="projection"` is reserved and raises `NotImplementedError`.
- `network_exposure`: only `design="bernoulli"` is implemented.
- `etwfe`: `panel=False` with `cgroup="nevertreated"` is not implemented.
- `continuous_did`: `method="cgs"` is an MVP without full CGS parity.
- `did_multiplegt_dyn`: experimental MVP; switch-off events, analytical IF variance, and heteroskedastic weights are not implemented.

## Auditing

```bash
python scripts/stability_audit.py
python scripts/stability_audit.py --unbacked
python scripts/stability_audit.py --check
python scripts/stability_audit.py --json
```

Programmatic evidence summaries:

```python
sp.validation_report()
sp.coverage_matrix(level="parity")
sp.parity_gap_report()
```

`sp.parity_gap_report()` parses the already-generated 3-way parity table and reports documented convention gaps, missing Stata siblings, priorities, and next actions.

*Last updated: v1.13.1+validation split (2026-05-05).*
