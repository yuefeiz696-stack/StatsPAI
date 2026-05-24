# JOSS Validation Dossier

This dossier collects reviewer-facing evidence for StatsPAI's readiness as
research software. It is intentionally factual and reproducible.

## Project Status

- Repository: <https://github.com/brycewang-stanford/StatsPAI>
- Package archive: <https://doi.org/10.5281/zenodo.19933900>
- PyPI: <https://pypi.org/project/StatsPAI/>
- License: MIT, with a plain-text `LICENSE` file in the repository.
- Current release at the time of this dossier: `1.15.5`, released on
  2026-05-21.
- Public GitHub repository creation date: 2025-07-26.

## Software Scope

StatsPAI exposes a unified Python interface for causal inference and applied
econometrics. As of release `1.15.5`, the registry reports 1,018 public
functions across 80 submodules:

```bash
python scripts/registry_stats.py --check
```

The registry and schema layer are part of the public surface. They support
programmatic discovery through `sp.list_functions()`, `sp.describe_function()`,
and `sp.function_schema()`.

## Validation Assets

The repository includes several independent validation tracks:

- Unit and integration tests across the main estimator families.
- R parity modules under `tests/r_parity/`.
- Stata parity modules under `tests/stata_parity/`.
- Reference-parity checks under `tests/reference_parity/`.
- Original-paper replay fixtures under `tests/orig_parity/`.
- Monte Carlo coverage checks under `tests/coverage_monte_carlo/`.
- Snapshot tests for publication-table output under `tests/output_snapshots/`.
- Citation and bibliography audits under `tools/`.

The maintained local full-suite report records:

```text
5200 passed, 98 skipped, 13 deselected, 1 xfailed, 2 xpassed
```

on Python 3.9.6 for the default local suite as of 2026-05-17. The exact report
is stored in `test_results_full_suite.md`.

## Parity And Replication Anchors

StatsPAI includes validation fixtures for common teaching and replication
benchmarks, including:

- Card-style returns-to-schooling IV estimates.
- LaLonde / Dehejia-Wahba job-training benchmarks.
- Lee-style close-election regression discontinuity.
- Callaway-Sant'Anna difference-in-differences examples.
- California Proposition 99 synthetic-control examples.

Known convention differences are documented in parity reports rather than
hidden. For example, bandwidth selectors, regularisation constants, small-sample
standard-error conventions, and fold-split randomness are recorded in the
R-parity report where they affect exact numerical matching.

## Research Use

At submission time, StatsPAI is being used in working-paper workflows connected
to the Stanford Rural Education Action Program and related empirical policy
evaluation work. No peer-reviewed research article using StatsPAI has yet been
published. The current impact claim is therefore based on credible near-term
research use, reproducible validation materials, public package distribution,
and reviewer-verifiable examples rather than published downstream citations.

## Commercial Downstream Disclosure

StatsPAI Inc. is the legal entity associated with the project. CoPaper.AI is a
commercial downstream product that may call the MIT-licensed StatsPAI package.
The StatsPAI package itself is permanently open source under the MIT license.
This is an open-core / commercial-downstream arrangement: the research software
submitted to JOSS remains open, while commercial products can build on it under
the same license terms available to all users.

## Reproducible Checks

From a repository checkout:

```bash
python -m pip install -e ".[dev,plotting]"
python -m pytest tests/test_ols.py tests/test_did.py tests/test_registry.py -q --no-cov
python scripts/registry_stats.py --check
python scripts/schema_quality.py
python tools/audit_bib_duplicates.py --strict
python tools/audit_bib_coverage.py --strict-dangling --hide-orphans
python -m build
python -m twine check dist/*
```

For a shorter package-level check, use the reviewer guide in
`docs/joss_reviewer_guide.md`.

