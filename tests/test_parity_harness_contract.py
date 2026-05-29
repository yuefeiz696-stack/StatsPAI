"""Fast metadata tests for the cross-language parity harness.

These tests do not run R or Stata. They make the already-materialized
parity artifacts a pytest-enforced contract so the published tables,
registered tolerances, and validation APIs cannot silently drift apart.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
R_PARITY = ROOT / "tests" / "r_parity"
R_RESULTS = R_PARITY / "results"
STATA_RESULTS = ROOT / "tests" / "stata_parity" / "results"


def _load_compare() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "statspai_r_parity_compare_for_tests",
        R_PARITY / "compare.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _module_stems(root: Path, suffix: str) -> set[str]:
    return {path.stem[: -len(suffix)] for path in root.glob(f"*{suffix}.json")}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _readme_module_numbers(path: Path) -> set[str]:
    numbers: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\|\s*(\d{2})\s*\|", line)
        if match:
            numbers.add(match.group(1))
    return numbers


def _headline_rows(
    compare: ModuleType,
    module: str,
) -> tuple[dict[str, Any], list[Any]]:
    cfg = compare.HEADLINE[module]
    diffs = compare.collect(module)
    filtered = [diff for diff in diffs if cfg["headline_filter"](diff)]
    return cfg, filtered or diffs


def test_parity_artifact_inventory_has_explicit_contracts():
    compare = _load_compare()
    py_modules = _module_stems(R_RESULTS, "_py")
    r_modules = _module_stems(R_RESULTS, "_R")
    stata_modules = _module_stems(STATA_RESULTS, "_Stata")

    assert len(py_modules) >= 51
    assert len(py_modules & r_modules) >= 50
    assert len(stata_modules) >= 44
    assert py_modules == set(compare.TOLERANCES)
    assert py_modules == set(compare.HEADLINE)
    assert py_modules - r_modules == {"50_xtabond"}
    assert set(compare.STATA_SKIP_REASON) == py_modules - stata_modules
    assert set(compare.STATA_HEADLINE_GAP_EXCEPTIONS) <= stata_modules

    for module in sorted(py_modules & r_modules):
        assert compare.collect(module), f"{module} has no joined py/R rows"


def test_parity_json_rows_keep_the_joinable_schema():
    artifacts = [
        *R_RESULTS.glob("*_py.json"),
        *R_RESULTS.glob("*_R.json"),
        *STATA_RESULTS.glob("*_Stata.json"),
    ]
    assert artifacts

    for path in artifacts:
        payload = _read_json(path)
        if path.name.endswith("_py.json"):
            module, side = path.stem[:-3], "py"
        elif path.name.endswith("_R.json"):
            module, side = path.stem[:-2], "R"
        else:
            module, side = path.stem[:-6], "Stata"

        assert payload["module"] == module
        assert payload["side"] == side
        assert payload["rows"], f"{path.name} has no parity rows"

        seen: set[str] = set()
        for row in payload["rows"]:
            assert row["module"] == module
            assert row["side"] == side
            assert isinstance(row["statistic"], str) and row["statistic"]
            assert row["statistic"] not in seen
            seen.add(row["statistic"])

            for key in ("estimate", "se", "ci_lo", "ci_hi"):
                value = row.get(key)
                assert value is None or isinstance(value, (int, float))
                assert not isinstance(value, float) or math.isfinite(value)


def test_headline_passes_are_inside_registered_r_tolerance():
    compare = _load_compare()
    r_modules = _module_stems(R_RESULTS, "_R")

    for module in sorted(r_modules):
        cfg, rows = _headline_rows(compare, module)
        metric = cfg["metric"]
        values = [
            getattr(row, metric)
            for row in rows
            if getattr(row, metric) is not None
        ]
        assert values, f"{module} headline has no {metric} values"

        if "PASS" not in cfg["verdict"]:
            continue
        tolerance = compare.TOLERANCES[module].get(metric)
        assert tolerance is not None, f"{module} PASS lacks {metric} tolerance"
        assert max(values) <= tolerance, (
            f"{module} R headline {metric}={max(values):.6g} exceeds {tolerance}"
        )


def test_stata_headline_over_budget_modules_are_explicitly_registered():
    compare = _load_compare()
    stata_modules = _module_stems(STATA_RESULTS, "_Stata")
    over_budget: dict[str, tuple[float, float]] = {}

    for module in sorted(stata_modules):
        if not compare.collect(module):
            continue
        cfg, rows = _headline_rows(compare, module)
        metric = cfg["metric"]
        stata_metric = "rel_est_st" if metric == "rel_est" else "abs_est_st"
        values = [
            getattr(row, stata_metric)
            for row in rows
            if getattr(row, stata_metric) is not None
        ]
        if not values:
            continue
        tolerance = compare.TOLERANCES[module].get(metric)
        if tolerance is not None and max(values) > tolerance:
            over_budget[module] = (max(values), tolerance)

    assert set(over_budget) == set(compare.STATA_HEADLINE_GAP_EXCEPTIONS)


def test_bjs_stata_convention_gap_is_narrow_and_documented():
    compare = _load_compare()
    row = {
        diff.statistic: diff
        for diff in compare.collect("16_bjs")
    }["att_bjs"]

    assert row.rel_est < 1e-6
    assert 0.15 < row.rel_est_st < 0.18
    assert "autosample/aggregation" in (
        compare.STATA_HEADLINE_GAP_EXCEPTIONS["16_bjs"]
    )


def test_panel_sfa_stata_gap_is_parameterisation_not_slope_drift():
    compare = _load_compare()
    rows = {
        diff.statistic: diff
        for diff in compare.collect("29_panel_sfa")
    }

    assert rows["beta_lnk"].rel_est_st < 1e-3
    assert rows["beta_lnl"].rel_est_st < 1e-3
    assert 0.015 < rows["beta_intercept"].rel_est_st < 0.02
    assert rows["sigma_u"].rel_est_st > 0.25
    assert "parameterisation" in (
        compare.STATA_HEADLINE_GAP_EXCEPTIONS["29_panel_sfa"]
    )


def test_generated_parity_tables_are_in_sync_with_comparator():
    compare = _load_compare()
    modules = sorted(
        path.stem.replace("_py", "") for path in R_RESULTS.glob("*_py.json")
    )

    assert (R_RESULTS / "parity_table.md").read_text(encoding="utf-8") == (
        compare.render_md(modules)
    )
    assert (R_RESULTS / "parity_table.tex").read_text(encoding="utf-8") == (
        compare.render_tex(modules)
    )
    assert (R_RESULTS / "parity_table_3way.md").read_text(encoding="utf-8") == (
        compare.render_md_3way(modules)
    )
    assert (R_RESULTS / "parity_table_3way.tex").read_text(encoding="utf-8") == (
        compare.render_tex_3way(modules)
    )


def test_parity_readmes_match_current_artifact_inventory():
    py_numbers = {
        module.split("_", 1)[0] for module in _module_stems(R_RESULTS, "_py")
    }
    stata_numbers = {
        module.split("_", 1)[0] for module in _module_stems(STATA_RESULTS, "_Stata")
    }

    r_readme = R_PARITY / "README.md"
    stata_readme = ROOT / "tests" / "stata_parity" / "README.md"

    assert py_numbers <= _readme_module_numbers(r_readme)
    assert stata_numbers == _readme_module_numbers(stata_readme)
    assert "Modules (36)" not in r_readme.read_text(encoding="utf-8")
    assert "21 of 36" not in stata_readme.read_text(encoding="utf-8")
    assert (R_PARITY / "50_xtabond.R").exists()
