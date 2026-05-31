"""Contracts for the aggregate JSS formal-compliance audit."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT = (
    REPO_ROOT
    / "Paper-JSS"
    / "replication"
    / "scripts"
    / "jss_formal_compliance_audit.py"
)
RESULTS = REPO_ROOT / "Paper-JSS" / "replication" / "results"


def test_jss_formal_compliance_audit_maps_official_requirements() -> None:
    env = os.environ.copy()
    env.setdefault("SOURCE_DATE_EPOCH", "1780185600")
    res = subprocess.run(
        [sys.executable, str(AUDIT)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert res.returncode == 0, res.stderr

    payload = json.loads(
        (RESULTS / "jss_formal_compliance_audit.json").read_text()
    )
    checks = {item["requirement"]: item for item in payload["checks"]}

    assert payload["status"] == "PASS"
    assert payload["official_sources_checked"] == "2026-05-31"
    assert len(payload["checks"]) == 16
    assert payload["page_count"] and payload["page_count"] < 30
    assert payload["archive_present"] in {True, False}
    assert payload["tier1_transcript"]["passed"] == payload["tier1_transcript"]["total"]
    assert payload["tier1_transcript"]["total"] >= 20
    assert payload["tier1_transcript"]["within_one_hour"] is True
    install_probe = payload["install_probe"]
    assert install_probe["ok"] is True
    assert install_probe["version"] == "1.16.0"
    assert install_probe["function_count"] >= 1000
    assert {"statspai", "statspai-mcp"} <= set(install_probe["scripts"])
    assert any(name.endswith(".dist-info") for name in install_probe["dist_infos"])
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    assert "Development Status :: 4 - Beta" in pyproject
    assert "Development Status :: 3 - Alpha" not in pyproject
    assert {
        "JSS submissions checklist",
        "JSS information for authors",
        "JSS style guide",
        "JSS submission guide",
    } == {item["name"] for item in payload["official_sources"]}

    required = {
        "PDF manuscript in JSS LaTeX article style",
        "LaTeX build log is free of blocking layout/reference errors",
        "JSS markup macros and labelled floats are used",
        "source code is packaged for installation",
        "formatted package help/documentation files are included",
        "GPL-compatible software license is clearly indicated",
        "software citation metadata is included",
        "standalone replication script covers manuscript results",
        "reviewer output transcript for standalone replication script is included",
        "short reviewer replication path completes within one hour",
        "existing implementations and comparative scope are discussed",
        "Monte Carlo content is framed as validation rather than a standalone simulation study",
        "platform dependencies and RNG seeds are disclosed",
        "active manuscript tables and figures map to generators",
        "JSS attachment size and archive source set are bounded",
        "ASCII source/data contract is enforced inside the archive",
    }
    assert required <= set(checks)
    for name in required - {
        "JSS attachment size and archive source set are bounded",
        "ASCII source/data contract is enforced inside the archive",
    }:
        assert checks[name]["ok"] is True

    md = (RESULTS / "jss_formal_compliance_audit.md").read_text()
    assert "Official JSS pages checked: 2026-05-31" in md
    assert "LaTeX build log is free of blocking layout/reference errors" in md
    assert "JSS markup macros and labelled floats are used" in md
    assert "caption_label_failures=[]" in md
    assert "pip --no-deps --target install/import probe ok=True" in md
    assert "formatted package help/documentation files are included" in md
    assert "GPL-compatible software license is clearly indicated" in md
    assert "standalone replication script covers manuscript results" in md
    assert (
        "reviewer output transcript for standalone replication script is included"
        in md
    )
    assert "short reviewer replication path completes within one hour" in md
    assert "existing implementations and comparative scope are discussed" in md
    assert (
        "Monte Carlo content is framed as validation rather than a standalone simulation study"
        in md
    )
    assert "related-software table" in md
    assert "StatsPAI advantages and disadvantages" in md
    assert "missing_related_tokens=[]" in md
    assert "active manuscript tables and figures map to generators" in md

    comparative = checks["existing implementations and comparative scope are discussed"]
    assert comparative["ok"] is True
    evidence = comparative["evidence"]
    for snippet in (
        "cross-ecosystem comparators",
        "comparator strengths",
        "non-supersession",
        "T3/T4/licensing",
        "missing_related_tokens=[]",
    ):
        assert snippet in evidence

    mc_scope = checks[
        "Monte Carlo content is framed as validation rather than a standalone simulation study"
    ]
    assert mc_scope["ok"] is True
    assert "JSS discourages extensive simulation studies" in mc_scope["evidence"]
    assert "failure-mode guards" in mc_scope["evidence"]

    style_scope = checks["JSS markup macros and labelled floats are used"]
    assert style_scope["ok"] is True
    for snippet in ("'proglang':", "'pkg':", "'code':", "caption_label_failures=[]"):
        assert snippet in style_scope["evidence"]

    source_scope = checks["source code is packaged for installation"]
    assert source_scope["ok"] is True
    for snippet in (
        "development classifier is Beta rather than Alpha",
        "pip --no-deps --target install/import probe ok=True",
        "version=1.16.0",
        "functions=1020",
        "statspai-mcp",
    ):
        assert snippet in source_scope["evidence"]

    if payload["archive_present"]:
        size_scope = checks["JSS attachment size and archive source set are bounded"]
        assert "MiB" in size_scope["evidence"]
        assert "MB decimal" in size_scope["evidence"]
