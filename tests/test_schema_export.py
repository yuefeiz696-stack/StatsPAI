"""Contract for the machine-readable schema bundle (Day-4 agent-native).

Locks three things:

1. The agent-facing *result* payload that ``execute_tool`` / the MCP
   server return — ``result.to_dict(detail='agent')`` — actually
   validates against the published ``RESULT_AGENT_SCHEMA``, for BOTH the
   causal-effect shape (``sp.did`` -> CausalResult) and the regression
   shape (``sp.regress`` -> EconometricResults).  If a refactor changes
   the envelope an agent reasons over, this fails.
2. Every advertised tool entry carries the ``{name, description,
   input_schema}`` triple an agent needs to call it.
3. The committed ``schemas/`` bundle is in sync with the live surface
   (delegates to ``scripts/dump_schemas.py --check``), so the offline
   artifact never silently drifts from the package.

No network / R / Stata.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statspai._schema_export import (
    RESULT_AGENT_SCHEMA,
    build_schemas,
    render_files,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMAS_DIR = REPO_ROOT / "schemas"

jsonschema = pytest.importorskip("jsonschema")


# --------------------------------------------------------------------------- #
#  Real fitted-result payloads validate against the published result schema
# --------------------------------------------------------------------------- #


def _did_dataset() -> pd.DataFrame:
    rng = np.random.RandomState(0)
    n = 400
    df = pd.DataFrame(
        {
            "id": np.repeat(np.arange(n // 2), 2),
            "time": np.tile([0, 1], n // 2),
        }
    )
    df["treat"] = (df["id"] % 2 == 0).astype(int)
    df["post"] = df["time"]
    df["y"] = 1.0 + 0.5 * df["treat"] * df["post"] + rng.normal(0, 1, len(df))
    return df


def _ols_dataset() -> pd.DataFrame:
    rng = np.random.RandomState(1)
    n = 300
    x = rng.normal(0, 1, n)
    z = rng.normal(0, 1, n)
    y = 0.3 + 1.2 * x - 0.5 * z + rng.normal(0, 1, n)
    return pd.DataFrame({"y": y, "x": x, "z": z})


def test_causal_result_agent_payload_matches_schema():
    import statspai as sp

    r = sp.did(_did_dataset(), y="y", treat="treat", time="post")
    payload = r.to_dict(detail="agent")
    # Round-trips through JSON (agents receive it as JSON) ...
    payload = json.loads(json.dumps(payload))
    # ... and conforms to the published contract.
    jsonschema.validate(payload, RESULT_AGENT_SCHEMA)
    # Envelope an agent reasons over is well-formed when present.
    assert isinstance(payload.get("warnings", []), list)
    assert isinstance(payload.get("suggested_functions", []), list)


def test_econometric_result_agent_payload_matches_schema():
    import statspai as sp

    r = sp.regress("y ~ x + z", data=_ols_dataset())
    payload = json.loads(json.dumps(r.to_dict(detail="agent")))
    jsonschema.validate(payload, RESULT_AGENT_SCHEMA)
    # Regression payloads carry a coefficient table.
    assert "coefficients" in payload


def test_minimal_and_standard_payloads_also_validate():
    """The schema is permissive enough to accept the leaner detail levels."""
    import statspai as sp

    r = sp.did(_did_dataset(), y="y", treat="treat", time="post")
    for detail in ("minimal", "standard", "agent"):
        payload = json.loads(json.dumps(r.to_dict(detail=detail)))
        jsonschema.validate(payload, RESULT_AGENT_SCHEMA)


def test_result_schema_is_itself_a_valid_jsonschema():
    # Will raise SchemaError if the schema document is malformed.
    jsonschema.Draft202012Validator.check_schema(RESULT_AGENT_SCHEMA)


# --------------------------------------------------------------------------- #
#  Tool manifest entries are complete
# --------------------------------------------------------------------------- #


def test_every_tool_entry_has_the_call_triple():
    bundle = build_schemas()
    bad = []
    for t in bundle["tools"]:
        if not t.get("name"):
            bad.append(repr(t)[:60])
            continue
        if not isinstance(t.get("description"), str):
            bad.append(f"{t['name']}: missing description")
        if not isinstance(t.get("input_schema"), dict):
            bad.append(f"{t['name']}: missing input_schema")
    assert not bad, (
        "Tool entries missing the {name, description, input_schema} triple:\n  "
        + "\n  ".join(bad)
    )


def test_bundle_renders_to_valid_json():
    files = render_files(build_schemas())
    assert {
        "index.json",
        "tools.json",
        "functions.json",
        "agent_cards.json",
        "result.schema.json",
    } <= set(files)
    for fname, text in files.items():
        json.loads(text)  # raises on malformed JSON


# --------------------------------------------------------------------------- #
#  Committed bundle stays in sync with the live surface
# --------------------------------------------------------------------------- #


def test_committed_schemas_dir_is_in_sync():
    """`scripts/dump_schemas.py --check` must pass against committed schemas/.

    If this fails, run `python scripts/dump_schemas.py` and commit the
    refreshed bundle — the offline artifact has drifted from the package.
    """
    if not SCHEMAS_DIR.exists():
        pytest.skip("schemas/ not generated in this checkout")
    res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "dump_schemas.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert (
        res.returncode == 0
    ), f"schemas/ is stale:\nstdout={res.stdout!r}\nstderr={res.stderr!r}"
