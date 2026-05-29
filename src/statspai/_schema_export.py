"""Machine-readable schema export for offline agent / client consumption.

StatsPAI is agent-native: ``sp.function_schema`` / ``sp.agent_card`` /
``sp.all_schemas`` already describe every *input* an agent can send.  What
was missing is a single, versioned, **import-free** bundle an external
client (or a non-Python agent runtime) can read to discover the whole
surface without executing the package — including a JSON Schema for the
*result* payload an estimator hands back.

``export_schemas(out_dir)`` writes:

* ``tools.json``        — the MCP tool manifest (input schemas the server
                          advertises), i.e. how to *call* a tool.
* ``functions.json``    — ``sp.all_schemas()``: OpenAI function-calling
                          schemas for every registered function.
* ``agent_cards.json``  — ``sp.agent_cards()``: assumptions / failure
                          modes / alternatives / typical_n_min planning
                          metadata.
* ``result.schema.json``— JSON Schema (draft 2020-12) for the agent-facing
                          result payload (``result.to_dict(detail='agent')``),
                          covering both the causal-effect and the
                          regression-coefficient shapes.
* ``index.json``        — provenance: version, counts, file list.

The result schema is the contract Day-4 adds on top of the input schemas:
it lets a downstream agent validate / typecheck what it receives, not just
what it sends.  It is intentionally permissive on method-specific extras
(``additionalProperties: true``) but strict on the shared *agent envelope*
(``violations`` / ``warnings`` / ``next_steps`` / ``suggested_functions``),
which is the part an agent actually reasons over.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

SCHEMA_VERSION = "1"

#: JSON Schema (draft 2020-12) for ``result.to_dict(detail='agent')``.
#: Grounded in the two concrete shapes emitted by
#: :meth:`statspai.core.results.CausalResult.to_dict` (estimand/estimate)
#: and :meth:`statspai.core.results.EconometricResults.to_dict`
#: (coefficient table).  Both share the agent envelope defined under
#: ``$defs`` and reused below.
RESULT_AGENT_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://statspai.org/schemas/result.agent.schema.json",
    "title": "StatsPAI agent result payload",
    "description": (
        "Shape of result.to_dict(detail='agent'). Covers the causal-effect "
        "payload (estimand/estimate/se/ci) and the regression payload "
        "(coefficients/glance). The shared agent envelope (violations, "
        "warnings, next_steps, suggested_functions) is what an agent reasons "
        "over to decide its next move."
    ),
    "type": "object",
    "required": ["method"],
    "additionalProperties": True,
    "properties": {
        "method": {"type": "string", "description": "Estimator / method name."},
        "model_type": {"type": "string"},
        "estimand": {
            "type": "string",
            "description": "ATE / ATT / LATE / CATE / ... (causal payloads).",
        },
        "estimate": {
            "type": ["number", "null"],
            "description": "Point estimate (causal payloads).",
        },
        "se": {"type": ["number", "null"], "description": "Standard error."},
        "pvalue": {"type": ["number", "null"]},
        "alpha": {"type": ["number", "null"]},
        "ci": {
            "type": ["array", "null"],
            "description": "[low, high] confidence interval.",
            "items": {"type": ["number", "null"]},
        },
        "n_obs": {"type": ["integer", "number", "null"]},
        "dependent_var": {"type": ["string", "null"]},
        "citation_key": {"type": ["string", "null"]},
        "coefficients": {
            "type": "object",
            "description": "Regression coefficient table (regression payloads).",
            "additionalProperties": {"type": "object"},
        },
        "fit_stats": {"type": "object"},
        "glance": {"type": "object"},
        "diagnostics": {"type": "object"},
        "detail_head": {"type": "array"},
        "violations": {
            "type": "array",
            "description": "Assumption / diagnostic violations flagged.",
            "items": {"$ref": "#/$defs/violation"},
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
        "next_steps": {
            "type": "array",
            "description": "Ranked, ready-to-run follow-up actions.",
            "items": {"$ref": "#/$defs/next_step"},
        },
        "suggested_functions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Flat list of sp.* functions worth calling next.",
        },
    },
    "$defs": {
        "violation": {
            "type": "object",
            "description": "One flagged assumption/diagnostic problem.",
            "additionalProperties": True,
            "properties": {
                "kind": {"type": "string"},
                "severity": {"type": "string"},
                "test": {"type": "string"},
                "value": {"type": ["number", "string", "null"]},
                "threshold": {"type": ["number", "string", "null"]},
                "message": {"type": "string"},
                "recovery_hint": {"type": "string"},
                "alternatives": {"type": "array", "items": {"type": "string"}},
            },
        },
        "next_step": {
            "type": "object",
            "description": "One ranked follow-up action.",
            "additionalProperties": True,
            "properties": {
                "action": {"type": "string"},
                "reason": {"type": "string"},
                "priority": {"type": ["string", "integer", "number"]},
                "category": {"type": "string"},
                "suggest_function": {"type": "string"},
            },
        },
    },
}


def build_schemas() -> Dict[str, Any]:
    """Return the full schema bundle as in-memory JSON-able objects.

    Pure (no I/O) so callers can serialize, diff, or embed without
    touching disk.
    """
    import statspai as sp
    from statspai.agent.tools import tool_manifest

    return {
        "index": {
            "schema_version": SCHEMA_VERSION,
            "statspai_version": sp.__version__,
            "files": [
                "tools.json",
                "functions.json",
                "agent_cards.json",
                "result.schema.json",
            ],
        },
        "tools": tool_manifest(),
        "functions": sp.all_schemas(),
        "agent_cards": sp.agent_cards(),
        "result_schema": RESULT_AGENT_SCHEMA,
    }


_FILE_MAP = {
    "tools": "tools.json",
    "functions": "functions.json",
    "agent_cards": "agent_cards.json",
    "result_schema": "result.schema.json",
}


def _json_default(obj: Any) -> Any:
    """Coerce non-JSON-serializable leaves so the bundle always renders.

    numpy scalars / arrays round-trip to native types; anything else
    (e.g. a dataclass ``default_factory`` sentinel that leaked into a
    ParamSpec ``default``) becomes ``None`` — i.e. "no explicit default",
    which is the faithful reading for a schema ``default`` field.
    """
    if hasattr(obj, "item") and not isinstance(obj, type):
        try:
            return obj.item()
        except (AttributeError, TypeError, ValueError):
            pass
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except (AttributeError, TypeError, ValueError):
            pass
    return None


def _dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n"


def render_files(bundle: Dict[str, Any]) -> Dict[str, str]:
    """Map filename -> pretty JSON text for ``bundle`` (deterministic)."""
    out: Dict[str, str] = {}
    index = dict(bundle["index"])
    index["counts"] = {
        "tools": len(bundle["tools"]),
        "functions": len(bundle["functions"]),
        "agent_cards": len(bundle["agent_cards"]),
    }
    out["index.json"] = _dumps(index)
    for key, fname in _FILE_MAP.items():
        out[fname] = _dumps(bundle[key])
    return out


def export_schemas(out_dir: "str | Path" = "schemas") -> List[Path]:
    """Write the machine-readable schema bundle to ``out_dir``.

    Parameters
    ----------
    out_dir : str or Path, default ``"schemas"``
        Directory to write the bundle into (created if absent).

    Returns
    -------
    list of Path
        The files written, sorted by name.

    Examples
    --------
    >>> import statspai as sp
    >>> from statspai._schema_export import export_schemas
    >>> export_schemas("schemas")          # doctest: +SKIP
    """
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    files = render_files(build_schemas())
    written: List[Path] = []
    for fname, text in files.items():
        path = dest / fname
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return sorted(written)


__all__ = [
    "SCHEMA_VERSION",
    "RESULT_AGENT_SCHEMA",
    "build_schemas",
    "render_files",
    "export_schemas",
]
