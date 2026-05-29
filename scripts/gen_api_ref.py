"""mkdocs-gen-files script: auto-generate the full API reference.

Run automatically by the ``gen-files`` plugin during ``mkdocs build`` /
``mkdocs serve``. For every public sub-package under ``src/statspai`` it emits
a virtual page ``reference/api/<module>.md`` containing a single mkdocstrings
directive (``::: statspai.<module>``), plus a ``reference/api/SUMMARY.md`` that
the ``literate-nav`` plugin turns into the API-reference nav tree, and an
index page.

This replaces hand-maintained per-module API stubs: the 86 sub-packages are
documented straight from their NumPy-style docstrings, so the reference can
never drift from the code. The curated, methodology-grouped pages under
``docs/reference/*.md`` remain the recommended entry points; this section is
the exhaustive fallback an agent or power user can grep.
"""

from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files

# Repo layout: this script lives in scripts/, the package in src/statspai/.
SRC = Path(__file__).resolve().parent.parent / "src" / "statspai"

# Infrastructure sub-packages that are not part of the public estimator API
# surface; skipped to keep the API tree focused on methods users call.
SKIP = {"__pycache__"}


def _public_subpackages() -> list[str]:
    mods = []
    for entry in sorted(SRC.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("_") or entry.name in SKIP:
            continue
        if not (entry / "__init__.py").exists():
            continue
        mods.append(entry.name)
    return mods


def main() -> None:
    modules = _public_subpackages()
    nav = mkdocs_gen_files.Nav()

    for mod in modules:
        doc_path = Path("reference", "api", f"{mod}.md")
        with mkdocs_gen_files.open(doc_path, "w") as fd:
            fd.write(f"# `statspai.{mod}`\n\n")
            fd.write(f"::: statspai.{mod}\n")
        mkdocs_gen_files.set_edit_path(
            doc_path, Path("src", "statspai", mod, "__init__.py")
        )
        nav[(mod,)] = f"{mod}.md"

    # Index / landing page for the auto API section.
    with mkdocs_gen_files.open(Path("reference", "api", "index.md"), "w") as fd:
        fd.write("# Full API reference (auto-generated)\n\n")
        fd.write(
            "Every public sub-package of `statspai`, documented directly from "
            "its NumPy-style docstrings. This section is exhaustive and always "
            "in sync with the code; for guided, methodology-grouped entry "
            "points see the curated **Reference** pages.\n\n"
        )
        fd.write(f"**{len(modules)} sub-packages** are documented here:\n\n")
        for mod in modules:
            fd.write(f"- [`statspai.{mod}`]({mod}.md)\n")

    with mkdocs_gen_files.open(Path("reference", "api", "SUMMARY.md"), "w") as fd:
        fd.write("* [Overview](index.md)\n")
        fd.writelines(nav.build_literate_nav())


main()
