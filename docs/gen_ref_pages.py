"""Generate the API reference pages from the source tree.

Run automatically by the mkdocs ``gen-files`` plugin at build time:
walks ``src/acropole``, emits one Markdown stub per public module
containing a ``::: dotted.path`` mkdocstrings directive, and writes a
``SUMMARY.md`` consumed by ``literate-nav`` to build the API navigation.

The package ``__init__`` is skipped on purpose: every public symbol it
re-exports (``FuelEstimator``, ``AircraftFuelEstimator``) already gets a
page from its defining module (``estimator``). Emitting ``::: acropole``
too would document those classes twice.

When a module declares ``__all__``, the stub pins ``members`` to exactly
that list so the rendered reference matches the declared public surface
(internal helpers such as ``diff_bfill``/``safe_divide`` stay out).
"""

from __future__ import annotations

import ast
from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()
src = Path(__file__).parent.parent / "src"


def _module_exports(module_file: Path) -> list[str] | None:
    """Return the ``__all__`` entries of a module, or None if undeclared."""
    tree = ast.parse(module_file.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = (t.id for t in node.targets if isinstance(t, ast.Name))
        if "__all__" in targets and isinstance(node.value, ast.List):
            return [
                el.value
                for el in node.value.elts
                if isinstance(el, ast.Constant) and isinstance(el.value, str)
            ]
    return None


for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", "api", doc_path)

    parts = tuple(module_path.parts)
    # Skip the package __init__ (re-export façade) and all private modules:
    # their symbols are documented from their real defining module.
    if parts[-1] == "__init__" or parts[-1].startswith("_"):
        continue

    if not parts:
        continue

    nav[parts] = doc_path.as_posix()

    identifier = ".".join(parts)
    exports = _module_exports(path)
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"::: {identifier}\n")
        if exports is not None:
            fd.write("    options:\n      members:\n")
            for name in exports:
                fd.write(f"        - {name}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(src.parent))

with mkdocs_gen_files.open("reference/api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
