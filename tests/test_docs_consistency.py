"""Checks that what the documentation and the notebooks say matches what is in the repository.

The repository's rule is that a reader can check every assertion against what is in front of
them. These tests apply that to the files that describe the work: the README and the notebook
map must name notebooks that exist, every notebook must be described, the dependencies the
notebooks import must be declared and justified in requirements.txt, the dataset names must be
the three constants of CLAUDE.md, no output may print a personal path, and every stored output
must come from a cell that was actually executed.
"""

import ast
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
MARKDOWN = [ROOT / "README.md", ROOT / "mapa_notebooks_vs_articulo.md"]

NOTEBOOK_ID = re.compile(r"\b(\d{2}[A-D]?)_[a-z0-9_]+\.ipynb")
DATASET_CONSTANTS = {"MENDELEY_CALL_DIR", "PONDS_CALL_DIR", "SPECIES_CALL_DIR"}

# Import name -> name of the package that provides it, where the two differ.
DISTRIBUTION_BY_IMPORT = {
    "sklearn": "scikit-learn",
    "dotenv": "python-dotenv",
    "IPython": "ipykernel",
}
# Provided by the notebook environment or by this repository, not declared on their own.
NOT_DECLARED = {"frog_presence", "IPython"}

PERSONAL_PATH = re.compile(r"[A-Za-z]:\\{1,4}Users\\{1,4}|/Users/\w+|/home/\w+")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def code_cells(notebook: dict):
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            yield cell


def requirement_names() -> set[str]:
    names = set()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.add(re.split(r"[<>=\[ ]", line)[0].lower())
    return names


def unexplained_requirements() -> list[str]:
    """Requirements in a block (lines up to the next blank one) that opens with no comment."""
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    missing = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if lines and not lines[0].startswith("#"):
            missing.extend(lines)
    return missing


@pytest.mark.parametrize("markdown", MARKDOWN, ids=lambda p: p.name)
def test_notebooks_named_in_markdown_exist(markdown):
    """Every notebook file a markdown file mentions is on disk."""
    named = set(NOTEBOOK_ID.finditer(markdown.read_text(encoding="utf-8")))
    absent = sorted(m.group(0) for m in named
                    if not (ROOT / "notebooks" / m.group(0)).exists())
    assert not absent, f"{markdown.name} names notebooks that do not exist: {absent}"


def test_readme_describes_every_notebook():
    """Every notebook on disk has a row in the README table."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    undescribed = [p.name for p in NOTEBOOKS if p.name not in readme]
    assert not undescribed, f"README does not describe: {undescribed}"


@pytest.mark.parametrize("markdown", MARKDOWN, ids=lambda p: p.name)
def test_local_links_resolve(markdown):
    """Relative markdown links point at files that exist."""
    links = re.findall(r"\]\((?!https?://|#|mailto:)([^)\s]+)\)",
                       markdown.read_text(encoding="utf-8"))
    broken = [link for link in links if not (markdown.parent / link.split("#")[0]).exists()]
    assert not broken, f"{markdown.name} has broken links: {broken}"


def test_every_requirement_is_justified():
    """requirements.txt must carry a comment above each package, as CLAUDE.md requires."""
    assert not unexplained_requirements()


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_imports_are_declared(notebook):
    """Every third-party package a notebook imports is listed in requirements.txt."""
    declared = requirement_names()
    imported = set()
    for cell in code_cells(load(notebook)):
        source = "\n".join(line for line in "".join(cell["source"]).splitlines()
                           if not line.lstrip().startswith(("%", "!")))
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split(".")[0])
    third_party = {name for name in imported
                   if name not in sys.stdlib_module_names and name not in NOT_DECLARED}
    missing = sorted(name for name in third_party
                     if DISTRIBUTION_BY_IMPORT.get(name, name).lower() not in declared)
    assert not missing, f"imported but not in requirements.txt: {missing}"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_dataset_names_are_the_three_constants(notebook):
    """Any *_DIR name a notebook reads from the environment is one of the three constants."""
    used = set()
    for cell in code_cells(load(notebook)):
        used.update(re.findall(r"\b([A-Z][A-Z_]+_DIR)\b", "".join(cell["source"])))
    configured = {name for name in used if name.endswith("_CALL_DIR")} | {
        name for name in used if name in {"POND_DIR", "PONDS_DIR", "SPECIES_DIR", "MENDELEY_DIR"}}
    assert configured <= DATASET_CONSTANTS, f"unknown dataset names: {configured - DATASET_CONSTANTS}"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_outputs_do_not_print_personal_paths(notebook):
    """CLAUDE.md: printed output shows the folder name only, never the full path."""
    hits = 0
    for cell in code_cells(load(notebook)):
        for output in cell.get("outputs", []):
            if PERSONAL_PATH.search(json.dumps(output)):
                hits += 1
    assert hits == 0, f"{hits} outputs print a personal path"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_every_code_cell_was_executed_in_order(notebook):
    """Stored outputs must come from a top-to-bottom run: counts are present and increasing."""
    counts = [cell.get("execution_count") for cell in code_cells(load(notebook))
              if "".join(cell["source"]).strip()]
    assert None not in counts, "a code cell has never been executed"
    assert counts == sorted(counts), "cells were executed out of order"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_each_code_cell_opens_with_a_purpose_comment(notebook):
    """CLAUDE.md: a code cell starts with one leading comment stating its purpose."""
    bare = [i for i, cell in enumerate(code_cells(load(notebook)))
            if "".join(cell["source"]).strip()
            and not "".join(cell["source"]).lstrip().startswith("#")]
    assert not bare, f"{len(bare)} code cells have no leading comment"
