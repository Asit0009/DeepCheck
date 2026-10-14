import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "colab_training.ipynb"
CONTINUATION = re.compile(r"\\\n\s*")             # trailing backslash + newline + indentation


def _nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_sources() -> list[str]:
    return [CONTINUATION.sub(" ", "".join(c["source"])) for c in _nb()["cells"] if c["cell_type"] == "code"]


def _commands() -> list[tuple[str, list[str]]]:
    found = []
    for src in _code_sources():
        for line in src.splitlines():
            m = re.match(r"!python -m ([\w.]+)(.*)", line.strip())
            if m:
                found.append((m.group(1), re.findall(r"(?<![\w-])(--[a-z][\w-]*)", m.group(2))))
    return found


def _flags_defined_by(module: str) -> set[str]:
    path = ROOT / (module.replace(".", "/") + ".py")
    assert path.is_file(), f"notebook runs a module that does not exist: {module}"
    flags = {"--help"}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            flags |= {a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith("--")}
    return flags


def test_notebook_has_expected_structure():
    nb = _nb()
    assert nb["nbformat"] == 4 and nb["cells"]
    assert all({"cell_type", "source", "metadata"} <= set(c) for c in nb["cells"])
    assert sum(c["cell_type"] == "code" for c in nb["cells"]) >= 8


def test_python_cells_compile():
    for i, src in enumerate(_code_sources()):
        python_only = "\n".join("pass" if ln.lstrip().startswith(("!", "%")) else ln for ln in src.splitlines())
        compile(python_only, f"code cell {i}", "exec")


def test_notebook_runs_the_documented_pipeline():
    modules = {m for m, _ in _commands()}
    for needed in ("src.data.acquire", "src.data.prepare", "src.data.splits", "src.training.linear_head",
                   "src.training.train", "src.evaluation.run_eval", "src.evaluation.calibrate", "src.evaluation.report"):
        assert needed in modules, f"notebook no longer runs {needed}"


@pytest.mark.parametrize("module,flags", _commands(), ids=lambda v: v if isinstance(v, str) else " ".join(v) or "-")
def test_every_cli_flag_in_the_notebook_exists(module, flags):
    unknown = [f for f in flags if f not in _flags_defined_by(module)]
    assert not unknown, f"{module}: flags not defined by its argument parser: {unknown}"


def test_environment_variables_are_the_ones_the_code_reads():
    config_text = (ROOT / "src" / "config.py").read_text(encoding="utf-8")
    for src in _code_sources():
        for name in re.findall(r"os\.environ\['(DFD_\w+)'\]", src):
            assert f'"{name}"' in config_text, f"notebook sets {name}, which src/config.py never reads"


def test_model_id_is_defined_before_it_is_used():
    seen_definition = False
    for src in _code_sources():
        if re.search(r"^MODEL_ID\s*=", src, flags=re.M):
            seen_definition = True
        if "{MODEL_ID}" in src:
            assert seen_definition, "a notebook cell uses {MODEL_ID} before any cell defines it"


def test_notebook_does_not_claim_to_have_been_run():
    nb = _nb()
    assert "NOT been executed" in "".join(nb["cells"][0]["source"])
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert c.get("outputs") == [] and c.get("execution_count") is None
