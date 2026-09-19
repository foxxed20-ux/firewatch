import ast
from pathlib import Path


def test_tree_module_imports_lightgbm_lazily():
    source = Path("competition/tree.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert not any(any(alias.name == "lightgbm" for alias in getattr(node, "names", [])) for node in imports)


def test_tree_exposes_portable_prediction_contract():
    source = Path("competition/tree.py").read_text(encoding="utf-8")
    assert "def predict_tree_features" in source
    assert '"model.txt"' in source
    assert '"metadata.json"' in source
