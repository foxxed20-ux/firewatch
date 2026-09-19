"""Compare delivered training logic with the frozen source archive used on Colab."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    archive_path, root = Path(args.archive), Path(args.root)
    modules = {}
    with zipfile.ZipFile(archive_path) as archive:
        for name in (
            "competition/data.py",
            "competition/models.py",
            "competition/prepare.py",
            "competition/train.py",
            "competition/tree.py",
            "competition/metrics.py",
            "train.py",
        ):
            before, after = archive.read(name), (root / name).read_bytes()
            modules[name] = {
                "training_source_sha256": hashlib.sha256(before).hexdigest(),
                "delivery_source_sha256": hashlib.sha256(after).hexdigest(),
                "ast_equal": ast.dump(ast.parse(before.decode("utf-8-sig")))
                == ast.dump(ast.parse(after.decode("utf-8-sig"))),
            }
    result = {
        "scope": "Training, features and metrics AST identity; not inference-source identity or retraining",
        "source_archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "modules": modules,
        "all_ast_equal": all(item["ast_equal"] for item in modules.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["all_ast_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
