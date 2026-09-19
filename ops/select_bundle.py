"""Package selected validation recipes with hashes and explicit postprocessing."""

from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path

BACKENDS = {"cnn": "torch", "tree": "lightgbm"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def copy_component(source, target, backend):
    source, target = Path(source), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    names = ("best.pt",) if backend == "torch" else ("model.txt", "metadata.json")
    for name in names:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
        shutil.copy2(source / name, target / name)
    return {
        "backend": backend,
        "path": str(target),
        "sha256": {name: digest(target / name) for name in names},
    }


def postprocess(task, recipe):
    return (
        {"threshold": recipe["threshold"]}
        if task == "af"
        else {"class_multipliers": recipe["class_multipliers"]}
    )


def apply_tree_recipe(folder, task, recipe):
    path = Path(folder) / "metadata.json"
    meta = json.loads(path.read_text())
    meta.update(postprocess(task, recipe))
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def apply_torch_recipe(folder, task, recipe):
    import torch

    path = Path(folder) / "best.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint.update(postprocess(task, recipe))
    torch.save(checkpoint, path)


def finalized_component(source, target, backend, task, recipe, bundle_root):
    component = copy_component(source, target, backend)
    (apply_torch_recipe if backend == "torch" else apply_tree_recipe)(
        target, task, recipe
    )
    component["sha256"] = {
        p.name: digest(p) for p in Path(target).iterdir() if p.is_file()
    }
    component["path"] = Path(target).relative_to(bundle_root).as_posix()
    return component


def single(task, recipe, source, output):
    backend = BACKENDS.get(recipe.get("backend"))
    if backend is None:
        raise ValueError(
            f"{task}: unsupported recipe backend {recipe.get('backend')!r}"
        )
    component = finalized_component(
        source, output / task, backend, task, recipe, output
    )
    return {
        "backend": backend,
        "path": task,
        "sha256": component["sha256"],
        "postprocess": postprocess(task, recipe),
        "selection": recipe,
    }


def ensemble(task, recipe, cnn_source, tree_source, output):
    if cnn_source is None or tree_source is None:
        raise ValueError(f"{task}: ensemble requires both CNN and tree directories")
    prefix = "ensemble_cnn_"
    selected = str(recipe["selected"])
    if not selected.startswith(prefix):
        raise ValueError(f"{task}: unknown ensemble recipe {selected!r}")
    cnn_weight = float(selected[len(prefix) :])
    tree_weight = 1.0 - cnn_weight
    root = output / task
    cnn = finalized_component(cnn_source, root / "cnn", "torch", task, recipe, output)
    tree = finalized_component(
        tree_source, root / "tree", "lightgbm", task, recipe, output
    )
    return {
        "backend": "ensemble",
        "path": task,
        "components": {"cnn": cnn, "tree": tree},
        "tree_weight": tree_weight,
        "postprocess": postprocess(task, recipe),
        "selection": recipe,
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--recipes", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--af-dir")
    p.add_argument("--bs-dir")
    p.add_argument("--af-cnn-dir")
    p.add_argument("--af-tree-dir")
    p.add_argument("--bs-cnn-dir")
    p.add_argument("--bs-tree-dir")
    p.add_argument("--bundle-id", default="official-v1")
    a = p.parse_args(argv)
    recipes = json.loads(Path(a.recipes).read_text())
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    tasks = {}
    for task in ("af", "bs"):
        recipe = recipes[task]
        if recipe.get("backend") == "ensemble":
            tasks[task] = ensemble(
                task,
                recipe,
                getattr(a, f"{task}_cnn_dir"),
                getattr(a, f"{task}_tree_dir"),
                output,
            )
        else:
            source = getattr(a, f"{task}_dir") or getattr(
                a, f"{task}_{recipe.get('backend')}_dir", None
            )
            if source is None:
                raise ValueError(
                    f"{task}: no directory for selected {recipe.get('backend')} backend"
                )
            tasks[task] = single(task, recipe, source, output)
    manifest = {
        "bundle_id": a.bundle_id,
        "feature_version": "official-named-v1",
        "tasks": tasks,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
