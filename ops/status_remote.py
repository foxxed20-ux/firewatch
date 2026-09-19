from pathlib import Path
import json, subprocess, time

r = Path("/content/firewatch_contest")
result = {
    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "jobs": {},
    "models": {},
}
for name, job in json.loads((r / "training_jobs.json").read_text()).items():
    p = Path("/proc") / str(job["pid"]) / "stat"
    result["jobs"][name] = {
        "pid": job["pid"],
        "state": p.read_text().split()[2] if p.exists() else "gone",
        "done": (r / (name + ".done")).exists(),
        "log_tail": (r / (name + ".log")).read_text()[-650:],
    }
for name in ("xgb", "tree_delivery", "eval_af"):
    pid_file = r / (name + ".pid")
    if pid_file.exists():
        pid = int(pid_file.read_text())
        p = Path("/proc") / str(pid) / "stat"
        result["jobs"][name] = {
            "pid": pid,
            "state": p.read_text().split()[2] if p.exists() else "gone",
            "done": (r / (name + ".done")).exists(),
            "log_tail": (r / (name + ".log")).read_text()[-450:],
        }
for path in (r / "runs").rglob("history.json"):
    hist = json.loads(path.read_text())
    result["models"][str(path.parent.relative_to(r))] = {
        "epochs": len(hist),
        "last": hist[-1],
        "best": max(hist, key=lambda v: v["selection_score"]),
    }
for path in (r / "runs").rglob("metadata.json"):
    data = json.loads(path.read_text())
    result["models"][str(path.parent.relative_to(r))] = {
        k: data[k]
        for k in ("best_iteration", "selection_score", "full_validation")
        if k in data
    }
result["gpu"] = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader",
    ],
    capture_output=True,
    text=True,
).stdout.strip()
print(json.dumps(result, indent=2))
