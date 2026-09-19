"""Optional isolated CPU training aid; never changes the deployed service.

Consumes the model owner's frozen source archive and only official public TRAIN.
The model owner must evaluate/select its output before serving it.
"""

from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request
import zipfile

root = Path("/home/red/firewatch-training").resolve()
root.mkdir(exist_ok=True)
source = root / "source-training-v1.zip"
print("START", os.getpid(), time.time(), flush=True)
with zipfile.ZipFile(source) as archive:
    for member in archive.infolist():
        target = (root / member.filename).resolve()
        if (
            not target.is_relative_to(root)
            or (member.external_attr >> 16) & 0o170000 == 0o120000
        ):
            raise ValueError("Unsafe source archive member")
    archive.extractall(root)

data = root / "data"
data.mkdir(exist_ok=True)
query = urllib.parse.urlencode(
    {
        "public_key": "https://disk.yandex.ru/d/-rpmevTflbXZQg",
        "path": "/fire-train-renamed.tar",
    }
)
api = "https://cloud-api.yandex.net/v1/disk/public/resources"
meta = json.load(urllib.request.urlopen(api + "?" + query, timeout=60))
archive_path = data / "fire-train-renamed.tar"
if not archive_path.exists():
    href = json.load(urllib.request.urlopen(api + "/download?" + query, timeout=60))[
        "href"
    ]
    partial = data / "fire-train-renamed.tar.part"
    with (
        urllib.request.urlopen(href, timeout=60) as incoming,
        partial.open("wb") as outgoing,
    ):
        count = 0
        last = time.monotonic()
        while chunk := incoming.read(4 * 1024 * 1024):
            outgoing.write(chunk)
            count += len(chunk)
            if time.monotonic() - last > 15:
                print("DOWNLOAD_BYTES", count, flush=True)
                last = time.monotonic()
    partial.rename(archive_path)
md5 = hashlib.md5()
sha = hashlib.sha256()
with archive_path.open("rb") as stream:
    while chunk := stream.read(4 * 1024 * 1024):
        md5.update(chunk)
        sha.update(chunk)
if archive_path.stat().st_size != meta["size"] or meta.get("md5") != md5.hexdigest():
    raise ValueError("Official train archive metadata/hash mismatch")
if meta.get("sha256") and meta["sha256"] != sha.hexdigest():
    raise ValueError("Official train archive SHA256 mismatch")
(root / "download-evidence.json").write_text(
    json.dumps(
        {
            "source": "official public TRAIN only",
            "bytes": archive_path.stat().st_size,
            "md5": md5.hexdigest(),
            "sha256": sha.hexdigest(),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
        indent=2,
    )
)
print("DOWNLOAD_VERIFIED", archive_path.stat().st_size, sha.hexdigest(), flush=True)
train = data / "train"
train.mkdir(exist_ok=True)
if not (train / "EXTRACTED").exists():
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            if not (train / member.name).resolve().is_relative_to(train) or not (
                member.isfile() or member.isdir()
            ):
                raise ValueError("Unsafe training archive member")
        archive.extractall(train)
    (train / "EXTRACTED").write_text("verified official train")
print("PREPARE_START", flush=True)
env = {
    **os.environ,
    "OMP_NUM_THREADS": "8",
    "OPENBLAS_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "PYTHONUNBUFFERED": "1",
}
subprocess.run(
    [
        sys.executable,
        "-u",
        "-m",
        "competition.prepare",
        "--data-dir",
        str(train / "train"),
        "--output",
        "prepared",
        "--split",
        "artifacts/split.json",
    ],
    cwd=root,
    env=env,
    check=True,
)
print("TREE_START", flush=True)
subprocess.run(
    [
        sys.executable,
        "-u",
        "-m",
        "competition.tree",
        "--records",
        "prepared/records.json",
        "--output",
        "runs/tree-fast",
        "--task",
        "bs",
        "--max-pixels",
        "1000000",
        "--val-pixels",
        "250000",
        "--num-threads",
        "8",
        "--rounds",
        "700",
    ],
    cwd=root,
    env=env,
    check=True,
)
print("TRAINING_COMPLETE", flush=True)
