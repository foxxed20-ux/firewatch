"""Launch the isolated train-only quality experiment on an owned Colab VM."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import subprocess
import sys
import zipfile

root = Path('/content/firewatch_quality_v4')
root.mkdir(exist_ok=True)
archive = Path('/content/firewatch-quality-source.zip')
assert hashlib.sha256(archive.read_bytes()).hexdigest() == '939c092a0f3098fb46d77b2bd56413567af8b587f562a7cd6d343b389433b7a0'
assert not (root / 'setup.pid').exists(), 'Inspect previous setup before retrying'
with zipfile.ZipFile(archive) as z:
    for member in z.infolist():
        path = PurePosixPath(member.filename)
        assert not path.is_absolute() and '..' not in path.parts
        assert (member.external_attr >> 16) & 0o170000 != 0o120000
    z.extractall(root)

worker = r'''
from pathlib import Path
import hashlib, json, os, subprocess, sys, tarfile, time, urllib.parse, urllib.request
root = Path('/content/firewatch_quality_v4')
started = time.time()
try:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'numpy==2.1.3', 'scipy==1.16.3', 'rasterio==1.5.1', 'lightgbm==4.6.0'], check=True)
    dest = root / 'data/fire-train-renamed.tar'
    dest.parent.mkdir(exist_ok=True)
    q = urllib.parse.urlencode({'public_key': 'https://disk.yandex.ru/d/-rpmevTflbXZQg', 'path': '/fire-train-renamed.tar'})
    info = json.load(urllib.request.urlopen('https://cloud-api.yandex.net/v1/disk/public/resources/download?' + q, timeout=60))
    print('DOWNLOAD_START', flush=True)
    partial = dest.with_suffix('.part')
    urllib.request.urlretrieve(info['href'], partial)
    assert partial.stat().st_size == 2305976320
    digest = hashlib.file_digest(partial.open('rb'), 'sha256').hexdigest()
    assert digest == '9cc2d532312dd91d24beb8c6abcfdddac0cff6ca095501be28d81f751a8f159a'
    partial.rename(dest)
    print('DOWNLOAD_VERIFIED', digest, time.time()-started, flush=True)
    target = root / 'data/train'
    target.mkdir(exist_ok=True)
    with tarfile.open(dest) as tar:
        tar.extractall(target, filter='data')
    print('EXTRACTED', time.time()-started, flush=True)
    subprocess.run([sys.executable, '-u', '-m', 'competition.prepare', '--data-dir', str(target/'train'), '--output', str(root/'prepared'), '--split', str(root/'artifacts/split.json')], cwd=root, check=True)
    import torch, numpy, scipy, rasterio, lightgbm
    result = {'exit_code': 0, 'elapsed_seconds': time.time()-started, 'python': sys.version, 'versions': {m.__name__: m.__version__ for m in (torch,numpy,scipy,rasterio,lightgbm)}, 'cuda': torch.cuda.is_available(), 'train_sha256': digest}
except Exception as exc:
    result = {'exit_code': 1, 'error': repr(exc), 'elapsed_seconds': time.time()-started}
    raise
finally:
    (root/'setup-result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
'''
script = root / 'setup_worker.py'
script.write_text(worker)
env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
log = (root/'setup.log').open('a', buffering=1)
quality_setup_process = subprocess.Popen([sys.executable, '-u', str(script)], stdout=log, stderr=subprocess.STDOUT, cwd=root, env=env, start_new_session=True)
(root/'setup.pid').write_text(str(quality_setup_process.pid))
print(json.dumps({'pid': quality_setup_process.pid, 'root': str(root)}))
