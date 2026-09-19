"""Stage verified test inputs and validation-only D4 caches in the quality VM."""
from pathlib import Path
import os
import subprocess
import sys

root = Path('/content/firewatch_quality_v4')
assert not (root/'cache.pid').exists()
cache_worker = r'''
from pathlib import Path
import json, subprocess, sys, time
root = Path('/content/firewatch_quality_v4')
results = []
while not (root/'setup-result.json').exists():
    time.sleep(5)
assert json.loads((root/'setup-result.json').read_text())['exit_code'] == 0
for task in ('bs','af'):
    command = [sys.executable,'-B','-u','ops/cache_quality_cnn.py','--records','prepared/records.json','--checkpoint',f'initial/cnn/{task}/best.pt','--task',task,'--output',f'runs/tta-baseline/{task}.npz','--device','cuda','--tta','d4']
    print('COMMAND', command, flush=True)
    result = subprocess.run(command,cwd=root)
    results.append({'task':task,'exit_code':result.returncode})
    if result.returncode:
        break
(root/'cache-result.json').write_text(json.dumps(results,indent=2))
'''
test_worker = r'''
from pathlib import Path
import hashlib,json,tarfile,time,urllib.request,urllib.parse
root=Path('/content/firewatch_quality_v4')
q=urllib.parse.urlencode({'public_key':'https://disk.yandex.ru/d/-rpmevTflbXZQg','path':'/fire-test-renamed.tar'})
info=json.load(urllib.request.urlopen('https://cloud-api.yandex.net/v1/disk/public/resources/download?'+q,timeout=60))
path=root/'data/fire-test-renamed.tar'
partial=path.with_suffix('.part')
urllib.request.urlretrieve(info['href'],partial)
assert partial.stat().st_size==920483840
digest=hashlib.file_digest(partial.open('rb'),'sha256').hexdigest()
assert digest=='0431f8ae3af3a266a99074f87ca4a4caf5b6ea2e3645048a905ccac10c51674f'
partial.rename(path)
target=root/'data/test'
target.mkdir(exist_ok=True)
with tarfile.open(path) as tar:
    tar.extractall(target,filter='data')
(root/'test-staging.json').write_text(json.dumps({'sha256':digest,'size':path.stat().st_size,'scope':'Inputs staged only; never used for fitting or selection'},indent=2))
print('TEST_STAGED_VERIFIED',digest,flush=True)
'''
quality_extra_processes = {}
for kind, worker in [('cache',cache_worker),('test-stage',test_worker)]:
    script=root/f'{kind}_worker.py'
    script.write_text(worker)
    log=(root/f'{kind}.log').open('a',buffering=1)
    proc=subprocess.Popen([sys.executable,'-u',str(script)],cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
    quality_extra_processes[kind]=proc
    (root/f'{kind}.pid').write_text(str(proc.pid))
    print(kind,proc.pid)
