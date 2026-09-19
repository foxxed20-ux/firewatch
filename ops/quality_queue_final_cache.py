"""Cache the final independently selected student and EMA checkpoints."""
from pathlib import Path
import os
import subprocess
import sys
root=Path('/content/firewatch_quality_v4')
assert not (root/'final-cache.pid').exists()
worker=r'''
from pathlib import Path
import hashlib,json,subprocess,sys,time,zipfile
root=Path('/content/firewatch_quality_v4')
while not (root/'cnn-result.json').exists():time.sleep(5)
assert json.loads((root/'cnn-result.json').read_text())['exit_code']==0
results=[]
for label in ('student','ema'):
    command=[sys.executable,'-B','-u','ops/cache_quality_cnn.py','--records','prepared/records.json','--checkpoint',f'runs/cnn-quality-v1/bs/best_{label}.pt','--task','bs','--output',f'runs/cnn-quality-v1/bs/{label}.npz','--device','cuda','--tta','none']
    result=subprocess.run(command,cwd=root)
    results.append({'label':label,'exit_code':result.returncode})
    if result.returncode:break
folder=root/'runs/cnn-quality-v1/bs'
files=[p for p in folder.iterdir() if p.is_file() and p.suffix!='.npz']
with zipfile.ZipFile(root/'cnn-final-weights.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,p.name)
    z.writestr('export-manifest.json',json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
(root/'final-cache-result.json').write_text(json.dumps(results,indent=2))
print('FINAL_CACHE_DONE',results,flush=True)
'''
script=root/'final_cache_worker.py'
script.write_text(worker)
log=(root/'final-cache.log').open('a',buffering=1)
quality_final_cache_process=subprocess.Popen([sys.executable,'-u',str(script)],cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
(root/'final-cache.pid').write_text(str(quality_final_cache_process.pid))
print('FINAL_CACHE_QUEUED',quality_final_cache_process.pid)
