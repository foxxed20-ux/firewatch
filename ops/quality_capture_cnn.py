"""Capture a consistent, loadable intermediate checkpoint without stopping training."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import torch

root=Path('/content/firewatch_quality_v4')
source=root/'runs/cnn-quality-v1/bs'
target=root/'runs/cnn-early/bs'
assert not target.exists(), 'Inspect existing snapshot instead of replacing it'
target.mkdir(parents=True)
shutil.copy2(source/'config.json',target/'config.json')
manifest={}
for name in ('best_student.pt','best_ema.pt'):
    for attempt in range(3):
        data=(source/name).read_bytes()
        digest=hashlib.sha256(data).hexdigest()
        (target/name).write_bytes(data)
        state=torch.load(target/name,map_location='cpu',weights_only=True)
        if hashlib.sha256((source/name).read_bytes()).hexdigest()==digest:
            manifest[name]={'sha256':digest,'epoch':state['epoch'],'selection_score':state['selection_score']}
            break
        time.sleep(.2)
    else:
        raise RuntimeError('checkpoint changed during capture')
(target/'snapshot.json').write_text(json.dumps(manifest,indent=2))
worker=r'''
from pathlib import Path
import json,subprocess,sys
root=Path('/content/firewatch_quality_v4')
results=[]
for label in ('student','ema'):
    cmd=[sys.executable,'-B','-u','ops/cache_quality_cnn.py','--records','prepared/records.json','--checkpoint',f'runs/cnn-early/bs/best_{label}.pt','--task','bs','--output',f'runs/cnn-early/bs/{label}.npz','--device','cuda','--tta','none']
    result=subprocess.run(cmd,cwd=root)
    results.append({'label':label,'exit_code':result.returncode})
    if result.returncode:break
(root/'early-cache-result.json').write_text(json.dumps(results,indent=2))
'''
script=root/'early_cache_worker.py'
script.write_text(worker)
log=(root/'early-cache.log').open('a',buffering=1)
quality_early_cache_process=subprocess.Popen([sys.executable,'-u',str(script)],cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
(root/'early-cache.pid').write_text(str(quality_early_cache_process.pid))
print(json.dumps({'pid':quality_early_cache_process.pid,'snapshot':manifest}))
