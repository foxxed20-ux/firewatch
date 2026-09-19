"""Queue bounded training after the independently verified preparation finishes."""
from pathlib import Path
import json
import os
import subprocess
import sys
import torch

root = Path('/content/firewatch_quality_v4')
kind = 'cnn' if torch.cuda.is_available() else 'tree'
assert not (root / f'{kind}.pid').exists(), 'Inspect existing job before retrying'
worker = r'''
from pathlib import Path
import datetime, json, os, subprocess, sys, time
root = Path('/content/firewatch_quality_v4')
kind = sys.argv[1]
deadline = datetime.datetime(2026,9,19,16,5,tzinfo=datetime.timezone.utc).timestamp()
results = []
started = time.time()
try:
    while not (root/'setup-result.json').exists():
        if time.time() > deadline - 600:
            raise RuntimeError('insufficient time after setup')
        time.sleep(5)
    assert json.loads((root/'setup-result.json').read_text())['exit_code'] == 0
    common = [sys.executable, '-B', '-u']
    if kind == 'cnn':
        budget = min(30, (deadline - time.time())/60 - 3)
        commands = [common + ['ops/train_quality_cnn.py', '--records','prepared/records.json','--source','initial/cnn/bs','--output','runs/cnn-quality-v1/bs','--device','cuda','--crop-size','384','--batch-size','1','--epochs','25','--time-limit-min',str(budget),'--lr','8e-5','--workers','2']]
    else:
        commands = []
        for label, seed, exponent in [('a',42,1),('b',137,.5)]:
            commands.append(common + ['-m','ops.train_quality_tree','--records','prepared/records.json','--output',f'runs/tree-quality-{label}/bs','--task','bs','--max-pixels','2000000','--val-pixels','400000','--num-threads','2','--seed',str(seed),'--rounds','1200','--early-stopping','100','--learning-rate','.035','--num-leaves','127','--min-data-in-leaf','80','--feature-fraction','.9','--bagging-fraction','.9','--lambda-l2','2','--prior-exponent',str(exponent),'--time-limit-seconds','960'])
    for command in commands:
        if time.time() >= deadline - 180:
            break
        if kind == 'tree':
            command[-1] = str(min(960, max(30,deadline-time.time()-180)))
        print('COMMAND', json.dumps(command), 'AT', datetime.datetime.now(datetime.timezone.utc).isoformat(), flush=True)
        before = time.time()
        completed = subprocess.run(command, cwd=root)
        results.append({'command':command,'exit_code':completed.returncode,'seconds':time.time()-before})
        (root/f'{kind}-progress.json').write_text(json.dumps(results,indent=2))
        if completed.returncode:
            raise RuntimeError(f'training exited {completed.returncode}')
    result = {'exit_code':0,'commands':results,'seconds':time.time()-started}
except Exception as exc:
    result = {'exit_code':1,'error':repr(exc),'commands':results,'seconds':time.time()-started}
    raise
finally:
    (root/f'{kind}-result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result), flush=True)
'''
script = root / f'{kind}_worker.py'
script.write_text(worker)
log = (root/f'{kind}.log').open('a', buffering=1)
env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
quality_training_process = subprocess.Popen([sys.executable,'-u',str(script),kind],cwd=root,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
(root/f'{kind}.pid').write_text(str(quality_training_process.pid))
print(json.dumps({'kind':kind,'pid':quality_training_process.pid}))
