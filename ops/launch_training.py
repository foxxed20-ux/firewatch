from pathlib import Path
import json,os,subprocess,sys,zipfile

root=Path('/content/firewatch_contest')
with zipfile.ZipFile(root/'source.zip') as z:
    for name in z.namelist():
        if not (root/name).resolve().is_relative_to(root): raise ValueError(name)
    z.extractall(root)
assert (root/'prepared/records.json').is_file(), 'Preparation is incomplete'
check=subprocess.run([sys.executable,'-m','ops.verify_prepared'],cwd=root,capture_output=True,text=True,timeout=120)
print(check.stdout);print(check.stderr)
if check.returncode: raise RuntimeError('Prepared data verification failed')
env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'}
jobs={}
chains={
    'cnn':[
      [sys.executable,'train.py','--records','prepared/records.json','--output','runs/cnn-v1','--task','af','--epochs','80','--batch-size','8','--time-limit-min','14','--workers','1'],
      [sys.executable,'train.py','--records','prepared/records.json','--output','runs/cnn-v1','--task','bs','--epochs','120','--batch-size','2','--time-limit-min','43','--workers','1'],
    ],
    'tree':[
      [sys.executable,'-m','competition.tree','--records','prepared/records.json','--output','runs/tree-v1','--task','af','--max-pixels','800000','--val-pixels','200000','--num-threads','2','--rounds','700'],
      [sys.executable,'-m','competition.tree','--records','prepared/records.json','--output','runs/tree-v1','--task','bs','--max-pixels','1000000','--val-pixels','250000','--num-threads','2','--rounds','700'],
    ]
}
for name,commands in chains.items():
    runner=root/f'run_{name}.py'
    runner.write_text('import subprocess,json,time\nfrom pathlib import Path\ncommands='+repr(commands)+'\nfor command in commands:\n print("START",command,flush=True)\n result=subprocess.run(command)\n if result.returncode: raise SystemExit(result.returncode)\nprint("CHAIN_COMPLETE",flush=True)\nPath('+repr(str(root/(name+'.done')))+').write_text("ok")\n')
    log=open(root/(name+'.log'),'a',buffering=1)
    child=subprocess.Popen([sys.executable,'-u',str(runner)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    jobs[name]={'pid':child.pid,'commands':commands,'log':str(root/(name+'.log'))}
(root/'training_jobs.json').write_text(json.dumps(jobs,indent=2));print(json.dumps(jobs))
