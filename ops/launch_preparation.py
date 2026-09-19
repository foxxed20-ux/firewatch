from pathlib import Path
import json
import subprocess
import sys
import zipfile

root=Path('/content/firewatch_contest')
with zipfile.ZipFile(root/'source.zip') as z:
    for name in z.namelist():
        if not (root/name).resolve().is_relative_to(root): raise ValueError(name)
    z.extractall(root)
check=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_competition_contract.py','tests/test_training_shapes.py','tests/test_tree_contract.py'],cwd=root,timeout=120,capture_output=True,text=True)
print(check.stdout);print(check.stderr)
if check.returncode: raise RuntimeError('Pre-training tests failed')
log=open(root/'prepare.log','a',buffering=1)
args=[sys.executable,'-u','-m','competition.prepare','--data-dir',str(root/'data/train/train'),'--output',str(root/'prepared'),'--split',str(root/'artifacts/split.json')]
p=subprocess.Popen(args,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'prepare.pid').write_text(str(p.pid)); print('PREPARE_PID',p.pid)
