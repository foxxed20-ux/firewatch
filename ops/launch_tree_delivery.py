"""Run validation, packaging and real test inference on the first trained bundle."""
import json, os, subprocess, sys, zipfile
from pathlib import Path

root = Path('/content/firewatch_contest')
with zipfile.ZipFile(root/'delivery-source.zip') as archive:
    for name in archive.namelist():
        if not (root/name).resolve().is_relative_to(root): raise ValueError(name)
    archive.extractall(root)
commands = [
    [sys.executable, '-m', 'competition.evaluate', '--records', 'prepared/records.json', '--output', 'runs/eval-tree', '--tree-root', 'runs/tree-v1', '--cache-dir', 'runs/eval-tree/cache'],
    [sys.executable, '-m', 'ops.select_bundle', '--recipes', 'runs/eval-tree/selected_recipes.json', '--output', 'bundles/tree-v1', '--af-dir', 'runs/tree-v1/af', '--bs-dir', 'runs/tree-v1/bs', '--bundle-id', 'official-tree-v1'],
    [sys.executable, 'inference.py', '--data-dir', 'data/test/test', '--output', 'deliveries/tree-v1/submission.csv', '--model-dir', 'bundles/tree-v1'],
]
runner = root/'run_tree_delivery.py'
runner.write_text('''import json, subprocess, zipfile
from pathlib import Path
commands = '''+repr(commands)+'''
Path('deliveries/tree-v1').mkdir(parents=True, exist_ok=True)
for command in commands:
    print('START', command, flush=True)
    result = subprocess.run(command)
    if result.returncode: raise SystemExit(result.returncode)
with zipfile.ZipFile('deliveries/tree-v1/model_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for path in Path('bundles/tree-v1').rglob('*'):
        if path.is_file(): z.write(path, path.relative_to('bundles/tree-v1'))
with zipfile.ZipFile('deliveries/tree-v1/evidence.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for path in [Path('runs/eval-tree/validation_report.json'),Path('runs/eval-tree/selected_recipes.json'),Path('deliveries/tree-v1/submission.csv'),Path('deliveries/tree-v1/submission.audit.json')]:
        if path.is_file(): z.write(path,path.name)
Path('tree_delivery.done').write_text('ok')
print('TREE_DELIVERY_COMPLETE', flush=True)
''')
env = {**os.environ, 'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'}
with open(root/'tree_delivery.log','a', buffering=1) as log:
    process = subprocess.Popen([sys.executable,'-u',str(runner)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'tree_delivery.pid').write_text(str(process.pid))
print(json.dumps({'pid':process.pid,'commands':commands}))
