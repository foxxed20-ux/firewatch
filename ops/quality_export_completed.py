"""Package completed tree candidates only; never read files being written."""
from pathlib import Path
import hashlib
import json
import zipfile
root=Path('/content/firewatch_quality_v4')
for label in ('a','b'):
    folder=root/f'runs/tree-quality-{label}/bs'
    metadata=folder/'metadata.json'
    if not metadata.exists():
        print(label,'training')
        continue
    meta=json.loads(metadata.read_text())
    if meta.get('status')!='complete':
        print(label,meta.get('status'))
        continue
    output=root/f'tree-quality-{label}.zip'
    if not output.exists():
        files=[p for p in folder.iterdir() if p.is_file()]
        manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
            for p in files:z.write(p,p.name)
            z.writestr('export-manifest.json',json.dumps(manifest,indent=2))
    print(json.dumps({'candidate':label,'zip':str(output),'bytes':output.stat().st_size,'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'iteration':meta['best_iteration'],'metrics':meta['full_validation']}))
