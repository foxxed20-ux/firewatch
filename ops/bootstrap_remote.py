import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path('/content/firewatch_contest')
root.mkdir(exist_ok=True)
print(json.dumps({'python': sys.version, 'packages': {n: bool(importlib.util.find_spec(n)) for n in ['torch','rasterio','numpy','sklearn','lightgbm','xgboost','scipy']}}))
import torch
print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
code = r'''
import json, urllib.request, urllib.parse, pathlib, tarfile, time
root=pathlib.Path('/content/firewatch_contest/data')
root.mkdir(parents=True,exist_ok=True)
key='https://disk.yandex.ru/d/-rpmevTflbXZQg'
base='https://cloud-api.yandex.net/v1/disk/public/resources'
for name in ['fire-train-renamed.tar', 'fire-test-renamed.tar']:
    dest=root/name
    if not dest.exists():
        q=urllib.parse.urlencode({'public_key':key,'path':'/'+name})
        info=json.load(urllib.request.urlopen(base+'/download?'+q,timeout=60))
        partial=dest.with_suffix('.part')
        urllib.request.urlretrieve(info['href'],partial)
        partial.rename(dest)
    print('DOWNLOADED',name,dest.stat().st_size,flush=True)
    target=root/('train' if 'train' in name else 'test')
    target.mkdir(exist_ok=True)
    if not (target/'EXTRACTED').exists():
        with tarfile.open(dest) as tar:
            members=tar.getmembers()
            print('MEMBERS',name,len(members),[m.name for m in members[:35]],flush=True)
            tar.extractall(target,filter='data')
        (target/'EXTRACTED').write_text('ok')
    print('EXTRACTED',str(target),flush=True)
print('DOWNLOAD_COMPLETE',flush=True)
'''
script=root/'download_data.py'
script.write_text(code)
log=open(root/'download.log','a',buffering=1)
proc=subprocess.Popen([sys.executable,'-u',str(script)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'download.pid').write_text(str(proc.pid))
print('DOWNLOAD_PID',proc.pid)
