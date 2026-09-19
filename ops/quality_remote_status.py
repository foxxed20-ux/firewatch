from pathlib import Path
import json
import os
import shutil
import subprocess
import time
root = Path('/content/firewatch_quality_v4')
print('TIME', time.time(), 'FREE_GB', round(shutil.disk_usage('/content').free/1e9, 2))
for name in ('setup', 'cnn', 'tree', 'cache', 'test-stage'):
    result = root / (name + '-result.json')
    pidfile = root / (name + '.pid')
    if pidfile.exists():
        pid = int(pidfile.read_text())
        try:
            os.kill(pid, 0)
            state = Path(f'/proc/{pid}/stat').read_text().split()[2]
        except (ProcessLookupError, FileNotFoundError):
            state = 'exited'
        print(name, 'PID', pid, state)
    if result.exists():
        parsed = json.loads(result.read_text())
        print(name, parsed if name != 'setup' else {k: parsed.get(k) for k in ('exit_code','elapsed_seconds','cuda')})
    log = root / (name + '.log')
    if log.exists() and not (name == 'setup' and result.exists()):
        print(name, log.read_text(errors='replace')[-2200:])
part = root/'data/fire-train-renamed.part'
if part.exists():
    print('DOWNLOAD_BYTES', part.stat().st_size)
if shutil.which('nvidia-smi'):
    print(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip())
