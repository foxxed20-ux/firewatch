"""Run on the Colab host to verify actual rasters and prepared coverage."""
from pathlib import Path
import hashlib,json
import numpy as np
from competition.data import discover_chips,read_training_mask

root=Path('/content/firewatch_contest')
inputs=discover_chips(root/'data/train/train')
split=json.loads((root/'artifacts/split.json').read_text())
assert set(inputs)==set(split)
counts={'af':np.zeros(256,dtype=np.int64),'bs':np.zeros(256,dtype=np.int64)}
for cid,paths in inputs.items():
    y=read_training_mask(cid,paths)
    counts[cid[:2].lower()]+=np.bincount(y.ravel(),minlength=256)
records=json.loads((root/'prepared/records.json').read_text())
assert len(records)==644 and {r['id'] for r in records}==set(inputs)
for r in records:
    assert r['split']==split[r['id']]
    assert (root/'prepared'/r['x_path']).is_file() and (root/'prepared'/r['y_path']).is_file()
report={'all_644_input_target_grids_verified':True,'coverage_exact':True,'counts':{k:{str(i):int(v) for i,v in enumerate(a) if v} for k,a in counts.items()},'split_sha256':hashlib.sha256((root/'artifacts/split.json').read_bytes()).hexdigest()}
(root/'prepared/verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
