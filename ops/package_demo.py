from pathlib import Path
import csv,json,zipfile
root=Path('/content/firewatch_contest')
data=root/'data/train/train'
split=json.loads((root/'artifacts/split.json').read_text())
selected=[]
for task in ('af','bs'):
    rows=list(csv.DictReader((data/task/'meta.csv').open()))
    valid=[r for r in rows if split[r['chip_id']]=='val']
    if task=='af':
        valid=sorted(valid,key=lambda r:float(r['n_fire_px']),reverse=True)[:2]
    else:
        valid=sorted(valid,key=lambda r:min(float(r['sev1_px']),float(r['sev2_px']),float(r['sev3_px'])),reverse=True)[:2]
    selected.extend(valid)
out=root/'model_demo_inputs.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for row in selected:
        chip_id=row['chip_id']; task=row['kind']
        for p in (data/task).rglob(chip_id+'_*.tif'):
            if p.parent.name!='masks': z.write(p,str(p.relative_to(data)))
    z.writestr('demo_metadata.json',json.dumps({'source':'Official competition TRAIN archive','usage':'Geospatial demonstration on held-out training chips; not private test','selection':'Two AF chips with most fire pixels and two BS chips with all severity classes in validation; selected for demonstration only','scenes':selected},indent=2))
print('DEMO',out.stat().st_size,[r['chip_id'] for r in selected])
