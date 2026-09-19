from pathlib import Path
import json
import numpy as np
import rasterio

root=Path('/content/firewatch_contest/data/train')
print('EXTRACTED', (root/'EXTRACTED').exists())
files=list(root.rglob('*'))
print('NON_RASTERS',[(str(p.relative_to(root)),p.stat().st_size) for p in files if p.is_file() and p.suffix.lower() not in ['.tif','.tiff']][:80])
groups={}
for p in files:
    if p.suffix.lower() in ['.tif','.tiff']:
        groups.setdefault(str(p.parent.relative_to(root)),[]).append(p)
result={}
for group,ps in groups.items():
    p=ps[0]
    with rasterio.open(p) as ds:
        a=ds.read()
        result[group]={'count':len(ps),'file':p.name,'shape':list(a.shape),'dtype':str(a.dtype),'nodata':ds.nodata,'crs':str(ds.crs),'transform':list(ds.transform),'descriptions':ds.descriptions,'tags':ds.tags(),'min':np.nanmin(a,axis=(1,2)).tolist(),'max':np.nanmax(a,axis=(1,2)).tolist()}
        if 'mask' in group or 'label' in group: result[group]['unique']=np.unique(a).tolist()
print(json.dumps(result,indent=2,ensure_ascii=False))
Path('/content/firewatch_contest/train_schema.json').write_text(json.dumps(result,indent=2))
for p in files:
    if p.suffix.lower() in ['.json','.csv','.txt','.md','.py'] and p.stat().st_size < 50000:
        print('FILE',str(p.relative_to(root)),p.read_text(errors='replace')[:18000])
