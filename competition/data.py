"""Named-band readers and deterministic features for the official chip format.

Inference never reads masks, meta.csv, event identifiers, dates or coordinates.
Cloud and invalid-observation indicators are features, not evaluation exclusions.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

FEATURE_VERSION = 'official-named-v1'
S2 = ('B2','B3','B4','B5','B6','B7','B8A','B11','B12','SCL')
VIIRS = ('I1','I2','I3','I4','I5','solar_zenith','sensor_zenith','valid')
AF_AUX = ('landcover','dem','t2m','rh2m','wind_speed')
BS_AUX = ('dem','slope','landcover')

@dataclass
class Chip:
    chip_id: str
    task: str
    x: np.ndarray
    feature_names: list[str]
    observation_valid: np.ndarray
    files: dict[str,str]

def _read(path: Path, expected: tuple[str,...], size: int, dtype: str):
    with rasterio.open(path) as ds:
        if ds.count != len(expected) or (ds.height,ds.width)!=(size,size):
            raise ValueError(f'Unexpected shape for {path.name}: {(ds.count,ds.height,ds.width)}')
        names=tuple(ds.descriptions)
        if names != expected:
            raise ValueError(f'Band descriptions for {path.name}: {names}, expected {expected}')
        if any(np.dtype(item)!=np.dtype(dtype) for item in ds.dtypes):
            raise ValueError(f'Unexpected dtype for {path.name}: {ds.dtypes}; expected {dtype}')
        a=ds.read().astype(np.float32)
        grid=(ds.width,ds.height,ds.crs,ds.transform)
    return a, grid

def discover_chips(data_root: str|Path) -> dict[str,dict[str,Path]]:
    root=Path(data_root)
    if not root.is_dir(): raise FileNotFoundError(root)
    result={}
    # Match input suffixes only. In particular, do not inspect target rasters.
    suffixes={'VIIRS_I1-I5':'viirs','AUX':'aux','Sentinel-2_pre':'s2_pre','Sentinel-2_post':'s2_post','Sentinel-1_pre':'s1_pre','Sentinel-1_post':'s1_post'}
    for p in root.rglob('*.tif'):
        for suffix,key in suffixes.items():
            ending='_'+suffix+'.tif'
            if p.name.endswith(ending):
                chip_id=p.name[:-len(ending)]
                if not re.fullmatch(r'(AF|BS)_[A-Za-z0-9_-]+',chip_id): continue
                paths=result.setdefault(chip_id,{})
                if key in paths: raise ValueError(f'Duplicate input {chip_id}/{key}; pass a single train or test root')
                paths[key]=p
                break
    return dict(sorted(result.items()))

def _ratio(a,b): return np.clip((a-b)/(np.abs(a)+np.abs(b)+1e-5),-1,1)

def read_chip(chip_id: str, paths: dict[str,Path]) -> Chip:
    task=chip_id[:2].lower()
    features=[]; names=[]; grids=[]
    def add(name,value):
        names.append(name); features.append(np.nan_to_num(value,nan=0.0,posinf=0.0,neginf=0.0).astype(np.float32))
    def read(key,expected,size):
        if key not in paths: raise FileNotFoundError(f'Missing {chip_id}/{key}')
        dtype='float32' if task=='af' else ('uint16' if key.startswith('s2_') else 'int16')
        a,g=_read(paths[key],expected,size,dtype); grids.append(g); return a
    if task=='af':
        v=read('viirs',VIIRS,256); aux=read('aux',AF_AUX,256)
        if not np.isin(v[7][np.isfinite(v[7])],[0,1]).all(): raise ValueError('VIIRS valid band must be binary')
        valid=np.isfinite(v[:5]).all(axis=0)&(v[7]>.5)
        for i,name in enumerate(VIIRS):
            value=v[i]
            if name in ('I4','I5'): value=(value-273.15)/50
            elif 'zenith' in name: value=value/90
            add(name,value)
        for i,name in enumerate(AF_AUX):
            value=aux[i]
            if name=='landcover': value=value/100
            elif name=='dem': value=value/1000
            elif name=='t2m': value=(value-273.15)/50
            elif name=='rh2m': value=value/100
            elif name=='wind_speed': value=value/20
            add(name,value)
        thermal=np.nan_to_num(v[3]-v[4])/20
        add('I4_minus_I5',thermal)
        add('I4_minus_t2m',(v[3]-aux[2])/20)
        add('I5_minus_t2m',(v[4]-aux[2])/20)
        add('reflectance_nd21',_ratio(v[1],v[0]))
        add('reflectance_nd32',_ratio(v[2],v[1]))
        for size in (3,11):
            mean=uniform_filter(thermal,size=size,mode='reflect')
            add(f'thermal_mean_{size}',mean)
            add(f'thermal_contrast_{size}',thermal-mean)
        i4=np.nan_to_num((v[3]-273.15)/50)
        add('I4_contrast_11',i4-uniform_filter(i4,size=11,mode='reflect'))
    elif task=='bs':
        pre=read('s2_pre',S2,512); post=read('s2_post',S2,512)
        if not np.isin(pre[9],range(12)).all() or not np.isin(post[9],range(12)).all(): raise ValueError('Invalid SCL classes')
        sarpre=read('s1_pre',('VV','VH'),512)/100
        sarpost=read('s1_post',('VV','VH'),512)/100
        aux=read('aux',BS_AUX,512)
        valid=np.isfinite(pre).all(axis=0)&np.isfinite(post).all(axis=0)&~np.isin(pre[9],[0,1,3,8,9,10,11])&~np.isin(post[9],[0,1,3,8,9,10,11])
        pre[:9]/=10000; post[:9]/=10000
        for date,a in [('pre',pre),('post',post)]:
            for i,name in enumerate(S2): add(f'{date}_{name}',a[i]/11 if name=='SCL' else a[i])
        for date,a in [('pre',sarpre),('post',sarpost)]:
            for i,name in enumerate(('VV','VH')): add(f'{date}_{name}_db',a[i]/30)
        add('dem',aux[0]/1000); add('slope',aux[1]/90); add('landcover',aux[2]/100)
        for i,name in enumerate(S2[:9]): add(f'delta_{name}',pre[i]-post[i])
        for i,name in enumerate(('VV','VH')): add(f'delta_{name}_db',(sarpre[i]-sarpost[i])/20)
        delta={}
        for name,i,j in [('NBR',6,8),('NDVI',6,2),('NBR2',7,8),('NDWI',1,6)]:
            a=_ratio(pre[i],pre[j]); b=_ratio(post[i],post[j]); delta[name]=a-b
            add(f'pre_{name}',a); add(f'post_{name}',b); add(f'delta_{name}',a-b)
        for size in (3,9):
            add(f'dNBR_mean_{size}',uniform_filter(delta['NBR'],size=size,mode='reflect'))
            add(f'dNDVI_mean_{size}',uniform_filter(delta['NDVI'],size=size,mode='reflect'))
        add('sar_VH_delta_mean_5',uniform_filter((sarpre[1]-sarpost[1])/20,size=5,mode='reflect'))
    else: raise ValueError(f'Unknown task: {chip_id}')
    if any(g!=grids[0] for g in grids[1:]): raise ValueError(f'Input grids disagree for {chip_id}')
    return Chip(chip_id,task,np.stack(features),names,valid,{k:str(v) for k,v in paths.items()})

def read_training_mask(chip_id: str, paths: dict[str,Path]) -> np.ndarray:
    # This is deliberately separate from every inference path.
    sample=next(iter(paths.values()))
    p=sample.parent.parent/'masks'/f'{chip_id}_MASK.tif'
    size=256 if chip_id.startswith('AF_') else 512
    expected='active_fire' if chip_id.startswith('AF_') else 'severity'
    with rasterio.open(sample) as ds: input_grid=(ds.width,ds.height,ds.crs,ds.transform)
    with rasterio.open(p) as ds:
        if ds.count!=1 or ds.descriptions!=(expected,) or (ds.height,ds.width)!=(size,size):
            raise ValueError(f'Unexpected label schema for {p}')
        if (ds.width,ds.height,ds.crs,ds.transform)!=input_grid:
            raise ValueError(f'Label grid disagrees with inputs for {chip_id}')
        y=ds.read(1)
        if not np.issubdtype(y.dtype,np.integer): raise ValueError(f'Labels must be integers: {p}')
    allowed=[0,1,255] if chip_id.startswith('AF_') else [0,1,2,3,255]
    if not np.isin(y,allowed).all(): raise ValueError(f'Invalid classes in {p}')
    return y.astype(np.uint8)
