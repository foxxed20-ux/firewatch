"""The shared, offline model entry point for the service and competition CLI."""
from __future__ import annotations
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from .data import discover_chips,read_chip,FEATURE_VERSION

def _root(model_dir=None):
    return Path(model_dir or os.environ.get('MODEL_ROOT') or Path(__file__).resolve().parents[1]/'model_bundle').resolve()

def _verify_spec(root,spec):
    if spec['backend']=='ensemble':
        if set(spec['components'])!={'cnn','tree'}: raise ValueError('Incomplete ensemble')
        if not 0<=float(spec['tree_weight'])<=1: raise ValueError('Invalid ensemble weight')
        for component in spec['components'].values(): _verify_spec(root,component)
        return
    if spec['backend'] not in ('lightgbm','torch'): raise ValueError('Unknown model backend')
    folder=(root/spec['path']).resolve()
    if not folder.is_relative_to(root): raise ValueError('Unsafe model path')
    required={'model.txt','metadata.json'} if spec['backend']=='lightgbm' else {'best.pt'}
    if not required.issubset(spec['sha256']): raise ValueError('Missing artifact hashes')
    for name,expected in spec['sha256'].items():
        file=(folder/name).resolve()
        if not file.is_relative_to(folder) or not file.is_file(): raise ValueError(f'Missing artifact {name}')
        if hashlib.sha256(file.read_bytes()).hexdigest()!=expected: raise ValueError(f'Corrupt artifact {name}')

def describe_models(model_dir=None):
    root=_root(model_dir); path=root/'manifest.json'
    if not path.is_file():
        return {'available':False,'bundle_id':None,'tasks':{},'reason':'Model bundle has not been installed'}
    try:
        data=json.loads(path.read_text(encoding='utf-8'))
        if set(data['tasks'])!={'af','bs'}: raise ValueError('Both AF and BS are required')
        if data['feature_version']!=FEATURE_VERSION: raise ValueError('Incompatible features')
        for spec in data['tasks'].values(): _verify_spec(root,spec)
        return {**data,'available':True}
    except (OSError,KeyError,ValueError) as exc:
        return {'available':False,'bundle_id':None,'tasks':{},'reason':str(exc)}

@lru_cache(maxsize=8)
def _load(root_string,task,manifest_digest):
    root=Path(root_string); manifest=json.loads((root/'manifest.json').read_text())
    if manifest['feature_version']!=FEATURE_VERSION: raise ValueError('Incompatible feature version')
    spec=manifest['tasks'][task]; _verify_spec(root,spec)
    if spec['backend']=='ensemble':
        parts={k:_load_component(root,task,v) for k,v in spec['components'].items()}
        return manifest,spec,{k:v[0] for k,v in parts.items()},{k:v[1] for k,v in parts.items()},{k:v[2] for k,v in parts.items()}
    model,meta,device=_load_component(root,task,spec)
    return manifest,spec,model,meta,device

def _load_component(root,task,spec):
    folder=(root/spec['path']).resolve()
    if spec['backend']=='lightgbm':
        import lightgbm as lgb
        return lgb.Booster(model_file=str(folder/'model.txt')),json.loads((folder/'metadata.json').read_text()),'cpu'
    if spec['backend']=='torch':
        import torch
        from .models import build_model
        checkpoint=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        cfg=checkpoint['model']; model=build_model(cfg['in_channels'],task,cfg['base_channels'])
        model.load_state_dict(checkpoint['model_state'],strict=True)
        requested=os.environ.get('DEVICE','auto')
        device='cuda' if requested=='auto' and torch.cuda.is_available() else ('cpu' if requested=='auto' else requested)
        model=model.to(device).eval()
        return model,checkpoint,device
    raise ValueError(f"Unknown backend {spec['backend']}")

def _probabilities(chip,spec,model,meta,device):
    expected=meta['feature_names'] if spec['backend']=='lightgbm' else meta['feature_meta']['names']
    if chip.feature_names!=expected: raise ValueError('Ordered feature names disagree with model')
    if spec['backend']=='lightgbm':
        # Bounded pixel batches; caller features are never changed.
        flat=chip.x.reshape(chip.x.shape[0],-1)
        probs=[]
        for start in range(0,flat.shape[1],65536):
            probs.append(model.predict(flat[:,start:start+65536].T,num_threads=2))
        p=np.concatenate(probs)
        if chip.task=='af':
            score=p.reshape(chip.x.shape[1:]).astype(np.float32)
        else:
            p=p.reshape(*chip.x.shape[1:],4).astype(np.float32)
            score=np.moveaxis(p,-1,0)
    else:
        import torch
        mean=np.asarray(meta['normalizer']['mean'],np.float32)[:,None,None]
        std=np.asarray(meta['normalizer']['std'],np.float32)[:,None,None]
        x=(np.where(np.isfinite(chip.x),chip.x,mean)-mean)/std
        with torch.inference_mode():
            logits=model(torch.from_numpy(x[None]).to(device)).float()
            p=(logits.sigmoid() if chip.task=='af' else logits.softmax(1))[0].cpu().numpy()
        score=p[0] if chip.task=='af' else p
    return score

def predict_features(chip,model_dir=None,bundle_id=None):
    root=_root(model_dir); raw=(root/'manifest.json').read_bytes()
    manifest,spec,model,meta,device=_load(str(root),chip.task,hashlib.sha256(raw).hexdigest())
    if bundle_id and bundle_id!=manifest['bundle_id']: raise ValueError(f'Unavailable bundle {bundle_id}')
    if spec['backend']=='ensemble':
        scores={k:_probabilities(chip,component,model[k],meta[k],device[k]) for k,component in spec['components'].items()}
        weight=float(spec['tree_weight']);score=((1-weight)*scores['cnn']+weight*scores['tree']).astype(np.float32)
        postprocess=spec['postprocess']
        hashes={k:v['sha256'] for k,v in spec['components'].items()}
    else:
        score=_probabilities(chip,spec,model,meta,device)
        postprocess=spec.get('postprocess',meta); hashes=spec['sha256']
    if chip.task=='af': mask=(score>=float(postprocess['threshold'])).astype(np.uint8)
    else:
        multipliers=np.asarray(postprocess.get('class_multipliers',[1,1,1,1]),np.float32)
        if multipliers.shape!=(4,) or not np.isfinite(multipliers).all() or (multipliers<=0).any(): raise ValueError('Invalid class multipliers')
        mask=(score*multipliers[:,None,None]).argmax(0).astype(np.uint8)
    if mask.shape!=chip.x.shape[1:] or not np.isin(mask,[0,1] if chip.task=='af' else [0,1,2,3]).all():
        raise RuntimeError('Invalid model output')
    return {'class_map':mask,'observation_valid':chip.observation_valid,'score':score,'provenance':{'bundle_id':manifest['bundle_id'],'task':chip.task,'backend':spec['backend'],'device':device,'feature_version':FEATURE_VERSION,'score_is_calibrated_probability':False,'cloud_policy':'All pixels predicted; observation_valid is informational, never an output mask','model_sha256':hashes}}

def predict_chip(dataset_root,chip_id,bundle_id=None,model_dir=None):
    inputs=discover_chips(dataset_root)
    if chip_id not in inputs: raise FileNotFoundError(f'Chip {chip_id} not found')
    return predict_features(read_chip(chip_id,inputs[chip_id]),model_dir,bundle_id)
