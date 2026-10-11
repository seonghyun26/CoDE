#!/usr/bin/env python3
"""Decode the same crystallographic targets/folds from one baseline's real embeddings."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from .affinity_density import decode,UNC
from .paths import REPRESENTATION

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def load_features(path):
    if path.suffix=='.npz':
        with np.load(path,allow_pickle=False) as data:
            ids=[str(p).lower() for p in data['pids']]
            matrix=np.asarray(data['features'],dtype=float)
        assert len(ids)==len(set(ids)) and matrix.ndim==2 and len(ids)==len(matrix)
        return dict(zip(ids,matrix))
    data=torch.load(path,map_location='cpu',weights_only=False)
    for key in ['features','feats','feat']:
        if key in data:
            data=data[key];break
    result={}
    for key,value in data.items():
        pid=str(key).lower();assert pid not in result
        if isinstance(value,torch.Tensor):value=value.detach().cpu().numpy()
        result[pid]=np.asarray(value,dtype=float).ravel()
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',choices=['DeepDTA','ProFSA','Nesso-1'],required=True)
    ap.add_argument('--features',type=Path,required=True)
    ap.add_argument('--provenance',type=Path,required=True)
    ap.add_argument('--layer',required=True)
    a=ap.parse_args()
    source=REPRESENTATION/'oof-predictions.csv';base=pd.read_csv(source)
    assert len(base)==732 and base.pid.is_unique
    features=load_features(a.features)
    missing=set(base.pid)-set(features)
    if missing:raise ValueError(f'{len(missing)} missing CL3 IDs: {sorted(missing)[:10]}')
    x=np.stack([features[pid] for pid in base.pid]);y=base[list(UNC)].to_numpy(dtype=float)
    assert x.shape[1]>1,'Scalar affinity predictions are not representation vectors.'
    assert np.isfinite(x).all() and np.isfinite(y).all()
    provenance=json.loads(a.provenance.read_text())
    if 'status' in provenance:assert provenance['status']=='complete'
    pred,baseline,fold,alphas=decode(x,y)
    assert np.array_equal(fold,base.fold.to_numpy())
    assert np.isfinite(pred).all()
    out=REPRESENTATION/'baselines'/a.name;out.mkdir(parents=True,exist_ok=True)
    scores=[];oof=base[['pid','fold',*UNC]].copy()
    for j,key in enumerate(UNC):
        oof[f'{a.name}__{key}']=pred[:,j]
        scores.append(dict(arm=a.name,descriptor=key,n=len(base),
                           rho=float(spearmanr(y[:,j],pred[:,j]).statistic),
                           r2=float(1-np.square(y[:,j]-pred[:,j]).sum()/np.square(y[:,j]-y[:,j].mean()).sum())))
    oof.to_csv(out/'oof-predictions.csv',index=False)
    scores=pd.DataFrame(scores);scores.to_csv(out/'decoding-statistics.csv',index=False)
    protocol=dict(name=a.name,features=str(a.features.resolve()),feature_sha256=sha(a.features),
                  source_provenance=provenance,source_provenance_sha256=sha(a.provenance),
                  feature_layer=a.layer,feature_dim=x.shape[1],n=len(base),
                  reference_oof_sha256=sha(source),reference_oof=str(source),
                  pid_order_sha256=hashlib.sha256('\n'.join(base.pid).encode()).hexdigest(),
                  protocol='same nested 5x3 ridge, alpha grid, train-fold scaling, targets and outer folds as CoE/CoDE',
                  selected_alphas=alphas,
                  scope='descriptive random-complex folds within CL3; not protein-family-disjoint validation')
    (out/'decoder.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(scores.to_string(index=False),flush=True)

if __name__=='__main__':main()
