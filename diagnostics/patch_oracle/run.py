"""Compute optimal unions on the frozen EXP-001 evaluation grid."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def optimal(tp, area):
    order=np.argsort(-tp/area,kind='stable')
    hit=np.r_[0,tp[order].cumsum()]
    false=np.r_[0,(area-tp)[order].cumsum()]
    miss=tp.sum()-hit
    scores=hit/(hit+false+miss).clip(1)
    k=scores.argmax()
    return order[:k], [int(hit[k]),int(false[k]),int(miss[k])], scores[k]

p=argparse.ArgumentParser()
p.add_argument('arrays',type=Path)
p.add_argument('out',type=Path)
args=p.parse_args()
ref=Path(__file__).resolve().parents[1]/'results'
out=args.out
out.mkdir(exist_ok=False,parents=True)
manifest=(ref/'manifest.jsonl').read_bytes()
assert manifest==(args.arrays/'manifest.jsonl').read_bytes()
pairs=pd.read_csv(ref/'paired.csv')
metadata=[json.loads(line) for line in manifest.splitlines()]
hashes=dict(line.split()[::-1] for line in (ref/'arrays.sha256').read_text().splitlines())
rows=[]
focus=['sheep+TAIL','cat+NECK','cow+NECK','bird+WINGS','bird+TORSO']
examples={}
for r,m in zip(pairs.itertuples(index=False),metadata):
    i=r.index
    assert i==m['index']==len(rows) and (r.part,r.query,r.support)==(m['part'],m['query'],m['support'])
    path=args.arrays/f'{i:04}.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[path.name]
    with np.load(path) as data:
        assert tuple(data['shape'])==(1024,1024) and data['labels'].shape==(64,64)
        gt=np.unpackbits(data['gt']).reshape(1024,1024)
        labels=data['labels'].astype(int)
    # Each nearest-neighbour patch occupies exactly 16 x 16 evaluation pixels.
    tp=gt.reshape(64,16,64,16).sum((1,3),dtype=np.int64).ravel()
    keep,pc,pi=optimal(tp,np.full(4096,256))
    ids=labels.ravel()
    ct=np.zeros(ids.max()+1,dtype=np.int64)
    np.add.at(ct,ids,tp)
    chosen,cc,ci=optimal(ct,np.bincount(ids)*256)
    assert cc==[r.A_tp,r.A_fp,r.A_fn] and abs(ci-r.A)<1e-12 and pi+1e-12>=ci
    rec=dict(index=i,part=r.part,query=r.query,support=r.support,A_patch=pi,A_cluster=ci,gain=pi-ci)
    for k,vals in [('patch',pc),('cluster',cc)]:
        rec.update({k+'_'+s:v for s,v in zip(['tp','fp','fn'],vals)})
    rows.append(rec)
    if r.part in focus and r.part not in examples:
        examples[r.part]=(i,gt,np.isin(np.arange(4096),keep).reshape(64,64),np.isin(labels,chosen),pi,ci)
df=pd.DataFrame(rows)
assert len(df)==len(pairs)==len(metadata)==2500 and df.part.nunique()==31
df.to_csv(out/'episodes.csv',index=False)
tab=[]
for name,g in df.groupby('part'):
    row={'part':name,'n':len(g)}
    for k in ['patch','cluster']:
        t,f,n=g[[k+'_tp',k+'_fp',k+'_fn']].sum()
        row['A_'+k]=t/max(1,t+f+n)
    tab.append(row)
t=pd.DataFrame(tab)
t.loc[len(t)]=['macro',len(df),t.A_patch.mean(),t.A_cluster.mean()]
t['grid_gap']=1-t.A_patch
t['merge_gap']=t.A_patch-t.A_cluster
t['remaining_fraction']=t.grid_gap/(1-t.A_cluster)
assert round(100*t.iloc[-1].A_cluster,2)==55.56
t.to_csv(out/'classes.csv',index=False)
df.gain.describe(percentiles=[.1,.25,.5,.75,.9]).to_csv(out/'gain_distribution.csv')
fig,ax=plt.subplots(5,3,figsize=(8,11))
for j,name in enumerate(focus):
    i,gt,pa,cl,pi,ci=examples[name]
    for a,img,title in zip(ax[j],[gt,pa,cl],[f'{name} #{i} | GT',f'Patch {pi:.3f}',f'Cluster {ci:.3f}']):
        a.imshow(img,cmap='gray',vmin=0,vmax=1,interpolation='nearest');a.set_title(title);a.axis('off')
fig.tight_layout();fig.savefig(out/'oracle.png',dpi=150)
print(t[t.part.isin(focus+['macro'])].to_string(index=False))
print(df.gain.describe())
