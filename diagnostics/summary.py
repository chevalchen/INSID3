import argparse,ast
from pathlib import Path
import numpy as np,pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import DatasetPASCALPart
p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('final',type=Path);p.add_argument('--pre-only',action='store_true');a=p.parse_args()
df=pd.read_csv(a.final/'episodes.csv')
stages='ABC' if a.pre_only else 'ABCD'
if a.pre_only: df=df.drop(columns=['D','D_tp','D_fp','D_fn'])
deltas=['1-A','A-B','B-C']+([] if a.pre_only else ['C-D'])
df['1-A']=1-df.A
for u,v in zip(stages,stages[1:]):df[u+'-'+v]=df[u]-df[v]
table=[]
for part,g in df.groupby('part',sort=False):
    row=dict(part=part,n=len(g))
    for k in stages:
        counts=g[[k+'_tp',k+'_fp',k+'_fn']].sum()
        row[k]=counts.iloc[0]/max(1,counts.sum())
    table.append(row)
t=pd.DataFrame(table)
t.loc[len(t)]=dict(part='class-mIoU',n=len(df),**t[list(stages)].mean().to_dict())
t.to_csv(a.final/'classes.csv',index=False)
df.to_csv(a.final/'paired.csv',index=False)
df[list(stages)+deltas].describe(percentiles=[.1,.25,.5,.75,.9]).to_csv(a.final/'distribution.csv')
ds=DatasetPASCALPart('/data6/chensq/datasets',0,1)
def readmask(arr,shape):return np.unpackbits(arr).reshape(shape).astype(bool)
def overlay(img,mask,color):
    im=np.asarray(img,dtype=float)/255
    m=np.array(Image.fromarray(mask).resize(img.size,Image.Resampling.NEAREST))
    im[m]=.47*im[m]+.53*np.array(color)
    return im
chosen=[]
def pick(title,series):
    for ix in series.sort_values(ascending=False).index:
        if ix not in [v for _,v in chosen]:chosen.append((title,ix));break
pick('Low raw ceiling',1-df.A)
pick('Candidate coverage drop',df.A-df.B)
pick('Good coverage, poor prediction',(df.B-df.C)[df.B>.65])
if not a.pre_only and (df.C-df.D>.05).any():pick('CRF harm',df.C-df.D)
last=stages[-1]
pick('Median case',-(df[last]-df[last].median()).abs())
pick('Success',df[last])
cols=['Support + GT','Query + GT','Raw clusters','Seed','Before CRF']+([] if a.pre_only else ['After CRF'])
fig,axes=plt.subplots(len(chosen),len(cols),figsize=(2.45*len(cols),2.2*len(chosen)),squeeze=False)
selected=[]
for j,(title,ix) in enumerate(chosen):
    r=df.loc[ix];i=int(r['index'])
    selected.append({'case':title,'index':i,'part':r.part,**{key:r[key] for key in stages}})
    raw=np.load(a.raw/f'{i:04}.npz')
    def load_crop(key):
        x,y,e,f=ast.literal_eval(r[key+'_box'])
        return Image.fromarray(np.asarray(Image.open(ds.img_file.format(r[key])).convert('RGB'))[y:f,x:e])
    # Replay reference mask from exactly the same official sampler
    if j==0:
        np.random.seed(0);refs={}
        for h in range(max(v for _,v in chosen)+1):
            batch=ds[h]
            if h in [v for _,v in chosen]:refs[h]=batch['ref_masks'][0].numpy().astype(bool)
    img=load_crop('query');lab=raw['labels'];shape=tuple(raw['shape'])
    rng=np.random.default_rng(73)
    images=[overlay(load_crop('support'),refs[i],(0,1,.2)),overlay(img,readmask(raw['gt'],shape),(0,1,.2)),rng.random((lab.max()+1,3))[lab],overlay(img,lab==raw['seed'],(1,.65,0)),overlay(img,readmask(raw['pre'],shape),(0,.75,1))]
    if not a.pre_only:images.append(overlay(img,readmask(np.load(a.final/f'{i:04}.npz')['final'],shape),(0,.75,1)))
    for c,(ax,im) in enumerate(zip(axes[j],images)):
        ax.imshow(im,interpolation='nearest',aspect='auto',extent=[0,img.width,0,img.height]);ax.set_xticks([]);ax.set_yticks([])
        if j==0:ax.set_title(cols[c])
    axes[j,0].set_ylabel(f'{title}\n#{i} {r.part}',fontsize=8)
    axes[j,-1].set_xlabel('  '.join(f'{k}={r[k]:.2f}' for k in stages))
pd.DataFrame(selected).to_csv(a.final/'figure_cases.csv',index=False)
fig.tight_layout();fig.savefig(a.final/'diagnostics.png',dpi=140);plt.close(fig)
print(t.tail(1).to_string(index=False))
