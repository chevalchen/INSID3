import argparse, csv, itertools, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
REPO = Path('/data6/chensq/INSID3')
sys.path.insert(0,str(REPO))
from datasets.pascal_part import DatasetPASCALPart
from models.insid3 import INSID3
from utils.refinement import upsample_mask
from utils.clustering import agglomerative_clustering
from utils.metrics import Evaluator, AverageMeter

def oracle(labels, gt, ids, ignore=None):
    valid = np.ones(gt.shape,bool) if ignore is None else ~ignore
    l,y=labels[valid],gt[valid]
    area=np.bincount(l)
    pos=np.bincount(l[y],minlength=len(area))
    ids=np.asarray(ids,dtype=int)
    ids=ids[np.argsort(-pos[ids]/area[ids].clip(1),kind='stable')]
    tp=np.r_[0,np.cumsum(pos[ids])]
    fp=np.r_[0,np.cumsum(area[ids]-pos[ids])]
    fn=y.sum()-tp
    k=np.argmax(tp/(tp+fp+fn).clip(1))
    return [int(tp[k]),int(fp[k]),int(fn[k])]

def check():
    l=np.array([0,0,1,2,2])
    for bits in itertools.product([False,True],repeat=5):
        y=np.array(bits)
        for ids in [[],[1,2],[0,1,2]]:
            for ig in [np.zeros(5,bool),l==1]:
                best=0
                for n in range(len(ids)+1):
                    for s in itertools.combinations(ids,n):
                        m=np.isin(l,s)
                        best=max(best,(m&y&~ig).sum()/max(1,((m|y)&~ig).sum()))
                t,p,f=oracle(l,y,ids,ig)
                assert abs(best-t/max(1,t+p+f))<1e-10

def capture(frame,event,result):
    if event=='call' and frame.f_code.co_name=='_finalize_mask':
        obs['raw']=frame.f_locals['mask'].cpu().numpy()
    if event!='return': return
    v=frame.f_locals
    name=frame.f_code.co_name
    if name=='sample_episode' and 'support_img_ids' in v:
        meta.update(query=str(v['query_img_id']),support=str(v['support_img_ids'][0]),query_box=v['query_obj_box'],support_box=v['support_boxes'][0])
    if name=='_seed_and_aggregate':
        for key in ['cluster_labels','matched_ids','cross_sim_matched','seed_id','combined']:
            if key in v:
                a=v[key]
                obs[key]=a.detach().cpu().float().numpy() if torch.is_tensor(a) else a
        obs['merged']=v['cluster_labels'][result].unique().cpu().numpy()
    if name=='predict_mask' and 'candidate_mask' in v and 'cluster_labels' not in v:
        x=v['feat_tgt'][0].flatten(1).T
        obs['cluster_labels']=agglomerative_clustering(x,v['self'].tau).reshape(v['h'],v['w']).cpu().numpy()
    if name=='_finalize_mask':
        obs['pre']=upsample_mask(torch.as_tensor(obs['raw'],device=result.device),*result.shape).cpu().numpy()

def segment(batch,trace):
    model.set_reference(batch['ref_imgs'][0],batch['ref_masks'][0])
    model.set_target(batch['tgt_img'])
    if trace: sys.setprofile(capture)
    try: return model.segment()
    finally: sys.setprofile(None)

if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('output',type=Path)
    a.add_argument('--no-crf',action='store_true')
    a.add_argument('--manifest-only',action='store_true')
    a.add_argument('--limit',type=int,default=2500)
    args=a.parse_args()
    out=args.output
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    np.random.seed(0);random.seed(0);torch.manual_seed(0)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    check()
    ds=DatasetPASCALPart('/data6/chensq/datasets',0,1)
    print(len(ds),ds.cat_part_name,flush=True)
    if not args.manifest_only:
        weight=next((REPO/'pretrain').glob('*8aa4cbdd.pth'))
        torch.hub.set_dir(str(ROOT/'hub'))
        enc=torch.hub.load(str(ROOT/'dinov3'),'dinov3_vitl16',source='local',weights=str(weight))
        model=INSID3(enc,resize_to_orig_size=False,mask_refiner='bilinear' if args.no_crf else 'crf').cuda().eval()
        meter=AverageMeter('pascal_part',ds.class_ids)
    start=time.monotonic()
    for i in range(min(len(ds),args.limit)):
        meta=dict(index=i,seed=0)
        sys.setprofile(capture)
        try: b=ds[i]
        finally: sys.setprofile(None)
        c=int(b['class_id']);meta['part']=ds.cat_part_name[c]
        for img,mask in [(b['ref_imgs'][0],b['ref_masks'][0]),(b['tgt_img'],b['tgt_mask'])]:
            assert img.size==tuple(reversed(mask.shape))
        with (out/'manifest.jsonl').open('a') as file: print(json.dumps(meta),file=file)
        if args.manifest_only: continue
        obs={}
        plain=segment(b,False) if i<8 else None
        pred=segment(b,True)
        if plain is not None: assert torch.equal(plain,pred)
        gt=F.interpolate(b['tgt_mask'][None,None],size=pred.shape,mode='nearest')[0,0].bool()
        inter,union=Evaluator.classify_prediction(pred,gt)
        meter.update(inter,union,b['class_id'].cuda())
        y=gt.numpy();d=pred.cpu().numpy()
        lab=obs['cluster_labels'].astype('int16')
        ids=obs.get('matched_ids',np.array([])).astype(int)
        mapped=F.interpolate(torch.tensor(lab)[None,None].float(),size=pred.shape,mode='nearest')[0,0].numpy().astype(int)
        row=dict(meta)
        for k,counts in [('A',oracle(mapped,y,np.unique(mapped))),('B',oracle(mapped,y,ids))]+[(k,[int((m&y).sum()),int((m&~y).sum()),int((~m&y).sum())]) for k,m in [('C',obs['pre']),('D',d)]]:
            for suffix,val in zip(['tp','fp','fn'],counts): row[k+'_'+suffix]=val
            row[k]=counts[0]/max(1,sum(counts))
        assert row['D_tp']==int(inter[1]) and sum(row['D_'+j] for j in ['tp','fp','fn'])==int(union[1])
        np.savez_compressed(out/f'{i:04}.npz',labels=lab,candidates=ids,scores=obs.get('cross_sim_matched',[]),seed=obs.get('seed_id',-1),merged=obs.get('merged',[]),raw=obs['raw'],pre=np.packbits(obs['pre']),final=np.packbits(d),gt=np.packbits(y),shape=y.shape)
        with (out/'episodes.csv').open('a') as file:
            w=csv.DictWriter(file,row.keys())
            if i==0:w.writeheader()
            w.writerow(row)
        if i%50==0:print(i,row['D'],time.monotonic()-start,flush=True)
    if not args.manifest_only: print('official mIoU, peak bytes, seconds',meter.compute_iou()[0].item(),torch.cuda.max_memory_allocated(),time.monotonic()-start,flush=True)
