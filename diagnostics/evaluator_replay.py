"""Check saved D masks using the unmodified official evaluate loop and sampler."""
import sys,json,time
from pathlib import Path
from types import SimpleNamespace
import torch,numpy as np
from run import ROOT, REPO
sys.path.insert(0,str(REPO))
from inference_segmentation import evaluate
class Replay:
    def __init__(self,folder):self.folder=folder;self.index=0
    def set_reference(self,image,mask):pass
    def set_target(self,image):pass
    def segment(self):
        with np.load(self.folder/f'{self.index:04}.npz') as data:
            mask=np.unpackbits(data['final']).reshape(tuple(data['shape'])).astype(bool)
        self.index+=1
        return torch.from_numpy(mask)
if __name__=='__main__':
    np.random.seed(0);torch.manual_seed(0);torch.set_num_threads(2)
    folder=Path(sys.argv[1])
    args=SimpleNamespace(dataset='pascal_part',data_root='/data6/chensq/datasets',fold=0,shots=1,num_workers=0,device='cpu')
    value=evaluate(args,Replay(folder),str(folder/'official_replay.log'))
    (folder/'official_replay.json').write_text(json.dumps({'class_mIoU_percent':value}))
