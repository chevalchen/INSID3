import csv,json,sys,re
from pathlib import Path
from collections import defaultdict
import numpy as np
run=Path(sys.argv[1]); root=run/'full'
assert (root/'manifest.jsonl').read_bytes()==(Path(__file__).parent/'manifest0/manifest.jsonl').read_bytes()
rows=list(csv.DictReader((root/'paired.csv').open()))
assert len(rows)==2500
counts=defaultdict(lambda:np.zeros(3,dtype=np.int64))
for i,row in enumerate(rows):
    with np.load(root/f'{i:04}.npz') as data:
        assert list(data['shape'])==[1024,1024] and data['labels'].shape==data['raw'].shape==(64,64)
        assert set(data['merged']).issubset(data['candidates'])
        assert np.array_equal(data['raw'],np.isin(data['labels'],data['merged']))
        assert float(row['A'])>=float(row['B'])-1e-14
        if i<8:
            with np.load(run/'smoke'/f'{i:04}.npz') as sm:
                for key in data.files:assert np.array_equal(data[key],sm[key]),(i,key)
    counts[row['part']]+=np.array([int(row['D_'+key]) for key in ('tp','fp','fn')])
exact=np.mean([v[0]/max(1,v.sum()) for v in counts.values()])*100
replay=json.loads((root/'official_replay.json').read_text())['class_mIoU_percent']
assert abs(exact-replay)<1e-4
log=(run.parent/(run.name+'.log')).read_text()
m=re.findall(r'official mIoU, peak bytes, seconds ([\d.]+) (\d+) ([\d.]+)',log)
official,peak,secs=map(float,m[-1])
assert abs(exact-official)<1e-4
msg=f'\n\n完整评测用时 {secs:.1f} 秒，PyTorch 峰值已分配显存 {peak/1024**3:.2f} GiB。按整数交并量重算的 D 为 {exact:.7f}%，推理时官方 AverageMeter 为 {official:.7f}%，原版 evaluate 单独回放为 {replay:.7f}%。完整清单与预先固定的 2500 个实例配对逐字节一致。8 个端到端对照的 CRF 后掩码开关导出逐像素一致，全部导出字段也通过独立冒烟与完整运行的复核。\n'
with (root/'report.md').open('a') as f:f.write(msg)
print(msg)
