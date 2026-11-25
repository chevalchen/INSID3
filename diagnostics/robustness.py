from pathlib import Path
import sys
import pandas as pd,numpy as np
root=Path(sys.argv[1])
df=pd.read_csv(root/'paired.csv')
rows=[]
for group in np.array_split(df.index,10):
    g=df.loc[group].groupby('part').sum(numeric_only=True)
    rates={k: (g[k+'_tp']/(g[k+'_tp']+g[k+'_fp']+g[k+'_fn']).clip(lower=1)).mean() for k in 'ABCD'}
    vals=dict(start=group[0],end=group[-1],**rates)
    vals.update({'1-A':1-rates['A'],'A-B':rates['A']-rates['B'],'B-C':rates['B']-rates['C'],'C-D':rates['C']-rates['D']})
    rows.append(vals)
blocks=pd.DataFrame(rows)
blocks.to_csv(root/'blocks.csv',index=False)
col=blocks[['1-A','A-B','B-C','C-D']].idxmax(axis=1)
text=f"\n按固定采样顺序分成 10 个连续块（每块 250 对），其中 {sum(col=='1-A')}/10 块的最大缺口为 1−A。各块的 1−A 范围为 {100*blocks['1-A'].min():.1f} 至 {100*blocks['1-A'].max():.1f} 个百分点。CRF 使 {sum(df.D-df.C>.05)} 对提高超过 5 个百分点，使 {sum(df.C-df.D>.05)} 对下降超过 5 个百分点。\n"
with (root/'report.md').open('a') as fp:fp.write(text)
