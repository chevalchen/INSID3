from pathlib import Path
import sys, numpy as np,pandas as pd
d=Path(sys.argv[1]); df=pd.read_csv(d/'paired.csv'); classes=pd.read_csv(d/'classes.csv')
k=[v for v in 'ABCD' if v in df]
c=classes.iloc[:-1].set_index('part')
def gaps(g):
    result={'原始簇离理想值的差距':1-g.A,'候选覆盖损失':g.A-g.B,'候选上界到实际聚合的差距':g.B-g.C}
    if 'D' in g: result['CRF 净损失']=g.C-g.D
    return pd.DataFrame(result)
stage=gaps(c)
dominant=stage.mean().idxmax()
episodes=gaps(df);v=episodes[dominant]
order=stage[dominant].sort_values(ascending=False)
text=f"""# PASCAL-Part 1-shot · fold 0 诊断
已测 {len(df)}/2500 个 episode，{len(c)} 类。"""
if len(df)!=2500:text+=' 当前为部分清单的探索结果。'
text+='\n\n按每类累计前景交并量计算后等权平均，'+ '，'.join(f'{s} 为 {100*classes.iloc[-1][s]:.2f}%' for s in k)+'。A/B 按各 episode 的最优簇并集累计。\n\n'
text+='各项平均差值为 '+ '，'.join(f'{col} {stage[col].mean()*100:.2f} 个百分点' for col in stage)+'。'
text+=f'最大项为{dominant}。逐 episode 看，其中 {sum(v>.10)}/{len(v)} 个差值超过 10 个百分点，中位数 {100*v.median():.2f}，去掉两端各 5% 后均值 {100*v.sort_values().iloc[len(v)//20:len(v)-len(v)//20].mean():.2f} 个百分点。该项在 {sum(episodes.idxmax(axis=1)==dominant)} 个 episode 和 {sum(stage.idxmax(axis=1)==dominant)} 个部件类中居首。按类差值最大的五类为 '+ '、'.join(f'{i} ({100*a:.1f})' for i,a in order.head().items())+'。\n\n'
text+='逐一剔除一个部件类，最大项保持相同的次数为 '+str(sum(stage.drop(index=i).mean().idxmax()==dominant for i in stage.index))+f'/{len(stage)}。这只反映本 fold 的敏感性。种子选择与区域聚合尚不能区分。'
if 'D' not in k:text+=' 当前 D 缺失，CRF 对结果的影响未测得。'
text+='\n\n'
if dominant=='候选上界到实际聚合的差距':text+='下一步只做候选种子 oracle：固定候选及原合并规则，离线遍历候选种子。若最佳种子仍不能明显缩小 B−C，便推翻“种子选择是主要原因”的解释。'
elif dominant=='原始簇离理想值的差距':text+='下一步只做 patch 上界对照：把 64×64 网格中的每个 patch 当作单独区域，在相同 episode 和 1024×1024 网格上重算 A。当前 1−A 同时包含 patch 分辨率和簇合并造成的限制。如果拆开全部簇以后仍保留超过一半的 1−A，就推翻“聚类合并解释了大部分原始区域缺口”的假设。'
else:text+='下一步在同一配对离线放宽候选过滤，保持后续规则固定。若覆盖和实际聚合均无改善，便推翻候选筛选主导损失的解释。'
(d/'report.md').write_text(text+'\n')
