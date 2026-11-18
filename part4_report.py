"""Regenerate experiment figures and QSCS weight-sensitivity CSV from saved results."""
# Re-run this script from the project root to regenerate figures and report.
from pathlib import Path
import pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent; r=p/'results/final'; figs=r/'figures'; figs.mkdir(exist_ok=True)
grid=pd.read_csv(r/'parameter_search.csv'); q=pd.read_csv(r/'qscs_comparison.csv'); st=pd.read_csv(r/'stability_summary.csv'); sen=pd.read_csv(r/'feature_sensitivity.csv')
selected=q.set_index('algorithm').key.to_dict(); chosen=grid[grid.key.isin(selected.values())].copy();chosen.to_csv(r/'algorithm_comparison.csv',index=False)
weights=[(.45,.35,.20),(1/3,1/3,1/3),(.7,.2,.1),(.2,.7,.1),(.2,.2,.6),(.65,.35,0),(.65,0,.35),(0,.65,.35)]
abl=pd.DataFrame([{'scenario':name,'algorithm':a.algorithm,'score':w[0]*a.Q+w[1]*a.S+w[2]*a.C} for name,w in zip(['default','equal','quality','stability','coverage','no coverage','no stability','no quality'],weights) for a in q.itertuples()]);abl.to_csv(r/'qscs_weight_sensitivity.csv',index=False)
def bar(df,col,title,ylabel,name):
 fig,ax=plt.subplots(figsize=(8,4.7));ax.bar(df.algorithm,df[col]);ax.set(title=title,ylabel=ylabel);fig.tight_layout();fig.savefig(figs/name,dpi=180);plt.close(fig)
bar(chosen,'silhouette','Selected models: silhouette','Silhouette on assigned customers','algorithm_silhouette.png')
bar(chosen,'coverage','Selected models: coverage','Customer coverage','algorithm_coverage.png')
bar(st,'ari_mean','12-trial subsample stability','Mean ARI on common assigned customers','stability.png')
bar(q,'QSCS','Proposed quality–stability–coverage score','QSCS','qscs.png')
fig,ax=plt.subplots(figsize=(8,5))
for alg,g in sen.groupby('algorithm'):ax.plot(g.variant,g.ari_vs_baseline,marker='o',label=alg)
ax.set(xlabel='Preprocessing',ylabel='ARI vs baseline',title='Preprocessing sensitivity');ax.legend();fig.tight_layout();fig.savefig(figs/'preprocessing_sensitivity.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,5))
for alg,g in abl.groupby('algorithm'):ax.plot(g.scenario,g.score,marker='o',label=alg)
ax.set(ylabel='Score',title='QSCS weight sensitivity');ax.tick_params(axis='x',rotation=35);ax.legend();fig.tight_layout();fig.savefig(figs/'qscs_weight_sensitivity.png',dpi=180);plt.close(fig)
