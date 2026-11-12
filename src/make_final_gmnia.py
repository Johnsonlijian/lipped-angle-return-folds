"""Regenerate the GMNIA figure from saved rows, displaying censoring explicitly."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
d=pd.read_csv(ROOT/'data/gmnia_results.csv').fillna({'tag':''})
b=d[(d.h==5)&(d.imp==1)&(d.tag=='')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':9,
 'axes.labelsize':9,'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,
 'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axs=plt.subplots(2,2,figsize=(7.2,5.0),layout='constrained')
colors=['#0072B2','#009E73','#E69F00','#D55E00']
def mark(ax,g,y,color):
    q=g[~g.post_peak]
    if len(q):
        ax.scatter(q.s,q[y],marker='^',s=48,facecolors='white',edgecolors=color,zorder=6)
        if y!='P_lip':
            for _,v in q.iterrows():
                ax.annotate('',xy=(v.s,v[y]+.025),xytext=(v.s,v[y]+.005),arrowprops={'arrowstyle':'->','color':color,'lw':.8})
for L,c in zip([800,1200,1800,2400],colors):
    g=b[(b.r==.6)&(b.L==L)&(b.fy==250)].sort_values('s')
    axs[0,0].plot(g.s,g.Pu_Py,'o-',color=c,ms=3,label=f'{L/1000:g} m')
    axs[0,0].plot(g.s,g.Pn_Py,'--',color=c,lw=1)
    mark(axs[0,0],g,'Pu_Py',c)
    axs[1,1].plot(g.s,g.P_lip,'o-',color=c,ms=3,label=f'{L/1000:g} m')
    mark(axs[1,1],g,'P_lip',c)
axs[0,0].axvline(.0868,color='.6',ls=':',lw=1)
axs[0,0].set(title='(a) Attained load and DSM prediction',ylabel=r'$P/P_y$')
axs[0,0].legend(ncol=2,loc='upper left')
for fy,r,c,m in [(250,.6,colors[0],'o'),(350,.6,colors[1],'s'),(450,.6,colors[3],'D'),(250,.5,'#CC79A7','x')]:
    g=b[(b.fy==fy)&(b.r==r)];q=g[g.post_peak]
    axs[0,1].scatter(q.s,q.ratio,s=15,marker=m,color=c,label=f'{fy} MPa, r={r}')
    mark(axs[0,1],g,'ratio',c)
axs[0,1].axhline(1,color='.6',lw=.8)
axs[0,1].set(title='(b) Strength ratio',ylabel=r'$P/P_{n,DSM}$')
axs[0,1].legend(loc='lower right')
sens=d[(d.h==5)&(d.tag=='')&(d.r==.6)&(d.L==800)&(d.fy==250)]
for imp,c in zip([.5,1,2],[colors[1],colors[0],colors[3]]):
    g=sens[sens.imp==imp].sort_values('s')
    axs[1,0].plot(g.s,g.Pu_Py,'o-',color=c,ms=3,label=f'Amplitude x{imp:g}')
    mark(axs[1,0],g,'Pu_Py',c)
axs[1,0].set(title='(c) Imperfection sensitivity',ylabel=r'$P/P_y$')
axs[1,0].legend(loc='upper left')
g=b[(b.r==.5)&(b.L==800)&(b.fy==250)].sort_values('s')
axs[1,1].plot(g.s,g.P_lip,'D--',color='#CC79A7',ms=3,label='r=0.5, 0.8 m')
axs[1,1].axhline(.5,color='.5',ls=':',lw=1)
axs[1,1].set(title='(d) Spatial displacement participation',ylabel=r'$P_{lip}$',ylim=(.47,1.01))
axs[1,1].legend(ncol=2,loc='lower left')
for ax in axs.flat:ax.set_xlabel(r'Return ratio $s=d/c$')
for ext in ['pdf','svg','png']:fig.savefig(ROOT/'figures'/('fig_gmnia.'+ext),dpi=240)
print('Wrote GMNIA PDF/SVG/PNG with lower-bound markers.')
