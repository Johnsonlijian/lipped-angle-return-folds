from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent/'R17_independent_rebuild_20261109'
OUT = ROOT/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':10,
                    'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False,
                    'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False,
                    'axes.linewidth':.7,'lines.linewidth':1.5,'savefig.dpi':220})
for f in (OLD/'figures').iterdir():
    if f.suffix in ['.pdf','.svg','.png']: shutil.copy2(f,OUT/f.name)

def save(fig,name):
    for ext in ['pdf','svg','png']: fig.savefig(OUT/(name+'.'+ext),bbox_inches='tight')
    plt.close(fig)

sh=pd.read_csv(ROOT/'data/shell_comparison.csv')
fig,axs=plt.subplots(1,2,figsize=(7.2,3.4),layout='constrained')
names=['plate','simple','return06','return08leg','return08lip','return10leg','return10lip','return50']
labels=['Plate','No return',r'$s=.06$, lip',r'$s=.08$, leg',r'$s=.08$, lip',r'$s=.10$, leg',r'$s=.10$, lip',r'$s=.50$, leg']
colors=['#333333','#0072B2','#E69F00','#009E73','#CC79A7','#56B4E9','#D55E00','#777700']
for i,(name,label,c) in enumerate(zip(names,labels,colors)):
    g=sh[sh.name.str.match('v_'+name+'_m')].sort_values('mesh')
    axs[0].plot(g.mesh,g.error_percent,marker=['o','s','^','v','D','P','X','>'][i],ms=4,color=c,label=label,ls='--' if i%2==0 else '-')
axs[0].axhline(0,color='.7',lw=.7,zorder=0)
axs[0].set(xscale='log',xticks=[12,24,48,96],xticklabels=['12','24','48','96'],xlabel=r'Mesh divisor $m$ (spacing $a/m$)',ylabel=r'$100(k_{shell}/k_{FSM}-1)$ (%)',title='(a) Shell mesh refinement')
axs[0].xaxis.set_minor_formatter(NullFormatter())
axs[0].legend(ncol=2,fontsize=7.5,loc='upper right',columnspacing=.7)
ratios=[]
for s,c,mark in [(0,'#0072B2','o'),(.5,'#D55E00','s')]:
    group=[]
    for r in [.2,.6]:
        q=sh[(sh.name.str.startswith('v_transfer')) & (sh.s==s) & (sh.r==r)].set_index('x')
        row=dict(s=s,r=r,shell=q.loc[1.5,'k_shell']/q.loc[1.1,'k_shell'],fsm=q.loc[1.5,'k_fsm']/q.loc[1.1,'k_fsm'])
        ratios.append(row);group.append(row)
    axs[1].plot([v['r'] for v in group],[v['fsm'] for v in group],color=c,marker=mark,ms=5,label=f'FSM, s={s:g}')
    axs[1].plot([v['r'] for v in group],[v['shell'] for v in group],color=c,marker=mark,ls='--',mfc='white',ms=7,label=f'Shell, s={s:g}')
axs[1].set(xlabel=r'Lip ratio $r=c/a$',ylabel=r'$k(1.5,r,s)/k(1.1,r,s)$',title='(b) Limb-ratio sensitivity',xticks=[.2,.4,.6],ylim=(1.,1.08))
axs[1].legend(loc='lower left',fontsize=8)
save(fig,'fig06_shell')
pd.DataFrame(ratios).to_csv(ROOT/'data/shell_ratio_checks.csv',index=False)

coarse=json.loads((ROOT/'data/crossover_results.json').read_text())
fine=json.loads((ROOT/'data/crossover_fine.json').read_text())
best=[v for v in coarse if v['r']==.5 and v['mesh']==48]+fine
pd.DataFrame([dict(x=v['x'],r=v['r'],s=v['s'],mesh=v['mesh'],d_over_a=v['d_over_a'],d_over_t=v['d_over_t'],lip_half_wave=v['lip']['half_wave'],leg_half_wave=v['leg']['half_wave'],k=v['leg']['k'],lip_P=v['lip']['lips'],leg_P=v['leg']['lips']) for v in best]).sort_values(['r','x']).to_csv(ROOT/'data/crossover_final.csv',index=False)
fig,axs=plt.subplots(2,2,figsize=(7.2,5.9),layout='constrained')
ax=axs[0,0]
for s,c,ls in [(.06,'#0072B2','-'),(.08,'#D55E00','--'),(.10,'#009E73','-.')]:
    g=pd.read_csv(ROOT/'data'/f'x1.30_r0.60_s{s:.4f}_m12.csv');g=g[g['mode']==1]
    ax.plot(g.half_wave,g.k,color=c,ls=ls,label=f's={s:.2f}')
ax.axvspan(.2,2,color='.9',alpha=.6,zorder=0)
ax.axvline(2,color='.5',lw=.8,ls=':')
ax.text(.40,7.9,'Initial search window',fontsize=8,color='.35')
ax.set(xscale='log',xlim=(.35,5),ylim=(2.2,8.5),xlabel=r'Half-wavelength $\ell/a$',ylabel=r'Buckling coefficient $k_a$',title='(a) Trough moves beyond the cutoff')
ax.set_xticks([.5,1,2,5],labels=['0.5','1','2','5'])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.legend(loc='lower left',fontsize=8)
ax=axs[0,1];troughs=[]
for f in sorted((ROOT/'data').glob('x1.30_r0.60_s*_m12.csv')):
    g=pd.read_csv(f);g=g[g['mode']==1].reset_index(drop=True)
    for j in find_peaks(-g.k.to_numpy())[0]:
        row=g.iloc[j]
        if row.s<=.15 and row.half_wave<6:
            label='lip' if row.lips>.5 else 'leg'
            troughs.append(dict(s=row.s,k=row.k,branch=label,L=row.half_wave))
td=pd.DataFrame(troughs)
for label,c,marker in [('lip','#0072B2','o'),('leg','#D55E00','s')]:
    g=td[td.branch==label].sort_values('s')
    ax.plot(g.s,g.k,color=c,marker=marker,ms=3,label=label.capitalize()+' trough')
cross=next(v for v in fine if v['x']==1.3)
ax.axvline(cross['s'],color='.4',ls=':',lw=1)
ax.plot(cross['s'],cross['leg']['k'],'*',color='black',ms=10)
ax.set(xlim=(.035,.125),ylim=(2.7,7.3),xlabel=r'Return ratio $s=d/c$',ylabel=r'Stationary-trough coefficient $k_a$',title='(b) Stationary troughs exchange order')
ax.legend(fontsize=8,loc='upper left')
ax.text(.089,4.3,r'$s_*\simeq0.0871$',fontsize=9)
ax=axs[1,0]
for r,c,mark in [(.5,'#0072B2','o'),(.6,'#D55E00','s')]:
    b=sorted([v for v in best if v['r']==r],key=lambda v:v['x'])
    prev=sorted([v for v in coarse if v['r']==r and v['mesh']==(24 if r==.5 else 48)],key=lambda v:v['x'])
    ax.plot([v['x'] for v in b],[v['s'] for v in b],color=c,marker=mark,label=f'r={r:g}')
    ax.plot([v['x'] for v in prev],[v['s'] for v in prev],color=c,marker='x',ls=':',ms=6)
ax.set(xlabel=r'Limb ratio $x=a/b$',ylabel=r'Crossover return ratio $s_*$',xticks=[1,1.3,1.6],ylim=(.069,.092),title='(c) Geometry-dependent thresholds')
ax.legend(fontsize=8,loc='upper left',ncol=2)
ax=axs[1,1];br=sh[sh.name.str.startswith('v_cross')]
for key,label,c,mark in [('k_shell','Shell at fixed trough lengths','#0072B2','o'),('k_fsm','FSM at the same lengths','#D55E00','s')]:
    vals=[]
    for s in [.086,.088]:
        g=br[br.s==s]
        vals.append(float(g[g.length>2][key].iloc[0]-g[g.length<1][key].iloc[0]))
    ax.plot([.086,.088],vals,color=c,marker=mark,label=label)
ax.axhline(0,color='.5',lw=.8);ax.axvline(cross['s'],color='.4',ls=':',lw=1)
ax.set(xlim=(.0857,.0883),xlabel=r'Return ratio $s=d/c$',ylabel=r'$k_{lip}-k_{leg}$',title='(d) Independent shell bracket',xticks=[.086,.087,.088])
ax.legend(fontsize=7.7,loc='upper left')
save(fig,'fig07_crossover')
print(pd.DataFrame(ratios).to_string(index=False))
print('Figures and final crossover table saved.')
