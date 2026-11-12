"""Reproducible geometry sensitivity, transfer comparison, and manuscript figures."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
FIG=ROOT/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
                    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
                    'svg.fonttype':'none','axes.labelsize':9,'legend.fontsize':8})
colors=['#245E83','#D77928','#368675']
df=pd.concat([pd.read_csv(DATA/'grid.csv'),pd.read_csv(DATA/'extended.csv')],ignore_index=True)
assert len(df)==456 and not df.duplicated(['x','r','s']).any()
df.to_csv(DATA/'combined_scan.csv',index=False)
primary=df[df.s.isin([0,.25,.5])]
checks=pd.read_csv(DATA/'checks.csv')
replay=pd.read_csv(DATA/'fresh_replay.csv')
assert len(primary)==399 and len(replay)==630
summary={'primary_spectra':len(primary),'additional_spectra':57,'fresh_replay_rows':len(replay)}

def export(fig,name):
    for suffix in ['pdf','svg','png']:
        fig.savefig(FIG/f'{name}.{suffix}',dpi=220,bbox_inches='tight')
    plt.close(fig)

def panel(ax,letter,title):
    ax.set_title(f'{letter}  {title}',loc='left',fontweight='bold',pad=10)
    ax.grid(alpha=.15)

metrics=[]
predictions=[]
for s in [0,.25,.5]:
    matrix=primary[primary.s==s].pivot(index='r',columns='x',values='k')
    anchor=primary[(primary.s==s)&(primary.x==1.3)].set_index('r')
    for _,row in primary[(primary.s==s)&primary.x.between(1.1,1.5)&(primary.x!=1.3)&(primary.r!=.2)].iterrows():
        factor=matrix.loc[.2,row.x]/matrix.loc[.2,1.3]
        p=anchor.loc[row.r]
        simple=p.k*factor
        attenuation=(1-p.lips)/(1-anchor.loc[.2].lips)
        corrected=p.k*(1+(factor-1)*attenuation)
        predictions.append(dict(s=s,x=row.x,r=row.r,k=row.k,baseline=simple,weighted=corrected,
                                baseline_error=100*(simple/row.k-1),weighted_error=100*(corrected/row.k-1)))
    mat=matrix.loc[:,1.1:1.5].values
    u,singular,v=np.linalg.svd(mat,full_matrices=False)
    estimate=singular[0]*np.outer(u[:,0],v[0])
    errors=100*(estimate/mat-1)
    ratio=matrix[1.5]/matrix[1.1]
    metrics.append(dict(s=s,ratio_r02=ratio.loc[.2],ratio_r05=ratio.loc[.5],ratio_r06=ratio.loc[.6],
                        rank_one_rmse=float(np.sqrt(np.mean(errors**2))),rank_one_max=float(abs(errors).max()),
                        singular_ratio=float(singular[1]/singular[0])))
pred=pd.DataFrame(predictions)
pred.to_csv(DATA/'transfer_predictions.csv',index=False)
for metric in metrics:
    group=pred[pred.s==metric['s']]
    for model in ['baseline','weighted']:
        errors=group[f'{model}_error']
        metric[f'{model}_rmse']=float(np.sqrt(np.mean(errors**2)))
        metric[f'{model}_max']=float(errors.abs().max())
    metric['scored_cells']=len(group)
pd.DataFrame(metrics).to_csv(DATA/'separability_metrics.csv',index=False)
summary['separability']=metrics
paired=checks[(checks.upper==2)&checks.x.isin([1.,1.6])].merge(primary,on=['x','r','s'],suffixes=('_fine','_base'))
paired['k_difference_pct']=100*(paired.k_base/paired.k_fine-1)
paired['lip_difference']=paired.lips_base-paired.lips_fine
paired.to_csv(DATA/'convergence_comparison.csv',index=False)
summary['convergence_max_k_pct']=float(paired.k_difference_pct.abs().max())
summary['convergence_max_lip_absolute']=float(paired.lip_difference.abs().max())
summary['replay_rmse_pct']=float(np.sqrt(np.mean(replay.error_pct**2)))
summary['replay_max_pct']=float(replay.error_pct.abs().max())
summary['primary_boundary_rows']=int(primary.boundary.sum())
summary['additional_boundary_rows']=int(df[~df.s.isin([0,.25,.5])].boundary.sum())
summary['return_scan_boundary_rows']=int(pd.read_csv(DATA/'return_scan.csv').boundary.sum())

# Four actual computed cross-section eigenvectors; displacement is visually amplified.
fig,axs=plt.subplots(2,2,figsize=(7.2,6.0),layout='constrained')
for ax,(r,s),letter in zip(axs.flat,[(.2,0),(.6,0),(.2,.5),(.6,.5)],'abcd'):
    file=DATA/f'x1.60_r{r:.2f}_s{s:.2f}_m24_u2.0.npz'
    arr=np.load(file)
    xy=arr['nodes'][:,1:3]/100
    displacement=arr['displacements']
    displacement=displacement/np.linalg.norm(displacement,axis=1).max()*.10
    ax.plot(xy[:,0],xy[:,1],color='#A8ADB3',lw=2,label='Undeformed')
    ax.plot(xy[:,0]+displacement[:,0],xy[:,1]+displacement[:,1],color=colors[0],lw=1.6,label='Eigenvector × scale')
    row=checks[(checks.x==1.6)&(checks.r==r)&(checks.s==s)].iloc[0]
    panel(ax,letter,rf'$r={r:.2f},\ s={s:.1f}$; $P_\ell={row.lips:.3f}$')
    ax.set(xlabel='Horizontal coordinate / a',ylabel='Vertical coordinate / a',aspect='equal',xlim=(-.15,1.2),ylim=(-.15,.9))
axs[0,0].legend(loc='upper center')
export(fig,'fig01_modes')

fig,axs=plt.subplots(2,2,figsize=(7.4,5.9),layout='constrained')
for s,color in zip([0,.25,.5],colors):
    anchor=primary[(primary.s==s)&(primary.x==1.3)].sort_values('r')
    label=rf'$s={s:g}$'
    axs[0,0].plot(anchor.r,anchor.k,color=color,label=label)
    axs[0,1].plot(anchor.r,anchor.lips,color=color,label=label)
    matrix=primary[primary.s==s].pivot(index='r',columns='x',values='k')
    axs[1,0].plot(matrix.index,matrix[1.5]/matrix[1.1],color=color,label=label)
    axs[1,1].plot(anchor.r,anchor.half_wave,color=color,label=label)
for ax,letter,title in zip(axs.flat,'abcd',['Buckling response','Lip displacement participation','Limb-ratio sensitivity','Critical half-wavelength']):
    panel(ax,letter,title)
    ax.set_xlabel(r'Lip ratio $r=c/a$')
axs[0,0].set_ylabel(r'$k_a$'); axs[0,0].legend()
axs[0,1].set_ylabel(r'$P_\ell$')
axs[1,0].set_ylabel(r'$k_a(1.5,r,s)/k_a(1.1,r,s)$')
axs[1,1].set_ylabel(r'$\ell_*/a$')
export(fig,'fig02_sensitivity')

fig,axs=plt.subplots(1,3,figsize=(9.0,3.0),layout='constrained')
for ax,s,letter in zip(axs,[0,.25,.5],'abc'):
    group=pred[pred.s==s].groupby('r')[['baseline_error','weighted_error']].agg(lambda z:np.sqrt(np.mean(z**2)))
    ax.plot(group.index,group.baseline_error,color=colors[1],label='Fixed factor')
    ax.plot(group.index,group.weighted_error,color=colors[0],label='Participation weighted')
    panel(ax,letter,rf'Return ratio $s={s:g}$')
    ax.set(xlabel=r'$r=c/a$',ylabel='Transfer RMSE (%)')
axs[0].legend()
export(fig,'fig03_transfer')

fig,axs=plt.subplots(2,2,figsize=(7.4,5.8),layout='constrained')
for s,color in zip([0,.5],[colors[0],colors[2]]):
    group=replay[replay.s==s]
    axs[0,0].scatter(group.r,group.error_pct,s=9,alpha=.45,color=color,label=rf'$s={s:g}$')
    vals=paired[paired.s==s]
    axs[0,1].scatter(vals.r,vals.k_difference_pct,s=30,color=color)
for ax,letter,title in zip(axs.flat,'abcd',['Fresh reproduction of 630 source cells','Mesh and wavelength refinement','Extended signature curves','Return-fold screening']):
    panel(ax,letter,title)
axs[0,0].set(xlabel=r'$r=c/a$',ylabel='Source replay error (%)'); axs[0,0].legend()
axs[0,1].set(xlabel=r'$r=c/a$',ylabel='Coarse / refined difference (%)')
for s,color in zip([0,.5],[colors[0],colors[2]]):
    ar=np.load(DATA/f'x1.30_r0.20_s{s:.2f}_m24_u10.0.npz')
    k=ar['curves'][:,0]*12*(1-.3**2)*100**2/(np.pi**2*210000)
    axs[1,0].plot(ar['lengths'],k,color=color,label=rf'$s={s:g}$')
axs[1,0].axvspan(.2,2,color='#E6EDF1',alpha=.5,zorder=-1)
axs[1,0].set(xscale='log',xlabel=r'$\ell/a$',ylabel=r'$k_a(\ell)$',ylim=(0,15)); axs[1,0].legend()
ret=pd.read_csv(DATA/'return_scan.csv')
rt=ret[(ret.x==1.3)&(ret.r==.6)&(ret.nmesh==12)]
axs[1,1].plot(rt.s,rt.k,color=colors[0])
axs[1,1].scatter(rt[rt.boundary].s,rt[rt.boundary].k,color=colors[1],marker='x',label='Window boundary')
axs[1,1].set(xlabel=r'$s=d/c$ at $r=0.6$',ylabel=r'$k_a$ within $0.2\leq\ell/a\leq2$'); axs[1,1].legend()
export(fig,'fig04_verification')

# Matrix rank residuals visualise how separability changes with return restraint.
fig,axs=plt.subplots(1,3,figsize=(8.5,3.2),layout='constrained')
for ax,s,letter in zip(axs,[0,.25,.5],'abc'):
    mat=primary[(primary.s==s)&primary.x.between(1.1,1.5)].pivot(index='r',columns='x',values='k')
    u,values,v=np.linalg.svd(mat.values,full_matrices=False)
    err=100*(values[0]*np.outer(u[:,0],v[0])/mat.values-1)
    im=ax.imshow(err,origin='lower',aspect='auto',extent=(1.05,1.55,.1375,.6125),cmap='RdBu_r',vmin=-2.6,vmax=2.6)
    ax.set(xlabel=r'$x=a/b$',ylabel=r'$r=c/a$')
    ax.set_title(f'{letter}  s={s:g}',loc='left',fontweight='bold')
fig.colorbar(im,ax=list(axs),label='Best rank-one residual (%)',shrink=.85)
export(fig,'fig05_separability')
(DATA/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
