import sys, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
R17=ROOT.parent/'R17_independent_rebuild_20261109'
sys.path.insert(0,str(R17/'src'))
import research_scan as fsm
import numpy as np
import pandas as pd
import argparse

def spectrum(x,r,s,mesh=12,nlength=140,upper=20):
    nodes,elems,tags,a=fsm.build(x,r,s,nmesh=mesh)
    lengths=a*np.geomspace(.2,upper,nlength)
    sig,curves,shapes=fsm.solver.signature(nodes,elems,lengths)
    kfactor=12*(1-fsm.NU**2)*100**2/(np.pi**2*fsm.E)
    parts=[]
    for i,L in enumerate(lengths):
        for j in range(4):
            P,disp=fsm.participation(shapes[i,j],nodes,tags)
            parts.append(dict(x=x,r=r,s=s,mesh=mesh,half_wave=L/a,mode=j+1,k=curves[i,j]*kfactor,lips=P['lips']))
    name=f'x{x:.2f}_r{r:.2f}_s{s:.4f}_m{mesh}'
    pd.DataFrame(parts).to_csv(ROOT/'data'/(name+'.csv'),index=False)
    np.savez_compressed(ROOT/'data'/(name+'.npz'),nodes=nodes,elements=elems,tags=tags,lengths=lengths/a,curves=curves*kfactor,shapes=shapes)
    print(name,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',default='pilot');args=p.parse_args()
    if args.mode=='pilot':cases=[(1.3,.6,s) for s in [0,.04,.06,.075,.08,.10,.12,.15,.20,.25,.5]]
    else:cases=[(x,r,s) for x in [1,1.3,1.6] for r in [.4,.5,.6] for s in np.round(np.arange(0,.151,.01),4)]
    for x,r,s in cases:
        name=f'x{x:.2f}_r{r:.2f}_s{s:.4f}_m12.csv'
        if not (ROOT/'data'/name).exists():spectrum(x,r,s)
