from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'R17_independent_rebuild_20261109'/'src'))
import research_scan as fsm
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar,brentq

def trough(x,r,s,branch,mesh=24):
    nodes,elems,tags,a=fsm.build(x,r,s,nmesh=mesh)
    factor=12*(1-.3**2)*10000/(np.pi**2*210000)
    bounds=(.65,1.1) if branch=='leg' else ((1.45,2.6) if r==.5 else (2.,3.5))
    def response(L):
        sig,eigs,shapes=fsm.solver.signature(nodes,elems,np.array([a*L]))
        return eigs[0,0]*factor
    opt=minimize_scalar(response,bounds=bounds,method='bounded',options={'xatol':1e-5})
    L=opt.x;sig,eigs,shapes=fsm.solver.signature(nodes,elems,np.array([a*L]))
    parts,disp=fsm.participation(shapes[0,0],nodes,tags)
    if abs(L-bounds[0])<.001 or abs(L-bounds[1])<.001:raise RuntimeError('Boundary trough '+str((x,r,s,branch,L)))
    return dict(half_wave=L,k=float(eigs[0,0]*factor),lips=parts['lips'])

def crossover(x,r,mesh):
    log=[]
    def gap(s):
        lip=trough(x,r,s,'lip',mesh);leg=trough(x,r,s,'leg',mesh)
        log.append(dict(x=x,r=r,s=s,mesh=mesh,lip=lip,leg=leg))
        return lip['k']-leg['k']
    root=brentq(gap,.07,.09 if r==.5 else .1,xtol=2e-6)
    gap(root)
    result=log[-1];result['d_over_a']=root*r;result['d_over_t']=root*r*100
    return result,log

if __name__=='__main__':
    records=[];alllog=[]
    for mesh in [12,24,48]:
        for x in [1,1.3,1.6]:
            for r in [.5,.6]:
                rec,log=crossover(x,r,mesh);records.append(rec);alllog+=log
                print(rec,flush=True)
                (ROOT/'data'/'crossover_results.json').write_text(json.dumps(records,indent=2))
                (ROOT/'data'/'crossover_iterations.json').write_text(json.dumps(alllog,indent=2))
