from pathlib import Path
import sys,json,re
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'R17_independent_rebuild_20261109'/'src'))
import research_scan as fsm
import numpy as np
import pandas as pd

def inner(a,b,h):
    return float(np.sum(h*(2*np.sum(a[:-1]*b[:-1],axis=1)+np.sum(a[:-1]*b[1:],axis=1)+np.sum(a[1:]*b[:-1],axis=1)+2*np.sum(a[1:]*b[1:],axis=1))/6))

def compare(folder):
    meta=json.loads((folder/'model.json').read_text());name=meta['name']
    frames=json.loads((folder/(name+'_results.json')).read_text())
    x,r,s,mesh,L=[meta[k] for k in ['x','r','s','mesh','length']]
    xy=np.array(meta['xy']);h=np.linalg.norm(np.diff(xy,axis=0),axis=1)
    n,nz=meta['ncross'],meta['nz'];sine=np.sin(np.pi*np.arange(nz+1)/nz)
    factor=12*(1-.3**2)*10000/(np.pi**2*210000)
    if not meta['plate']:
        nodes,elems,tags,a=fsm.build(x,r,s,nmesh=48)
        sig,eigs,shapes=fsm.solver.signature(nodes,elems,np.array([L*a]))
        P,disp=fsm.participation(shapes[0,0],nodes,tags)
        arcl=np.r_[0,np.cumsum(h)]
        fsml=np.r_[0,np.cumsum(np.linalg.norm(np.diff(nodes[:,1:3],axis=0),axis=1))]
        target=np.column_stack([np.interp(arcl,fsml,disp[:,j]) for j in range(2)])
        expected=eigs[0,0]*factor
    else:expected=4.;P={'lips':0};target=np.column_stack([np.zeros(n),np.sin(np.pi*xy[:,0]/100)])
    rows=[]
    for frame in frames[1:]:
        eig=float(re.search(r'EigenValue\s*=\s*([-+.0-9Ee]+)',frame['description']).group(1))
        u=np.array(sorted(frame['displacement']))[:,1:3].reshape(nz+1,n,2)
        amp=np.einsum('i,ijk->jk',sine,u)/np.dot(sine,sine)
        projected=sine[:,None,None]*amp[None,:,:]
        fraction=np.sum(projected**2)/np.sum(u**2)
        mac=inner(amp,target,h)**2/(inner(amp,amp,h)*inner(target,target,h))
        weights=h*(np.sum(amp[:-1]**2,axis=1)+np.sum(amp[:-1]*amp[1:],axis=1)+np.sum(amp[1:]**2,axis=1))/3
        lips=float(weights[np.array([tag.startswith('lip') for tag in meta['tags']])].sum()/weights.sum())
        rows.append(dict(name=name,x=x,r=r,s=s,length=L,mesh=mesh,mode=frame['mode'],k_shell=eig*factor,k_fsm=expected,
                         error_percent=100*(eig*factor/expected-1),sine_fraction=fraction,MAC=mac,lips_shell=lips,lips_fsm=P['lips']))
    valid=[row for row in rows if row['sine_fraction']>.99]
    if not valid:raise RuntimeError(name+' has no matching half-wave')
    return max(valid,key=lambda v:v['MAC']),rows

if __name__=='__main__':
    records=[];allm=[]
    for folder in sorted((ROOT/'shell').iterdir()):
        if not folder.name.startswith('v_'):continue
        if (folder/(folder.name+'_results.json')).exists() and (folder/'model.json').exists():
            row,others=compare(folder);records.append(row);allm.extend(others)
    pd.DataFrame(records).to_csv(ROOT/'data'/'shell_comparison.csv',index=False)
    pd.DataFrame(allm).to_csv(ROOT/'data'/'shell_all_modes.csv',index=False)
    print(pd.DataFrame(records)[['name','mode','k_shell','k_fsm','error_percent','MAC','sine_fraction','lips_shell','lips_fsm']].to_string(index=False))
