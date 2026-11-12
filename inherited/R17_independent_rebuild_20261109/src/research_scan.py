"""Fresh finite-strip spectra and length-weighted translation participation."""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[variable] = '1'
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from scipy.linalg import eig
import legacy_forward as solver

ROOT = Path(__file__).resolve().parents[1]
E, NU = 210000.0, 0.3
solver.PROPS = np.array([[100, E, E, NU, NU, E/(2*(1+NU))]])
solver.N_EIGS = 4

def build(x, r, s, nmesh=12, at=100, thickness=1):
    a = at * thickness
    b, c = a/x, a*r
    points = solver.section_polyline(a, b, c, s*c, s > 0)
    names = solver.SEG_NAMES_COMPLEX if s > 0 else solver.SEG_NAMES_SIMPLE
    coords, tags = [], []
    for i, (start, end) in enumerate(zip(points[:-1], points[1:])):
        start, end = np.array(start), np.array(end)
        count = max(2, int(np.ceil(np.linalg.norm(end-start)/(a/nmesh))))
        coords.extend(start+(end-start)*j/count for j in range(count))
        tags.extend([names[i]]*count)
    coords.append(np.array(points[-1]))
    n = len(coords)
    nodes = np.zeros((n, 8))
    nodes[:, 0] = np.arange(n)
    nodes[:, 1:3] = coords
    nodes[:, 3:8] = 1
    elems = np.column_stack([np.arange(n-1), np.arange(n-1), np.arange(1,n),
                             np.full(n-1, thickness), np.full(n-1, 100)])
    return nodes, elems, tags, a

def participation(vec, nodes, tags):
    n = len(nodes)
    u, w = vec[:2*n:2], vec[2*n:4*n:2]
    displacement = np.column_stack([u, w])
    lengths = np.linalg.norm(np.diff(nodes[:, 1:3], axis=0), axis=1)
    # Exact integral of squared linearly interpolated translations.
    weights = lengths*(np.sum(displacement[:-1]**2,axis=1)
                       + np.sum(displacement[:-1]*displacement[1:],axis=1)
                       + np.sum(displacement[1:]**2,axis=1))/3
    total = weights.sum()
    parts = {name: float(weights[np.array(tags)==name].sum()/total) for name in set(tags)}
    parts['lips'] = sum(value for name,value in parts.items() if name.startswith('lip'))
    return parts, displacement

def solve(x, r, s, nmesh=12, nlength=36, at=100, upper=2.0, thickness=1, save=False):
    nodes, elems, tags, a = build(x,r,s,nmesh,at,thickness)
    lengths = a*np.geomspace(0.2,upper,nlength)
    signature, curves, shapes = solver.signature(nodes,elems,lengths)
    stress, length, index = solver.refined_minimum(lengths, signature)
    # Re-evaluate at the interpolated wavelength for an actual eigenpair.
    sig_at, eig_at, shapes_at = solver.signature(nodes,elems,np.array([length]))
    stress = float(sig_at[0])
    parts, disp = participation(shapes_at[0,0],nodes,tags)
    k = stress*12*(1-NU**2)*at**2/(np.pi**2*E)
    result = dict(x=x,r=r,s=s,at=at,thickness=thickness,nmesh=nmesh,nlength=nlength,
                  upper=upper,k=k,sigma=stress,half_wave=length/a,boundary=index in (0,nlength-1),
                  lips=parts['lips'],gap=eig_at[0,1]/eig_at[0,0]-1)
    result.update({f'part_{key}':parts.get(key,0.0) for key in sorted(solver.SEG_NAMES_COMPLEX)})
    if save:
        name=f'x{x:.2f}_r{r:.2f}_s{s:.2f}_m{nmesh}_u{upper:.1f}'
        np.savez_compressed(ROOT/'data'/f'{name}.npz',nodes=nodes,elements=elems,
                            lengths=lengths/a,curves=curves,displacements=disp)
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode', choices=['smoke','grid','extended','checks','return_scan'])
    args=parser.parse_args()
    if args.mode=='smoke':
        print(json.dumps(solve(1.3,0.3,0,save=True),indent=2))
        return
    if args.mode=='grid':
        cases=[dict(x=x,r=float(r),s=s) for s in [0,.25,.5] for x in [1.,1.3,1.6]
               for r in np.round(np.arange(.15,.601,.025),3)]
    elif args.mode=='extended':
        cases=[dict(x=x,r=float(r),s=s) for s in [0,.25,.5] for x in [1.1,1.2,1.4,1.5]
               for r in np.round(np.arange(.15,.601,.025),3)]
        cases += [dict(x=1.3,r=float(r),s=s) for s in [.05,.1,.15]
                  for r in np.round(np.arange(.15,.601,.025),3)]
    elif args.mode=='return_scan':
        cases=[dict(x=x,r=r,s=float(s)) for x in [1.,1.3,1.6] for r in [.4,.5,.6]
               for s in np.round(np.arange(.02,.121,.005),3)]
        cases += [dict(x=1.3,r=.6,s=s,nmesh=24,nlength=72,save=True) for s in [.05,.06,.07,.08,.1]]
    else:
        cases=[dict(x=x,r=r,s=s,nmesh=m,nlength=72,save=True)
               for x in [1.,1.6] for r in [.2,.35,.5,.6] for s in [0,.5] for m in [24]]
        cases += [dict(x=1.3,r=r,s=s,nmesh=24,nlength=100,upper=10,save=True)
                  for r in [.2,.4,.6] for s in [0,.5]]
        cases += [dict(x=1.3,r=.4,s=s,nmesh=24,nlength=72,at=at,thickness=t)
                  for s in [0,.5] for at,t in [(70,1),(100,1),(140,1),(100,2)]]
    path=ROOT/'data'/f'{args.mode}.csv'
    done=pd.read_csv(path) if path.exists() else pd.DataFrame()
    start=time.time()
    for i,case in enumerate(cases):
        if i<len(done):
            continue
        rec=solve(**case)
        pd.DataFrame([rec]).to_csv(path,mode='a',header=not path.exists(),index=False)
        print(f'{args.mode} {i+1}/{len(cases)} k={rec["k"]:.5f} lip={rec["lips"]:.3f} elapsed={time.time()-start:.1f}s',flush=True)

if __name__=='__main__':
    main()
