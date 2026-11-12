import os
"""Independent Abaqus shell meshes: no finite-strip assembly or eigenvectors used."""
from pathlib import Path
import argparse
import json
import math
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def make(name, x=1.3, r=.6, s=0., length=1.3, mesh=20, element='S4', plate=False):
    a, t = 100., 1.
    b, c, d, L = a/x, a*r, a*r*s, a*length
    if plate:
        points, names = [(0.,0.),(a,0.)], ['plate']
    elif s:
        points = [(c,b-d),(c,b),(0,b),(0,0),(a,0),(a,c),(a-d,c)]
        names = ['lip_b_d','lip_b','leg_b','leg_a','lip_a','lip_a_d']
    else:
        points = [(c,b),(0,b),(0,0),(a,0),(a,c)]
        names = ['lip_b','leg_b','leg_a','lip_a']
    xy, tags = [], []
    for p,q,tag in zip(points[:-1],points[1:],names):
        n = max(2,math.ceil(math.dist(p,q)/(a/mesh)))
        for j in range(n):
            xy.append([p[k]+(q[k]-p[k])*j/n for k in range(2)])
            tags.append(tag)
    xy.append(list(points[-1]))
    nc = len(xy)
    nz = 2*math.ceil(L/(a/mesh)/2)
    node = lambda iz,ix: iz*nc+ix+1
    lines = ['*Heading',name,'*Node']
    for iz in range(nz+1):
        lines.extend('%d, %.12g, %.12g, %.12g'%(node(iz,j),p[0],p[1],L*iz/nz) for j,p in enumerate(xy))
    lines += ['*Element, type='+element+', elset=ALL']
    for iz in range(nz):
        for j in range(nc-1):
            lines.append('%d, %d, %d, %d, %d'%(iz*(nc-1)+j+1,node(iz,j),node(iz,j+1),node(iz+1,j+1),node(iz+1,j)))
    lines += ['*Nset, nset=ENDS']
    labels = [node(iz,j) for iz in [0,nz] for j in range(nc)]
    lines += [', '.join(map(str,labels[i:i+12])) for i in range(0,len(labels),12)]
    if plate:
        lines += ['*Nset, nset=SIDES']
        labels=[node(iz,j) for iz in range(nz+1) for j in [0,nc-1]]
        lines += [', '.join(map(str,labels[i:i+12])) for i in range(0,len(labels),12)]
    lines += ['*Material, name=STEEL','*Elastic','210000., 0.3','*Shell Section, elset=ALL, material=STEEL','1., 5']
    origin = min(range(nc),key=lambda i: xy[i][0]**2+xy[i][1]**2)
    along = min(range(nc),key=lambda i: (xy[i][0]-a)**2+xy[i][1]**2)
    lines += ['*Step, name=BUCKLE, perturbation','*Buckle, eigensolver=Lanczos','12, 0., , ,',
              '*Boundary, load case=1',f'{node(0,origin)}, 3, 3','ENDS, 6, 6']
    # Prescribe the exact Poisson expansion under the unit-stress live load.
    # The same DOFs become homogeneous constraints for the buckling modes.
    for iz in [0,nz]:
        for j,p in enumerate(xy):
            for dof in [1,2]:
                lines.append('%d, %d, %d, %.12g'%(node(iz,j),dof,dof,.3*p[dof-1]/210000.))
    if plate:lines += ['SIDES, 2, 2','SIDES, 4, 4']
    lines += ['*Boundary, op=NEW, load case=2','ENDS, 1, 2','ENDS, 6, 6',f'{node(0,origin)}, 3, 3']
    if plate:
        lines += ['SIDES, 2, 2','SIDES, 4, 4']
    widths=[0.]*nc
    for j in range(nc-1):
        h=math.dist(xy[j],xy[j+1])/2
        widths[j]+=h; widths[j+1]+=h
    lines += ['*Cload']
    for iz,sign in [(0,1),(nz,-1)]:
        lines += ['%d, 3, %.12g'%(node(iz,j),sign*w*t) for j,w in enumerate(widths)]
    lines += ['*Output, field','*Node Output','U,','*Element Output','S,','*End Step']
    folder=ROOT/'shell'/name
    folder.mkdir(exist_ok=True)
    (folder/(name+'.inp')).write_text('\n'.join(lines)+'\n')
    meta=dict(name=name,x=x,r=r,s=s,length=length,mesh=mesh,element=element,plate=plate,
              xy=xy,tags=tags,ncross=nc,nz=nz,nodes=(nz+1)*nc,elements=nz*(nc-1),end_force=sum(widths)*t,bc_version='affine_poisson_v2')
    (folder/'model.json').write_text(json.dumps(meta,indent=2))
    return folder

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--x',type=float,default=1.3);p.add_argument('--r',type=float,default=.6);p.add_argument('--s',type=float,default=0);p.add_argument('--length',type=float,default=1.3);p.add_argument('--mesh',type=int,default=20);p.add_argument('--element',default='S4');p.add_argument('--plate',action='store_true');p.add_argument('--run',action='store_true')
    args=vars(p.parse_args());run=args.pop('run');folder=make(**args)
    if run:
        cmd=['cmd','/c',os.environ.get('ABAQUS_COMMAND', 'abaqus'),'job='+args['name'],'interactive','cpus=2','ask_delete=OFF']
        with (folder/'console.log').open('w') as log:
            result=subprocess.run(cmd,cwd=folder,stdout=log,stderr=subprocess.STDOUT)
        print(folder,result.returncode,flush=True)
