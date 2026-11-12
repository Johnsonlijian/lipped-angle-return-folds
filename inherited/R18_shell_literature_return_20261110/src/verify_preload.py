import os
from shell_models import ROOT,make
import subprocess,re,json
import numpy as np
rows=[]
for s in [0,.08,.5]:
    name='preload_affine_s'+str(int(s*100))
    folder=make(name,s=s,length=2.72,mesh=12)
    inp=folder/(name+'.inp');txt=inp.read_text()
    txt=txt.replace('*Step, name=BUCKLE, perturbation\n*Buckle, eigensolver=Lanczos\n12, 0., , ,','*Step, name=STATIC\n*Static\n1., 1.')
    txt=txt.replace('*Boundary, load case=1','*Boundary')
    start=txt.index('*Boundary, op=NEW');end=txt.index('*Cload',start)
    txt=txt[:start]+txt[end:]
    inp.write_text(txt)
    with (folder/'console.log').open('w') as log:
        subprocess.run(['cmd','/c',os.environ.get('ABAQUS_COMMAND', 'abaqus'),'job='+name,'interactive','cpus=2','ask_delete=OFF'],cwd=folder,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run(['cmd','/c',os.environ.get('ABAQUS_COMMAND', 'abaqus'),'python',str(ROOT/'src'/'extract_odb.py'),name+'.odb'],cwd=folder,check=True)
    data=json.loads((folder/(name+'_results.json')).read_text())
    stress=np.array(data[-1]['stress'])
    center=(stress[:,0]+stress[:,1])/2
    radius=np.sqrt(((stress[:,0]-stress[:,1])/2)**2+stress[:,3]**2)
    row=dict(s=s,min_principal_min=float((center-radius).min()),min_principal_max=float((center-radius).max()),max_principal_abs=float(np.abs(center+radius).max()),samples=len(stress))
    rows.append(row);print(row,flush=True)
(ROOT/'data'/'preload_checks.json').write_text(json.dumps(rows,indent=2))
