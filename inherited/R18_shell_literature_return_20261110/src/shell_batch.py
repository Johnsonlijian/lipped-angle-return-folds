import os
from shell_models import make,ROOT
import subprocess, json
cases=[dict(name='plate_m'+str(m),plate=True,length=1,mesh=m) for m in [12,24,48]]
for label,x,r,s,L in [('simple',1.3,.6,0,1.36),('return06',1.3,.6,.06,2.25),('return08lip',1.3,.6,.08,2.72),('return08leg',1.3,.6,.08,.847),('return10lip',1.3,.6,.10,3.13),('return10leg',1.3,.6,.10,.837),('return50',1.3,.6,.50,.83)]:
    for mesh in [12,24,48]:
        cases.append(dict(name=label+'_m'+str(mesh),x=x,r=r,s=s,length=L,mesh=mesh))
for case in cases:
    case['name']='v_'+case['name']
    folder=make(**case);name=case['name']
    if not (folder/(name+'_results.json')).exists():
        with (folder/'console.log').open('w') as log:
            subprocess.run(['cmd','/c',os.environ.get('ABAQUS_COMMAND', 'abaqus'),'job='+name,'interactive','cpus=2','ask_delete=OFF'],cwd=folder,stdout=log,stderr=subprocess.STDOUT,check=True)
        if 'COMPLETED' not in (folder/'console.log').read_text():raise RuntimeError(name)
        subprocess.run(['cmd','/c',os.environ.get('ABAQUS_COMMAND', 'abaqus'),'python',str(ROOT/'src'/'extract_odb.py'),name+'.odb'],cwd=folder,check=True)
    print(name,'complete',flush=True)
