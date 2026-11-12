import os
from shell_models import ROOT,make
import subprocess
cases=[]
for label,s,L in [('simple',0,1.36),('return06',.06,2.25),('return50',.5,.83)]:
    cases.append(dict(name='v_'+label+'_m96',s=s,length=L,mesh=96))
for s in [.086,.088]:
    for branch,L in [('lip',2.875),('leg',.842)]:
        cases.append(dict(name=f'v_cross{int(s*1000)}_{branch}_m48',s=s,length=L,mesh=48))
for case in cases:
    folder=make(**case);name=case['name'];inp=folder/(name+'.inp')
    inp.write_text(inp.read_text().replace('*Element Output\nS,\n',''))
    if not (folder/(name+'_results.json')).exists():
        with (folder/'console.log').open('w') as log:
            subprocess.run(['cmd','/c',os.environ.get('ABAQUS_COMMAND', 'abaqus'),'job='+name,'interactive','cpus=4','ask_delete=OFF'],cwd=folder,stdout=log,stderr=subprocess.STDOUT,check=True)
        if 'COMPLETED' not in (folder/'console.log').read_text():raise RuntimeError(name)
        subprocess.run(['cmd','/c',os.environ.get('ABAQUS_COMMAND', 'abaqus'),'python',str(ROOT/'src'/'extract_odb.py'),name+'.odb'],cwd=folder,check=True)
    print(name,flush=True)
