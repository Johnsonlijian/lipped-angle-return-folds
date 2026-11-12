import os
"""Independent shell checks at the eight endpoints defining the ratio trend.

Lengths are selected from the archived FSM grid; shell stiffness, load and
meshes are generated independently. This is a cross-solver check at fixed
lengths, not independent optimisation of each shell signature curve.
"""
from shell_models import ROOT, make
import csv
import subprocess

source = ROOT.parent / 'R17_independent_rebuild_20261109/data/combined_scan.csv'
with source.open() as stream:
    rows = list(csv.DictReader(stream))
for x in [1.1, 1.5]:
    for r in [.2, .6]:
        for s in [0., .5]:
            row = next(v for v in rows if all(abs(float(v[k])-n)<1e-8 for k,n in [('x',x),('r',r),('s',s)]))
            name = f'v_transfer_x{x:.1f}_r{r:.1f}_s{s:.1f}_m48'
            folder = make(name, x=x, r=r, s=s, length=float(row['half_wave']), mesh=48)
            inp = folder/(name+'.inp')
            inp.write_text(inp.read_text().replace('*Element Output\nS,\n',''))
            if not (folder/(name+'_results.json')).exists():
                with (folder/'console.log').open('w') as log:
                    subprocess.run(['cmd','/c',os.environ.get('ABAQUS_COMMAND', 'abaqus'),'job='+name,'interactive','cpus=4','ask_delete=OFF'],cwd=folder,stdout=log,stderr=subprocess.STDOUT,check=True)
                if 'COMPLETED' not in (folder/'console.log').read_text():
                    raise RuntimeError(name)
                subprocess.run(['cmd','/c',os.environ.get('ABAQUS_COMMAND', 'abaqus'),'python',str(ROOT/'src/extract_odb.py'),name+'.odb'],cwd=folder,check=True)
            print(name, flush=True)
