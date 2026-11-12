# Compatible with Abaqus 2020 Python 2.7.
from odbAccess import openOdb
import json, sys, os
path=sys.argv[-1]
odb=openOdb(path,readOnly=True)
frames=[]
for frame in odb.steps[list(odb.steps.keys())[-1]].frames:
    row={'description':frame.description,'mode':frame.mode,'frameValue':frame.frameValue}
    if 'U' in frame.fieldOutputs:
        row['displacement']=[[int(v.nodeLabel)]+[float(x) for x in v.data] for v in frame.fieldOutputs['U'].values]
    if 'S' in frame.fieldOutputs and 'EigenValue' not in frame.description:
        row['stress']=[[float(x) for x in v.data] for v in frame.fieldOutputs['S'].values]
    frames.append(row)
with open(os.path.splitext(path)[0]+'_results.json','w') as f:json.dump(frames,f)
odb.close()
print('Extracted '+path)
