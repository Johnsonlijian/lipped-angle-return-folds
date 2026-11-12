"""Reproduce the three fine return-threshold checks without hidden commands."""
from refine_crossover import ROOT, crossover
import json

if __name__ == '__main__':
    rows=[]
    log=[]
    for x in [1., 1.3, 1.6]:
        row,steps=crossover(x,.6,96)
        rows.append(row);log.extend(steps)
        (ROOT/'data/crossover_fine.json').write_text(json.dumps(rows,indent=2))
        (ROOT/'data/crossover_fine_iterations.json').write_text(json.dumps(log,indent=2))
        print(row,flush=True)
