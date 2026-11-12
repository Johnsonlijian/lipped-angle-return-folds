"""R19b elastic regime map over lip ratio r=c/a and return ratio s=d/c (x=1.3, a/t=100).

For each (r, s) every interior first-eigenvalue minimum on 0.3a-10a is refined
and labelled by its dominant cFSM class (vector norm).  The governing minimum
is the lowest one.  Output: one JSON record per (r, s).
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r19env  # noqa: F401
import numpy as np  # noqa: E402
import research_scan as fsm  # noqa: E402
import r19b_lib as lb  # noqa: E402

OUT = lb.DATA / 'regime_map.json'
R_SET = np.round(np.arange(0.15, 0.601, 0.05), 3)
S_SET = np.round(np.arange(0.0, 0.2001, 0.02), 3)

if __name__ == '__main__':
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    t0 = time.time()
    for r in R_SET:
        for s in S_SET:
            key = f'r{r:.3f}_s{s:.3f}'
            if key in res:
                continue
            nodes, elems, tags, a = fsm.build(1.3, float(r), float(s), nmesh=24, at=100)
            tr, _ = lb.troughs(nodes, elems, tags, a, 100, lo=0.3, hi=10.0, n=80, norms=(1,))
            gov = min(tr, key=lambda t: t['k']) if tr else None
            res[key] = dict(x=1.3, r=float(r), s=float(s), troughs=tr, governing=gov,
                            n_minima=len(tr))
            OUT.write_text(json.dumps(res, indent=1))
            print(key, f'{time.time()-t0:.0f}s', len(tr),
                  None if gov is None else (round(gov['L_over_a'], 2), round(gov['k'], 3),
                                            gov['dominant'], round(gov['lips'], 2)), flush=True)
