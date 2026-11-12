"""R19b conventional lipped channel with a stockier web (h=2b, b/t=25, h/t=50).

At b/t=50 (channel_crossover.py, family C) web-local buckling governs from s=0,
so no return threshold exists.  Here the web is stockier, which raises the
local stress relative to the distortional one; the same upward s-scan and
Brent solve test whether a return-driven exchange then appears.
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r19env  # noqa: F401
import numpy as np  # noqa: E402
import r19b_lib as lb  # noqa: E402
from channel_crossover import build_std_channel, label_governing  # noqa: E402

OUT = lb.DATA / 'channel_crossover_c25.json'

if __name__ == '__main__':
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for r in (0.2, 0.3, 0.4, 0.6):
        key = f'C25_r{r}'
        if key in res:
            continue
        t0 = time.time()
        builder = lambda s, r=r: build_std_channel(r, float(s), bt=25.0)  # noqa: E731
        try:
            out, log = lb.crossover_scan(builder, 25.0, np.round(np.arange(0.0, 1.0001, 0.02), 3),
                                         lo=0.3, hi=20.0, n=80)
        except Exception as exc:
            out, log = dict(status='error', error=repr(exc)), []
        out['s0_troughs_labelled'] = label_governing(builder, 0.0, 25.0, 0.3, 20.0)
        if out.get('s_star') is not None:
            out['root_troughs_labelled'] = label_governing(builder, out['s_star'], 25.0, 0.3, 20.0)
            out['d_over_t'] = out['s_star'] * r * 25.0
        out.update(family='C25', r=r, at=25.0, elapsed_s=time.time() - t0, scan=log)
        res[key] = out
        OUT.write_text(json.dumps(res, indent=1))
        print(key, out['status'], out.get('s_star'), out.get('d_over_t'), out.get('first_local_s'),
              [(round(t['L_over_a'], 2), round(t['k'], 3), t['cls'], round(t['rho_L'], 2), round(t['lips'], 2))
               for t in out['s0_troughs_labelled']], f"{out['elapsed_s']:.0f}s", flush=True)
