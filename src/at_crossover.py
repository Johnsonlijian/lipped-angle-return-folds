"""R19b thickness dependence of the return-fold crossover (supersedes at_robustness.py).

at_robustness.py probed only s=0.06 and 0.08, returned an *upper bound* at
a/t=200 through an exception fallback, and failed (empty log -> IndexError)
for both a/t=100 cases.  Here each (r, a/t) is scanned upward in s on a wide
wavelength probe (0.3a-12a); the first sign change of k_lip - k_leg between
two coexisting stationary troughs is bracketed and solved by Brent.  The full
trough log is kept for plotting.
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

OUT = lb.DATA / next((a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--out=')),
                     'at_crossover.json')
CASES = [(r, at) for r in (0.6, 0.5) for at in (100, 200, 150, 75, 50)]


def s_grid(at):
    return (np.round(np.arange(0.0, 0.5001, 0.01), 3) if at <= 75
            else np.round(np.arange(0.0, 0.3001, 0.005), 4))


if __name__ == '__main__':
    only = [a for a in sys.argv[1:] if a.startswith('r')]
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for r, at in CASES:
        key = f'x1.3_r{r}_at{at}'
        if key in res or (only and key.split('_', 1)[1] not in only):
            continue
        t0 = time.time()
        builder = lambda s, r=r, at=at: fsm.build(1.3, r, float(s), nmesh=24, at=at)  # noqa: E731
        log = []
        try:
            out, log = lb.crossover_scan(builder, at, s_grid(at), lo=0.3, hi=12.0, n=80, log=log)
        except Exception as exc:  # keep the partial log; never silently substitute a bound
            out = dict(status='error', error=repr(exc))
        if out.get('s_star') is not None:
            out['d_over_t'] = out['s_star'] * r * at
        out.update(x=1.3, r=r, at=at, mesh=24, elapsed_s=time.time() - t0, scan=log)
        res[key] = out
        OUT.write_text(json.dumps(res, indent=1))
        print(key, out['status'], out.get('s_star'), out.get('d_over_t'),
              out.get('first_local_s'), f"{out['elapsed_s']:.0f}s", flush=True)
