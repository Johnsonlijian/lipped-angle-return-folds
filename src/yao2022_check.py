"""R19b external mode-identity check: double-lipped equal-leg angles of Yao et al. (2022).

Nominal dimensions (Buildings 12:1775, Table 1): legs a = 60/90/120 mm, first lips
c = 20/20/24 mm, second lips (returns) d = 10 mm, t = 2 mm (DLA9020 uses the
measured equal legs of Table 2, ~91.7 mm, not the conflicting nominal 60 mm).
Two readings bound the outer/centreline ambiguity: 'nominal' treats the table
values as centreline widths; 'centreline' subtracts t at each fold (a-t, c-t,
d-t/2).  For each section: free troughs with cFSM labels at the tested return,
and the s-scan crossover for the same a/t and c/a.
Observed modes (Table 3): DLA6020-400 D; DLA9020-400 L+D; DLA12024-400 D+F+L.
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

OUT = lb.DATA / 'yao2022_check.json'
SECTIONS = {'DLA6020': (60.0, 20.0, 10.0), 'DLA9020': (91.7, 20.0, 10.0),
            'DLA12024': (120.0, 24.0, 10.0)}
T = 2.0


def reading(a, c, d, mode):
    if mode == 'centreline':
        a, c, d = a - T, c - T, d - T / 2
    return dict(at=a / T, r=c / a, s=d / c, d_over_t=d / T)


if __name__ == '__main__':
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, (a, c, d) in SECTIONS.items():
        for mode in ('nominal', 'centreline'):
            key = f'{name}_{mode}'
            if key in res:
                continue
            t0 = time.time()
            g = reading(a, c, d, mode)
            nodes, elems, tags, aa = fsm.build(1.0, g['r'], g['s'], nmesh=24, at=g['at'])
            tr, _ = lb.troughs_id(nodes, elems, tags, aa, g['at'], lo=0.3, hi=12.0, n=80)
            builder = lambda s, g=g: fsm.build(1.0, g['r'], float(s), nmesh=24, at=g['at'])  # noqa: E731
            try:
                cross, log = lb.crossover_scan(builder, g['at'], np.round(np.arange(0.0, 1.0001, 0.02), 3),
                                               lo=0.3, hi=12.0, n=80)
            except Exception as exc:
                cross, log = dict(status='error', error=repr(exc)), []
            if cross.get('s_star') is not None:
                cross['d_over_t'] = cross['s_star'] * g['r'] * g['at']
            res[key] = dict(section=name, reading=mode, **g, troughs_at_test=tr,
                            crossover=cross, scan=log, elapsed_s=time.time() - t0)
            OUT.write_text(json.dumps(res, indent=1))
            print(key, {k: round(v, 3) for k, v in g.items()},
                  [(round(t['L_over_a'], 2), round(t['k'], 3), t['cls'], round(t['rho_L'], 2),
                    round(t['rho_FT'], 1), round(t['lips'], 2)) for t in tr],
                  cross.get('status'), cross.get('s_star'), cross.get('d_over_t'),
                  f'{time.time()-t0:.0f}s', flush=True)
