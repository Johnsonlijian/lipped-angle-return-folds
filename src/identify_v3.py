"""R19b mode identification v3 (local-subspace test + classical global test + fold kinematics).

Replaces the G/D/L/O participations of cfsm_identify.py: after repairing four porting
errors in the vendored cFSM (cutwp warping, y_dofs warping flags, other-mode indices,
classify), the global subspace still fails the classical check (pure-G 25.1 vs 19.4 MPa
at L = 60a), so global/distortional participations are not trusted.  The local subspace
is trusted because it reproduces the free local minima (to 0.01-1 %) and excludes
non-local modes; the classical Vlasov flexural-torsional solution is trusted because it
equals the free FSM at long half-waves.
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r19env  # noqa: F401
import research_scan as fsm  # noqa: E402
import cfsm_tools as ct  # noqa: E402
import r19b_lib as lb  # noqa: E402
import r19lib  # noqa: E402

OUT = lb.DATA / 'identify_v3.json'
CASES = ([(1.3, 0.6, s) for s in (0.0, 0.02, 0.04, 0.06, 0.08, 0.0868, 0.10, 0.12, 0.15, 0.25, 0.5)]
         + [(1.3, r, 0.0) for r in (0.2, 0.3, 0.4, 0.5)] + [(1.3, 0.5, 0.076)])

if __name__ == '__main__':
    res = {}
    kf = r19lib.kfactor(100)
    for x, r, s in CASES:
        t0 = time.time()
        nodes, elems, tags, a = fsm.build(x, r, s, nmesh=24, at=100)
        tr, _ = lb.troughs_id(nodes, elems, tags, a, 100, lo=0.3, hi=10.0, n=80)
        key = f'x{x}_r{r}_s{s}'
        res[key] = dict(x=x, r=r, s=s, troughs=tr)
        print(key, f'{time.time()-t0:.0f}s',
              [(round(t['L_over_a'], 3), round(t['k'], 3), round(t['lips'], 3), t['cls'],
                round(t['rho_L'], 3), round(t['rho_FT'], 1), round(t['internal_fold_max'], 2),
                round(t['rigid_fraction'], 2)) for t in tr], flush=True)
    # global-mode sanity check at long half-wavelength
    for s in (0.0, 0.08):
        nodes, elems, tags, a = fsm.build(1.3, 0.6, s, nmesh=24, at=100)
        for La in (24.0, 60.0):
            sig, phi = lb.eig1(nodes, elems, La * a)
            idn = lb.identify(nodes, elems, a, La, sig, phi)
            res[f'global_check_s{s}_L{La}'] = dict(sigma_free=sig, **idn)
            print('global check', s, La, round(sig, 2), idn['cls'], round(idn['rho_FT'], 3),
                  round(idn['rho_L'], 1), round(idn['rigid_fraction'], 3), flush=True)
    OUT.write_text(json.dumps(res, indent=1))
