"""R19b member-level direct-strength screening with cFSM-identified elastic inputs.

Supersedes member_dsm.py, whose P_crl / P_crd were grid minima inside fixed
windows (0.65-1.1a, 2.0-3.5a): for s<=0.04 and s>=0.12 these were window-boundary
values on the flank of another mode, not stationary troughs.

Elastic inputs per geometry (a=100 mm, t=1 mm, a/t=100):
  * free stationary minima on 0.3a-10a, labelled by dominant cFSM class (vector norm);
  * pure-local and pure-distortional cFSM minima over wavelength;
  * at each member length L: free first eigenvalue at half-wave L (+ label) and
    pure-global cFSM eigenvalue.
Method A (free FSM, cFSM-identified): P_crl / P_crd = lowest free minimum whose
  dominant class is L / D, otherwise the pure-class minimum; P_cre = free value at
  L if G-dominant, otherwise pure-G.
Method B (pure-mode cFSM): pure-L min, pure-D min, pure-G at L.
DSM: AISI S100-16 E2 (global), E3.2.1 (local), E4.1 (distortional).
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r19env  # noqa: F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import research_scan as fsm  # noqa: E402
import r19lib  # noqa: E402
import cfsm_tools as ct  # noqa: E402
import r19b_lib as lb  # noqa: E402

LENGTHS = (800.0, 1200.0, 1800.0, 2400.0)
FYS = (250.0, 350.0, 450.0)
S_SET = np.round(np.arange(0.0, 0.1601, 0.01), 3)
GEOMS = [(x, 0.6) for x in (1.3, 1.0, 1.6)] + [(1.3, 0.5)]
ELASTIC = lb.DATA / 'dsm_v2_elastic.json'
OUT = lb.DATA / 'dsm_v2_strength.csv'


def dsm(p_cre, p_crl, p_crd, p_y):
    lamc = np.sqrt(p_y / p_cre)
    p_ne = (0.658 ** (lamc ** 2)) * p_y if lamc <= 1.5 else (0.877 / lamc ** 2) * p_y
    laml = np.sqrt(p_ne / p_crl)
    p_nl = p_ne if laml <= 0.776 else (1 - 0.15 * (p_crl / p_ne) ** 0.4) * (p_crl / p_ne) ** 0.4 * p_ne
    lamd = np.sqrt(p_y / p_crd)
    p_nd = p_y if lamd <= 0.561 else (1 - 0.25 * (p_crd / p_y) ** 0.6) * (p_crd / p_y) ** 0.6 * p_y
    vals = (p_ne, p_nl, p_nd)
    return dict(p_ne=p_ne, p_nl=p_nl, p_nd=p_nd, p_n=min(vals), lamc=lamc, laml=laml,
                lamd=lamd, governing=('global', 'local', 'distortional')[int(np.argmin(vals))])


def elastic_inputs(x, r, s, at=100, mesh=24):
    nodes, elems, tags, a = fsm.build(x, r, s, nmesh=mesh, at=at)
    counts = ct.class_counts(nodes, elems)
    tr, _ = lb.troughs(nodes, elems, tags, a, at, lo=0.3, hi=10.0, n=80, norms=(1,))
    pl = lb.pure_min(nodes, elems, a, 'local', 0.3, 3.0, counts=counts)
    pdm = lb.pure_min(nodes, elems, a, 'dist', 0.8, 12.0, counts=counts)
    glob = {}
    for L in LENGTHS:
        sig_f, phi = lb.eig1(nodes, elems, L)
        lab = lb.label(nodes, elems, L, phi)
        try:
            ev = ct.pure_class_eig(nodes, elems, L, 'glob', counts=counts)[0]
        except Exception:
            ev = []
        glob[str(int(L))] = dict(free=sig_f, free_dominant=lab['dominant'], G_free=lab['G'],
                                 pure=float(ev[0]) if len(ev) else None)
    return dict(x=x, r=r, s=float(s), area=r19lib.section_area_mm2(nodes, 1.0), troughs=tr,
                pure_local=dict(L_over_a=pl[0], sigma=pl[1], boundary=pl[2]),
                pure_dist=dict(L_over_a=pdm[0], sigma=pdm[1], boundary=pdm[2]), glob=glob)


def _ok(v, big=1e9):
    """Missing or non-finite pure-mode value: mode does not exist -> cannot govern."""
    return big if v is None or not np.isfinite(v) else v


def method_inputs(e, L):
    def lowest(cls):
        c = [t['sigma'] for t in e['troughs'] if t['dominant'] == cls]
        return min(c) if c else None
    g = e['glob'][str(int(L))]
    sl, sd = lowest('L'), lowest('D')
    A = dict(sig_l=sl if sl is not None else _ok(e['pure_local']['sigma']),
             src_l='free_min' if sl is not None else 'pure_min',
             sig_d=sd if sd is not None else _ok(e['pure_dist']['sigma']),
             src_d='free_min' if sd is not None else 'pure_min',
             sig_e=g['free'] if g['free_dominant'] == 'G' else _ok(g['pure']),
             src_e='free_at_L' if g['free_dominant'] == 'G' else 'pure_G')
    B = dict(sig_l=_ok(e['pure_local']['sigma']), sig_d=_ok(e['pure_dist']['sigma']),
             sig_e=_ok(g['pure']))
    return A, B


if __name__ == '__main__':
    el = json.loads(ELASTIC.read_text()) if ELASTIC.exists() else {}
    t0 = time.time()
    for x, r in GEOMS:
        for s in S_SET:
            key = f'x{x}_r{r}_s{s:.3f}'
            if key in el:
                continue
            el[key] = elastic_inputs(x, r, s)
            ELASTIC.write_text(json.dumps(el, indent=1))
            e = el[key]
            print(key, f'{time.time()-t0:.0f}s',
                  [(round(t['L_over_a'], 2), round(t['sigma'] * r19lib.kfactor(100), 3), t['dominant'])
                   for t in e['troughs']],
                  'pureL', round(e['pure_local']['sigma'] * r19lib.kfactor(100), 3),
                  'pureD', round(e['pure_dist']['sigma'] * r19lib.kfactor(100), 3), flush=True)
    rows = []
    for key, e in el.items():
        for L in LENGTHS:
            A, B = method_inputs(e, L)
            for fy in FYS:
                p_y = e['area'] * fy
                rec = dict(x=e['x'], r=e['r'], s=e['s'], L_mm=L, fy=fy, area=e['area'], p_y=p_y,
                           src_l=A['src_l'], src_d=A['src_d'], src_e=A['src_e'])
                for tag, m in (('A', A), ('B', B)):
                    res = dsm(m['sig_e'] * e['area'], m['sig_l'] * e['area'], m['sig_d'] * e['area'], p_y)
                    rec.update({f'{k}_{tag}': v for k, v in res.items()})
                    rec.update({f'sig_{c}_{tag}': m[f'sig_{c}'] for c in 'lde'})
                rows.append(rec)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print('wrote', len(rows), 'rows', flush=True)
