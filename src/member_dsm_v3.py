"""R19b direct-strength screening with v3-identified elastic inputs (supersedes v1 and v2).

v1 (member_dsm.py): fixed-window grid minima (window-boundary values for s<=0.04, s>=0.12)
    and P_cre from the signature curve at half-wave L (a short-wave local/distortional
    branch at L=800 mm, 304 MPa, instead of the 970 MPa flexural-torsional load).
v2 (member_dsm_v2.py): labels and pure-D / pure-G inputs from unvalidated cFSM bases.
v3: P_crl  = lowest free minimum identified as local (r19b_lib.identify); if none, the
             pure-local cFSM minimum (local subspace validated);
    P_crd  = lowest free minimum identified as distortional; if none, no distortional
             check (the free spectrum has no distortional trough);
    P_cre  = classical Vlasov flexural-torsional load, pinned and warping-free, at L.
    Variant B takes P_crl from the pure-local minimum in every case.
AISI S100-16 Chapter E expressions (E2 global, E3.2.1 local, E4.1 distortional).
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r19env  # noqa: F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import r19b_lib as lb  # noqa: E402

FYS = (250.0, 350.0, 450.0)


def dsm(p_cre, p_crl, p_crd, p_y):
    lamc = np.sqrt(p_y / p_cre)
    p_ne = (0.658 ** (lamc ** 2)) * p_y if lamc <= 1.5 else (0.877 / lamc ** 2) * p_y
    laml = np.sqrt(p_ne / p_crl)
    p_nl = p_ne if laml <= 0.776 else (1 - 0.15 * (p_crl / p_ne) ** 0.4) * (p_crl / p_ne) ** 0.4 * p_ne
    if p_crd is None:
        lamd, p_nd = float('nan'), float('inf')
    else:
        lamd = np.sqrt(p_y / p_crd)
        p_nd = p_y if lamd <= 0.561 else (1 - 0.25 * (p_crd / p_y) ** 0.6) * (p_crd / p_y) ** 0.6 * p_y
    vals = (p_ne, p_nl, p_nd)
    return dict(p_ne=p_ne, p_nl=p_nl, p_nd=p_nd, p_n=min(vals), lamc=lamc, laml=laml, lamd=lamd,
                governing=('global', 'local', 'distortional')[int(np.argmin(vals))])


if __name__ == '__main__':
    el = json.loads((lb.DATA / 'dsm_v3_elastic.json').read_text())
    rows = []
    for key, e in el.items():
        A = e['area']
        locs = [t['sigma'] for t in e['troughs'] if t['cls'] == 'L']
        dists = [t['sigma'] for t in e['troughs'] if t['cls'] == 'D']
        sig_pl = e['pure_local']['sigma']
        for Ls, g in e['ft'].items():
            sig_e = g['FT']
            for fy in FYS:
                p_y = A * fy
                rec = dict(x=e['x'], r=e['r'], s=e['s'], L_mm=float(Ls), fy=fy, area=A, p_y=p_y,
                           sig_e=sig_e, sig_l_free=min(locs) if locs else None,
                           sig_d=min(dists) if dists else None, sig_l_pure=sig_pl,
                           src_l='free_min' if locs else 'pure_local_min')
                sig_l_A = min(locs) if locs else sig_pl
                for tag, sig_l in (('A', sig_l_A), ('B', sig_pl)):
                    res = dsm(sig_e * A, sig_l * A, None if not dists else min(dists) * A, p_y)
                    rec.update({f'{k}_{tag}': v for k, v in res.items()})
                rows.append(rec)
    df = pd.DataFrame(rows).sort_values(['x', 'r', 'L_mm', 'fy', 's'])
    df.to_csv(lb.DATA / 'dsm_v3_strength.csv', index=False)
    print('wrote', len(df), 'rows')
    t = df[(df.x == 1.3) & (df.r == 0.6)]
    for L in (800.0, 1200.0, 1800.0, 2400.0):
        for fy in FYS:
            u = t[(t.L_mm == L) & (t.fy == fy)]
            seq = ''.join({'local': 'L', 'distortional': 'D', 'global': 'G'}[g] for g in u.governing_A)
            pn0 = u.p_n_A.iloc[0]
            print(f'L={L:.0f} fy={fy:.0f}  {seq}  Pn(s=0)={pn0/1e3:.2f} kN  min={u.p_n_A.min()/1e3:.2f}  '
                  f'max={u.p_n_A.max()/1e3:.2f}  Pn(0.16)/Pn(0)={u.p_n_A.iloc[-1]/pn0:.3f}')
