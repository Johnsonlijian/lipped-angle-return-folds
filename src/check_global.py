"""R19b sanity check of the cFSM pure-global solution against classical column theory.

For x=1.3, r=0.6, s in {0, 0.01, 0.02, 0.08} at L=800/1200/1800/2400 mm compares:
  free first eigenvalue at half-wave L, pure-G cFSM (all 4 global vectors),
  and classical flexural / flexural-torsional buckling stresses from cutwp
  section properties (pinned-pinned, warping free, single half-wave).
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r19env  # noqa: F401
import numpy as np  # noqa: E402
import research_scan as fsm  # noqa: E402
import cfsm_tools as ct  # noqa: E402
import r19b_lib as lb  # noqa: E402

E, NU = fsm.E, fsm.NU
G = E / (2 * (1 + NU))


def classical(sp, L):
    """Pinned-pinned elastic global buckling stresses (flexural about principal axes, FT)."""
    A = float(sp['A'])
    I11, I22 = float(sp['I11']), float(sp['I22'])
    J, Cw = float(sp['J']), float(sp['Cw'])
    # shear centre relative to centroid, rotated into principal axes
    xc, yc = float(sp['cx']), float(sp['cy'])
    x0, y0 = float(sp['x0']) - xc, float(sp['y0']) - yc
    th = float(sp['phi'])
    c, s = np.cos(th), np.sin(th)
    u0, v0 = c * x0 + s * y0, -s * x0 + c * y0
    r0sq = (I11 + I22) / A + u0 ** 2 + v0 ** 2
    P1 = np.pi ** 2 * E * I11 / L ** 2
    P2 = np.pi ** 2 * E * I22 / L ** 2
    Pz = (G * J + np.pi ** 2 * E * Cw / L ** 2) / r0sq
    # cubic for general flexural-torsional buckling (Timoshenko & Gere form)
    coeffs = [r0sq - u0 ** 2 - v0 ** 2,
              -(r0sq * (P1 + P2 + Pz) - P1 * u0 ** 2 * 0 - P2 * v0 ** 2 * 0
                - P2 * u0 ** 2 - P1 * v0 ** 2),
              r0sq * (P1 * P2 + P1 * Pz + P2 * Pz) - P1 * P2 * 0
              - P1 * v0 ** 2 * Pz * 0 - P1 * P2 * (u0 ** 2 + v0 ** 2) * 0,
              -r0sq * P1 * P2 * Pz]
    roots = np.roots(coeffs)
    roots = np.sort(roots[np.isreal(roots)].real)
    roots = roots[roots > 0]
    return dict(A=A, P1=P1 / A, P2=P2 / A, Pz=Pz / A, FT_min=(roots[0] / A if roots.size else None),
                u0=u0, v0=v0, J=J, Cw=Cw)


if __name__ == '__main__':
    out = {}
    for s in (0.0, 0.01, 0.02, 0.08):
        nodes, elems, tags, a = fsm.build(1.3, 0.6, s, nmesh=24, at=100)
        sp = ct.sect_props(nodes, elems)
        counts = ct.class_counts(nodes, elems)
        for L in (800.0, 1200.0, 1800.0, 2400.0):
            sig_f, phi = lb.eig1(nodes, elems, L)
            lab = lb.label(nodes, elems, L, phi)
            ev = ct.pure_class_eig(nodes, elems, L, 'glob', counts=counts)[0]
            cl = classical(sp, L)
            key = f's{s}_L{int(L)}'
            out[key] = dict(free=sig_f, free_label=lab, pureG=[float(v) for v in ev], classical=cl)
            print(key, f'free={sig_f:8.2f} ({lab["dominant"]}, G={lab["G"]:.0f}%)',
                  'pureG', np.round(ev, 1), 'classical P1,P2,Pz,FT =',
                  round(cl['P1'], 1), round(cl['P2'], 1), round(cl['Pz'], 1),
                  None if cl['FT_min'] is None else round(cl['FT_min'], 1), flush=True)
        print('   props: J', round(float(sp['J']), 2), 'Cw', f"{float(sp['Cw']):.4g}",
              'keys', sorted(sp.keys()) if s == 0 else '', flush=True)
    (lb.DATA / 'check_global.json').write_text(json.dumps(out, indent=1, default=float))
