"""Verification of the cutwp.prop2 warping fix (R19b).

1. Plain channel h=100, b=50, t=1 (centreline): closed form
   Cw = t b^3 h^2 (3b + 2h) / (12 (6b + h)),  e = 3 b^2 / (6b + h) from the web.
2. Plain equal angle: Cw -> 0 (all walls radiate from the shear centre).
3. Lipped angle x=1.3, r=0.6, s=0: pure-G cFSM vs classical pinned
   flexural-torsional buckling with the corrected section properties.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r19env  # noqa: F401
import numpy as np  # noqa: E402
import research_scan as fsm  # noqa: E402
import cfsm_tools as ct  # noqa: E402
import r19b_lib as lb  # noqa: E402
from check_global import classical  # noqa: E402


def poly(points, n=10, t=1.0):
    P = [np.array(p, float) for p in points]
    coords = []
    for p0, p1 in zip(P[:-1], P[1:]):
        coords += [p0 + (p1 - p0) * j / n for j in range(n)]
    coords.append(P[-1])
    k = len(coords)
    nodes = np.zeros((k, 8))
    nodes[:, 0] = np.arange(k)
    nodes[:, 1:3] = coords
    nodes[:, 3:8] = 1
    elems = np.column_stack([np.arange(k - 1), np.arange(k - 1), np.arange(1, k),
                             np.full(k - 1, t), np.full(k - 1, 100)])
    return nodes, elems


if __name__ == '__main__':
    h, b, t = 100.0, 50.0, 1.0
    nodes, elems = poly([(b, 0), (0, 0), (0, h), (b, h)])
    sp = ct.sect_props(nodes, elems)
    cw_exact = t * b ** 3 * h ** 2 * (3 * b + 2 * h) / (12 * (6 * b + h))
    e_exact = 3 * b ** 2 / (6 * b + h)
    print(f"channel Cw = {float(sp['Cw']):.6g} (exact {cw_exact:.6g}); shear-centre x0 = {float(sp['x0']):.4f} "
          f"(exact {-e_exact:.4f}); y0 = {float(sp['y0']):.4f} (exact {h/2})")
    nodes, elems = poly([(0, 60), (0, 0), (60, 0)])
    sp = ct.sect_props(nodes, elems)
    print(f"plain angle Cw = {float(sp['Cw']):.4g}; shear centre ({float(sp['x0']):.3g}, {float(sp['y0']):.3g})")
    for s in (0.0, 0.08):
        nodes, elems, tags, a = fsm.build(1.3, 0.6, s, nmesh=24, at=100)
        sp = ct.sect_props(nodes, elems)
        counts = ct.class_counts(nodes, elems)
        b_v, cc = ct.gdlo_basis(nodes, elems, 800.0)
        print(f"  basis rank {np.linalg.matrix_rank(b_v)} of {b_v.shape[0]} (G,D,L = {cc})")
        print(f"\nlipped angle s={s}: Cw = {float(sp['Cw']):.5g}, J = {float(sp['J']):.4g}, "
              f"x0,y0 = ({float(sp['x0']):.3f}, {float(sp['y0']):.3f})")
        for L in (800.0, 1200.0, 1800.0, 2400.0, 6000.0):
            ev = ct.pure_class_eig(nodes, elems, L, 'glob', counts=counts)[0]
            sig_f, phi = lb.eig1(nodes, elems, L)
            lab = lb.label(nodes, elems, L, phi)
            cl = classical(sp, L)
            print(f"  L={L:6.0f}: pure-G {np.round(ev[:3], 1)}  classical P1={cl['P1']:.1f} P2={cl['P2']:.1f} "
                  f"Pz={cl['Pz']:.1f} FT={cl['FT_min']:.1f} | free {sig_f:.1f} "
                  f"(G {lab['G']:.0f} D {lab['D']:.0f} L {lab['L']:.0f} O {lab['O']:.0f})")
