"""R19 second-family check: unequal-flange lipped channel.

Section traversal (centreline, sharp corners, web h = a):
  (a-d, c) -> (a, c) -> (a, 0) -> (0, 0) -> (0, a) -> (b, a) -> (b, a-c) -> (b-d, a-c)
with x = a/b (flange ratio), r = c/a, s = d/c. For s=0 the two return
segments collapse to lip tips (a,c) and (b,a-c).

Produces the same endpoint-ratio drift R(r,s) and lip participation used for
the angle family, plus a compact 5x x 19r response grid per s for the
rank-one separability score.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r19env  # noqa: F401
import r19lib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import research_scan as fsm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data'

SEG_COMPLEX = ['lip_a_d', 'lip_a', 'leg_a', 'leg_web', 'leg_b', 'lip_b', 'lip_b_d']
SEG_SIMPLE = ['lip_a', 'leg_a', 'leg_web', 'leg_b', 'lip_b']


def channel_polyline(a, b, c, d, is_complex):
    if is_complex:
        return [(a - d, c), (a, c), (a, 0.0), (0.0, 0.0), (0.0, a),
                (b, a), (b, a - c), (b - d, a - c)]
    return [(a, c), (a, 0.0), (0.0, 0.0), (0.0, a), (b, a), (b, a - c)]


def build_channel(x, r, s, nmesh=24, at=100, t=1.0):
    a = at * t
    b, c = a / x, a * r
    d = s * c
    P = [np.array(p, dtype=float) for p in channel_polyline(a, b, c, d, s > 0)]
    names = SEG_COMPLEX if s > 0 else SEG_SIMPLE
    coords, tags = [], []
    for i, (p0, p1) in enumerate(zip(P[:-1], P[1:])):
        n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / (a / nmesh))))
        for j in range(n):
            coords.append(p0 + (p1 - p0) * j / n)
            tags.append(names[i])
    coords.append(P[-1])
    n = len(coords)
    nodes = np.zeros((n, 8))
    nodes[:, 0] = np.arange(n)
    nodes[:, 1:3] = coords
    nodes[:, 3:8] = 1
    elems = np.column_stack([np.arange(n - 1), np.arange(n - 1), np.arange(1, n),
                             np.full(n - 1, t), np.full(n - 1, 100)])
    return nodes, elems, tags, a


def solve_channel(x, r, s, nmesh=16, nlength=30, upper=2.0):
    nodes, elems, tags, a = build_channel(x, r, s, nmesh=nmesh)
    lengths = a * np.geomspace(0.2, upper, nlength)
    sig, curves, shapes = fsm.solver.signature(nodes, elems, lengths)
    first = np.asarray(curves)[:, 0]
    i = int(np.argmin(first))
    stress = float(first[i])
    # refine with parabola then exact re-solve
    sig_r, L_r, _ = fsm.solver.refined_minimum(lengths, first)
    sig2, eigs2, shapes2 = fsm.solver.signature(nodes, elems, np.array([L_r]))
    parts, disp = fsm.participation(shapes2[0, 0], nodes, tags)
    k = float(eigs2[0, 0]) * r19lib.kfactor(100)
    return dict(x=x, r=r, s=s, k=k, sigma=float(eigs2[0, 0]),
                half_wave=L_r / a, lips=parts['lips'],
                boundary=(i in (0, nlength - 1)))


if __name__ == '__main__':
    path = OUT / 'channel_grid.csv'
    done = pd.read_csv(path) if path.exists() else pd.DataFrame()
    cases = [dict(x=float(x), r=float(r), s=float(s))
             for s in (0.0, 0.25, 0.5)
             for x in (1.0, 1.1, 1.2, 1.3, 1.5)
             for r in np.round(np.arange(0.15, 0.601, 0.025), 3)]
    for i, case in enumerate(cases):
        if i < len(done):
            continue
        rec = solve_channel(**case)
        pd.DataFrame([rec]).to_csv(path, mode='a',
                                   header=not path.exists(), index=False)
        print(i + 1, '/', len(cases), rec['x'], rec['r'], rec['s'],
              round(rec['k'], 4), round(rec['lips'], 3), flush=True)
