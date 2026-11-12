"""R19 rounded-corner sensitivity of the return-fold threshold.

Each polyline bend is replaced by a circular arc of centreline radius
rm = (rho + 0.5) * t with inner radius ri = rho * t, subdivided into N_ARC
straight strips; the two adjacent legs shorten by rm each. If a segment is
too short to host both tangent offsets the geometry is marked infeasible.

The threshold s* is re-solved at (x=1.3, r=0.6, a/t=100, mesh 24) for
rho in {0, 0.5, 1, 2} plus a coarse rho=4 diagnostic.
"""
import sys
import json
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r19env  # noqa: F401
import r19lib  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize_scalar, brentq  # noqa: E402
import research_scan as fsm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data'

SEG_COMPLEX = ['lip_b_d', 'lip_b', 'leg_b', 'leg_a', 'lip_a', 'lip_a_d']
SEG_SIMPLE = ['lip_b', 'leg_b', 'leg_a', 'lip_a']

N_ARC = 8  # strips per corner


class RadiusInfeasible(Exception):
    pass


def base_polyline(a, b, c, d, is_complex):
    if is_complex:
        return [(c, b - d), (c, b), (0.0, b), (0.0, 0.0), (a, 0.0), (a, c), (a - d, c)]
    return [(c, b), (0.0, b), (0.0, 0.0), (a, 0.0), (a, c)]


def rounded_nodes(x, r, s, rho, nmesh=24, at=100, t=1.0):
    """Nodes/elems/tags with every polyline bend filleted at rm=(rho+0.5)*t."""
    a = at * t
    b, c = a / x, a * r
    d = s * c
    P = [np.array(p, dtype=float) for p in base_polyline(a, b, c, d, s > 0)]
    names = SEG_COMPLEX if s > 0 else SEG_SIMPLE
    rm = (rho + 0.5) * t

    coords, tags = [], []
    nseg = len(P) - 1
    for i in range(nseg):
        p0, p1 = P[i], P[i + 1]
        e1 = (p1 - p0) / np.linalg.norm(p1 - p0)
        a_off = rm if i > 0 else 0.0
        b_off = rm if i < nseg - 1 else 0.0
        q0 = p0 + e1 * a_off
        q1 = p1 - e1 * b_off
        seg_len = np.linalg.norm(q1 - q0)
        if seg_len <= 0:
            raise RadiusInfeasible(
                f'segment {names[i]} too short for radius {rho}: flat={np.linalg.norm(p1-p0):.2f}mm, need>{2*rm:.2f}mm')
        n = max(2, int(np.ceil(seg_len / (a / nmesh))))
        for j in range(n):
            coords.append(q0 + (q1 - q0) * j / n)
            tags.append(names[i])
        if i < nseg - 1:
            e2 = (P[i + 2] - p1) / np.linalg.norm(P[i + 2] - p1)
            vend = p1 + e2 * rm
            cross = e1[0] * e2[1] - e1[1] * e2[0]
            side = 1.0 if cross > 0 else -1.0
            centre = q1 + np.array([-e1[1], e1[0]]) * (rm * side)
            th0 = math.atan2(*(q1 - centre)[::-1])
            th1 = math.atan2(*(vend - centre)[::-1])
            if abs(th1 - th0) > math.pi:
                th1 += 2 * math.pi * np.sign(th0 - th1)
            for j in range(1, N_ARC):
                th = th0 + (th1 - th0) * j / N_ARC
                coords.append(centre + rm * np.array([math.cos(th), math.sin(th)]))
                tags.append('corner')
    coords.append(P[-1])

    n = len(coords)
    nodes = np.zeros((n, 8))
    nodes[:, 0] = np.arange(n)
    nodes[:, 1:3] = coords
    nodes[:, 3:8] = 1
    elems = np.column_stack([np.arange(n - 1), np.arange(n - 1), np.arange(1, n),
                             np.full(n - 1, t), np.full(n - 1, 100)])
    return nodes, elems, tags, a


def trough_rounded(nodes, elems, tags, a, branch, r):
    bounds = (0.65, 1.1) if branch == 'leg' else ((1.45, 2.6) if r == .5 else (2.0, 3.5))

    def response(L):
        sig, eigs, shapes = fsm.solver.signature(nodes, elems, np.array([a * L]))
        return eigs[0, 0]

    opt = minimize_scalar(response, bounds=bounds, method='bounded',
                          options={'xatol': 1e-5})
    L = opt.x
    sig, eigs, shapes = fsm.solver.signature(nodes, elems, np.array([a * L]))
    parts, disp = fsm.participation(shapes[0, 0], nodes, tags)
    return dict(half_wave=L, k=float(eigs[0, 0] * r19lib.kfactor(100)),
                lips=parts['lips'])


def crossover_rounded(x, r, rho, mesh=24, lo=.06, hi=.10):
    log = []

    def gap(s):
        nodes, elems, tags, a = rounded_nodes(x, r, s, rho, nmesh=mesh)
        lip = trough_rounded(nodes, elems, tags, a, 'lip', r)
        leg = trough_rounded(nodes, elems, tags, a, 'leg', r)
        log.append(dict(s=s, lip=lip, leg=leg))
        return lip['k'] - leg['k']

    # find a feasible lower bound (radius may not fit at small s)
    s_lo = lo
    while s_lo < hi - 0.005:
        try:
            g_lo = gap(s_lo)
            break
        except RadiusInfeasible as exc:
            log.append(dict(s=s_lo, infeasible=str(exc)))
            s_lo += 0.01
    else:
        return None, log
    g_hi = gap(hi)
    if g_lo * g_hi > 0:
        return dict(no_crossing=True, g_lo=g_lo, g_hi=g_hi, s_lo=s_lo), log
    root = brentq(gap, s_lo, hi, xtol=2e-6)
    gap(root)
    return root, log


if __name__ == '__main__':
    results = {}
    path = OUT / 'corner_radius.json'
    if path.exists():
        results = json.loads(path.read_text())
    for rho in (0.0, 0.5, 1.0, 2.0, 4.0):
        key = f'{rho}'
        if key in results:
            continue
        try:
            root, log = crossover_rounded(1.3, 0.6, rho)
        except Exception as exc:  # record and continue
            results[key] = dict(error=str(exc))
            path.write_text(json.dumps(results, indent=2))
            print('rho=', rho, 'ERROR', exc, flush=True)
            continue
        if isinstance(root, dict):
            results[key] = dict(**root, d_over_t=None)
        elif root is None:
            results[key] = dict(infeasible_range=True)
        else:
            results[key] = dict(s_star=root, d_over_t=root * 0.6 * 100,
                                final=log[-1])
        path.write_text(json.dumps(results, indent=2))
        print('rho=', rho, 'result=', results[key], flush=True)
