"""Shared helpers for R19 uplift computations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r19env  # noqa: F401

import numpy as np
from scipy.optimize import minimize_scalar, brentq

import research_scan as fsm


def kfactor(at=100):
    """Dimensionless plate coefficient normalisation for a given a/t."""
    return 12 * (1 - fsm.NU**2) * at**2 / (np.pi**2 * fsm.E)


def trough(x, r, s, branch, mesh=24, at=100, nodes=None, elems=None, tags=None,
           a=None, bounds=None):
    """Nested minimisation of the first eigenvalue inside the named bracket."""
    if nodes is None:
        nodes, elems, tags, a = fsm.build(x, r, s, nmesh=mesh, at=at)
    if bounds is None:
        if branch == 'leg':
            bounds = (0.6, 1.4)
        elif r == 0.5:
            bounds = (1.05, 2.8)
        else:
            bounds = (1.7, 3.6)

    def response(L):
        sig, eigs, shapes = fsm.solver.signature(nodes, elems, np.array([a * L]))
        return eigs[0, 0]

    opt = minimize_scalar(response, bounds=bounds, method='bounded',
                          options={'xatol': 1e-5})
    L = opt.x
    sig, eigs, shapes = fsm.solver.signature(nodes, elems, np.array([a * L]))
    parts, disp = fsm.participation(shapes[0, 0], nodes, tags)
    if abs(L - bounds[0]) < .001 or abs(L - bounds[1]) < .001:
        raise RuntimeError('Boundary trough ' + str((x, r, s, branch, L)))
    return dict(half_wave=L, k=float(eigs[0, 0] * kfactor(at)),
                sigma=float(eigs[0, 0]), lips=parts['lips'])


def crossover(x, r, mesh, lo=.06, hi=None, at=100, nodes_builder=None):
    """Brent root of lip_trough(s) - leg_trough(s); returns result + iteration log."""
    if hi is None:
        hi = 0.09 if r == .5 else .10
    log = []

    def gap(s):
        lip = trough(x, r, s, 'lip', mesh, at=at)
        leg = trough(x, r, s, 'leg', mesh, at=at)
        log.append(dict(x=x, r=r, s=s, mesh=mesh, at=at, lip=lip, leg=leg))
        return lip['k'] - leg['k']

    root = brentq(gap, lo, hi, xtol=2e-6)
    gap(root)
    result = log[-1]
    result['d_over_a'] = root * r
    result['d_over_t'] = root * r * at
    result['s_star'] = root
    return result, log


def spectrum_eigs(nodes, elems, half_waves_a, n_eigs=4):
    """First-n_eigs eigenvalue curves over the given half-wave grid (in units of a)."""
    a_half = np.asarray(half_waves_a)
    sig, curve, shapes = fsm.solver.signature(nodes, elems, a_half)
    return np.asarray(curve), shapes


def section_area_mm2(nodes, t=1.0):
    """Centreline length times thickness (gross area)."""
    seg = np.linalg.norm(np.diff(nodes[:, 1:3], axis=0), axis=1)
    return float(seg.sum() * t)
