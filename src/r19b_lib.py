"""R19b shared helpers: wide-band trough probing, refinement and cFSM labelling.

Replaces the fixed-window grid minima used in member_dsm.py (which returned
window-boundary values for s<=0.04 and s>=0.12) and the narrow probes of
at_robustness.py (s in {0.06, 0.08} only, failing silently at a/t=100).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r19env  # noqa: F401
import numpy as np  # noqa: E402
from scipy.optimize import minimize_scalar, brentq  # noqa: E402
import research_scan as fsm  # noqa: E402
import r19lib  # noqa: E402
import cfsm_tools as ct  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'


def eig1(nodes, elems, L_mm):
    sig, curve, shapes = fsm.solver.signature(nodes, elems, np.array([L_mm]))
    return float(np.asarray(curve)[0, 0]), np.asarray(shapes)[0, 0]


def probe(nodes, elems, a, lo=0.3, hi=10.0, n=80):
    La = np.geomspace(lo, hi, n)
    sig, curve, shapes = fsm.solver.signature(nodes, elems, La * a)
    return La, np.asarray(curve)[:, 0]


def minima_idx(c):
    return [i for i in range(1, len(c) - 1) if c[i] < c[i - 1] and c[i] <= c[i + 1]]


def refine_min(nodes, elems, a, lo, hi, xatol=1e-5):
    opt = minimize_scalar(lambda L: eig1(nodes, elems, L * a)[0], bounds=(lo, hi),
                          method='bounded', options={'xatol': xatol})
    sigma, phi = eig1(nodes, elems, opt.x * a)
    return float(opt.x), sigma, phi


def label(nodes, elems, L_mm, phi, norms=(1,)):
    out = {}
    for nrm in norms:
        b_v, counts = ct.gdlo_basis(nodes, elems, L_mm, orth=2, norm=nrm)
        g = ct.participation_gdlo(b_v, counts, phi)
        suffix = '' if nrm == 1 else f'_n{nrm}'
        out.update({f'G{suffix}': g[0], f'D{suffix}': g[1], f'L{suffix}': g[2],
                    f'O{suffix}': g[3], f'dominant{suffix}': 'GDLO'[int(np.argmax(g))]})
    return out


def troughs(nodes, elems, tags, a, at, lo=0.3, hi=10.0, n=80, norms=(1,)):
    """All interior stationary minima of the first eigenvalue on [lo,hi]*a."""
    La, c = probe(nodes, elems, a, lo, hi, n)
    kf = r19lib.kfactor(at)
    out = []
    for i in minima_idx(c):
        L, sigma, phi = refine_min(nodes, elems, a, La[i - 1], La[i + 1])
        rec = dict(L_over_a=L, sigma=sigma, k=sigma * kf,
                   lips=fsm.participation(phi, nodes, tags)[0]['lips'])
        if norms:
            rec.update(label(nodes, elems, L * a, phi, norms))
        out.append(rec)
    return out, (La, c * kf)


def pure_min(nodes, elems, a, cls, lo, hi, n=12, xatol=2e-3, counts=None):
    """Minimum over half-wavelength of the first pure-class (cFSM) eigenvalue."""
    counts = counts or ct.class_counts(nodes, elems)

    def f(L):
        # reduced K can be numerically non-positive-definite at isolated L;
        # retry at slightly perturbed half-wavelengths before giving up
        for fac in (1.0, 1.0007, 0.9993, 1.003, 0.997):
            try:
                ev = np.ravel(ct.pure_class_eig(nodes, elems, L * fac * a, cls, counts=counts)[0])
            except np.linalg.LinAlgError:
                continue
            ev = ev[np.isfinite(ev) & (ev > 0)]
            if ev.size:
                return float(ev[0])
        return np.nan

    La = np.geomspace(lo, hi, n)
    v = np.array([f(L) for L in La])
    if not np.isfinite(v).any():
        return float('nan'), float('nan'), True
    i = int(np.nanargmin(v))
    if 0 < i < n - 1 and np.isfinite(v[i - 1]) and np.isfinite(v[i + 1]):
        opt = minimize_scalar(lambda L: f(L) if np.isfinite(f(L)) else 1e12,
                              bounds=(La[i - 1], La[i + 1]), method='bounded',
                              options={'xatol': xatol})
        return float(opt.x), float(opt.fun), False
    return float(La[i]), float(v[i]), True  # boundary flag


def classical_ft(sp, L, E=210000.0, nu=0.3):
    """Pinned (warping-free) flexural / torsional / flexural-torsional buckling stresses.

    Vlasov cubic  r0^2 (P-P1)(P-P2)(P-Pz) - u0^2 P^2 (P-P2) - v0^2 P^2 (P-P1) = 0,
    shear-centre offsets (u0, v0) in principal axes (1, 2).  Checked against the free
    FSM at L = 60a (19.4 vs 19.4 MPa) once cutwp's warping constant was repaired."""
    G = E / (2 * (1 + nu))
    A, I11, I22 = float(sp['A']), float(sp['I11']), float(sp['I22'])
    J, Cw = float(sp['J']), float(sp['Cw'])
    x0, y0 = float(sp['x0']) - float(sp['cx']), float(sp['y0']) - float(sp['cy'])
    th = float(sp['phi'])
    u0 = np.cos(th) * x0 + np.sin(th) * y0
    v0 = -np.sin(th) * x0 + np.cos(th) * y0
    r0sq = (I11 + I22) / A + u0 ** 2 + v0 ** 2
    P1, P2 = np.pi ** 2 * E * I11 / L ** 2, np.pi ** 2 * E * I22 / L ** 2
    Pz = (G * J + np.pi ** 2 * E * Cw / L ** 2) / r0sq
    roots = np.roots([r0sq - u0 ** 2 - v0 ** 2, -r0sq * (P1 + P2 + Pz) + u0 ** 2 * P2 + v0 ** 2 * P1,
                      r0sq * (P1 * P2 + P1 * Pz + P2 * Pz), -r0sq * P1 * P2 * Pz])
    roots = np.sort(roots[np.abs(roots.imag) < 1e-6 * np.abs(roots).max()].real)
    roots = roots[roots > 0]
    return dict(P1=P1 / A, P2=P2 / A, Pz=Pz / A, FT=float(roots[0] / A) if roots.size else np.nan)


def vertex_nodes(nodes):
    xy = nodes[:, 1:3]
    d = np.diff(xy, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    turn = [i for i in range(1, len(xy) - 1)
            if abs(np.angle(np.exp(1j * (ang[i] - ang[i - 1])))) > 1e-6]
    return [0] + turn + [len(xy) - 1]


def fold_kinematics(nodes, phi):
    """In-plane translation of every fold line / free edge relative to the largest nodal
    translation, and the rigid-body share of the fold-line translation field."""
    n = len(nodes)
    u, w = phi[:2 * n:2], phi[2 * n:4 * n:2]
    mag = np.hypot(u, w)
    vid = vertex_nodes(nodes)
    vm = (mag[vid] / mag.max()).tolist()
    x, z = nodes[vid, 1], nodes[vid, 2]
    xc, zc = x.mean(), z.mean()
    m = len(vid)
    A = np.zeros((2 * m, 3))
    A[:m, 0], A[:m, 2] = 1.0, -(z - zc)
    A[m:, 1], A[m:, 2] = 1.0, (x - xc)
    b = np.concatenate([u[vid], w[vid]])
    coef = np.linalg.lstsq(A, b, rcond=None)[0]
    res = b - A @ coef
    rigid = float(1 - (res @ res) / (b @ b)) if b @ b > 0 else float('nan')
    return dict(vertex_translation=vm, internal_fold_max=float(max(vm[1:-1])) if m > 2 else 0.0,
                rigid_fraction=rigid)


def identify(nodes, elems, a, L_over_a, sigma_free, phi, counts=None, tol=1.10):
    """Mode identity without the unvalidated cFSM G/D bases.

    local       : the cFSM local subspace reproduces the free eigenvalue (rho_L <= tol);
    global      : classical flexural-torsional buckling at the same half-wave does (rho_FT <= tol);
    distortional: neither, i.e. fold lines translate but not as a rigid cross-section."""
    counts = counts or ct.class_counts(nodes, elems)
    L = L_over_a * a
    sig_l = np.nan
    for fac in (1.0, 1.0007, 0.9993, 1.003):
        try:
            ev = np.ravel(ct.pure_class_eig(nodes, elems, L * fac, 'local', counts=counts)[0])
        except np.linalg.LinAlgError:
            continue
        ev = ev[np.isfinite(ev) & (ev > 0)]
        if ev.size:
            sig_l = float(ev[0])
            break
    ft = classical_ft(ct.sect_props(nodes, elems), L)['FT']
    rho_l, rho_ft = sig_l / sigma_free, ft / sigma_free
    cls = 'L' if rho_l <= tol else ('G' if rho_ft <= tol else 'D')
    return dict(rho_L=rho_l, rho_FT=rho_ft, sigma_pureL=sig_l, sigma_FT=ft, cls=cls,
                **fold_kinematics(nodes, phi))


def troughs_id(nodes, elems, tags, a, at, lo=0.3, hi=10.0, n=80):
    """Free stationary minima, each with the identification of identify()."""
    tr, curve = troughs(nodes, elems, tags, a, at, lo, hi, n, norms=())
    counts = ct.class_counts(nodes, elems)
    for t in tr:
        sigma, phi = eig1(nodes, elems, t['L_over_a'] * a)
        t.update(identify(nodes, elems, a, t['L_over_a'], sigma, phi, counts))
    return tr, curve


def pick_pair(tr):
    """(leg-dominated, lip-dominated) minima by lip participation; None if <2."""
    if len(tr) < 2:
        return None
    leg = min(tr, key=lambda t: t['lips'])
    lip = max(tr, key=lambda t: t['lips'])
    return (leg, lip) if leg is not lip else None


def gap_at(builder, s, at, lo=0.3, hi=12.0, n=80):
    nodes, elems, tags, a = builder(s)
    tr, _ = troughs(nodes, elems, tags, a, at, lo, hi, n, norms=())
    pair = pick_pair(tr)
    if pair is None:
        return None, tr
    leg, lip = pair
    return lip['k'] - leg['k'], tr


def crossover_scan(builder, at, s_grid, lo=0.3, hi=12.0, n=80, log=None):
    """Scan s upward; return the first two-trough sign change bracket and Brent root."""
    log = [] if log is None else log
    prev = None
    for s in s_grid:
        g, tr = gap_at(builder, s, at, lo, hi, n)
        log.append(dict(s=float(s), gap=g, troughs=tr))
        if g is not None and g > 0:
            if prev is not None and prev[1] is not None and prev[1] < 0:
                root = brentq(lambda z: gap_at(builder, z, at, lo, hi, n)[0],
                              prev[0], s, xtol=1e-5)
                g2, tr2 = gap_at(builder, root, at, lo, hi, n)
                return dict(status='crossing', s_star=float(root), bracket=[prev[0], float(s)],
                            troughs_at_root=tr2), log
            status = 'local_from_start' if prev is None else 'emergent'
            return dict(status=status, first_local_s=float(s),
                        previous=None if prev is None else dict(s=prev[0], gap=prev[1])), log
        prev = (float(s), g)
    return dict(status='no_crossing_in_range', s_max=float(s_grid[-1])), log
