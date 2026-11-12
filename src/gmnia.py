"""R19b GMNIA of lipped angles with return folds (Abaqus 2020, S4R, Riks).

Purpose: test the two DSM-screening findings with nonlinear member analysis -
(i) local-governed strength needs s = 0.13-0.15 at L = 0.8 m, (ii) the first fold
changes the DSM strength by -25 % although the elastic load does not change.

Model: centreline geometry identical to the finite-strip section (same section nodes);
S4R, 5 points through thickness; E = 210000 MPa, nu = 0.3, elastic-perfectly plastic
(f_y, 1.0001 f_y at plastic strain 0.2); pinned, warping-free ends (end-section nodes
U1 = U2 = UR3 = 0; one U3 restraint); uniform end forces = tributary width * t * f_y,
so the Riks load proportionality factor is P/P_y.
Imperfections (defined by deformation location, not by modal label, so that the first
fold does not change them): lip-dominated FSM eigenvector at its stationary minimum,
amplitude 0.94t; leg-dominated local eigenvector of the cFSM local subspace at 0.84a,
amplitude 0.34t (50 % CDF values of Schafer & Pekoz 1998); classical flexural-torsional
rigid-section shape, amplitude L/1000; each with sin(n pi z/L), n = round(L/l).
"""
import sys
import json
import math
import time
import shutil
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r19env  # noqa: F401
import numpy as np  # noqa: E402
import research_scan as fsm  # noqa: E402
import cfsm_tools as ct  # noqa: E402
import r19b_lib as lb  # noqa: E402

ROUND = Path(__file__).resolve().parents[1]
RUNROOT = ROUND / 'gmnia_runs'
ARCH = ROUND / 'gmnia'
import os
ABQ = os.environ.get('ABAQUS_COMMAND', 'abaqus')
E, NU, T, A_LEG = 210000.0, 0.3, 1.0, 100.0


def section(x, r, s, h):
    a, b, c, d = A_LEG, A_LEG / x, A_LEG * r, A_LEG * r * s
    if s > 0:
        pts = [(c, b - d), (c, b), (0, b), (0, 0), (a, 0), (a, c), (a - d, c)]
        names = ['lip_b_d', 'lip_b', 'leg_b', 'leg_a', 'lip_a', 'lip_a_d']
    else:
        pts = [(c, b), (0, b), (0, 0), (a, 0), (a, c)]
        names = ['lip_b', 'leg_b', 'leg_a', 'lip_a']
    xy, tags = [], []
    for p, q, tag in zip(pts[:-1], pts[1:], names):
        n = max(2, math.ceil(math.dist(p, q) / h))
        xy += [[p[k] + (q[k] - p[k]) * j / n for k in range(2)] for j in range(n)]
        tags += [tag] * n
    xy.append(list(pts[-1]))
    n = len(xy)
    nodes = np.zeros((n, 8))
    nodes[:, 0] = np.arange(n)
    nodes[:, 1:3] = xy
    nodes[:, 3:8] = 1
    elems = np.column_stack([np.arange(n - 1), np.arange(n - 1), np.arange(1, n),
                             np.full(n - 1, T), np.full(n - 1, 100)])
    return np.array(xy), tags, nodes, elems


def inplane(vec, n):
    v = np.column_stack([vec[:2 * n:2], vec[2 * n:4 * n:2]])
    return v / np.abs(np.hypot(v[:, 0], v[:, 1])).max()


def ft_shape(nodes, elems, L):
    sp = ct.sect_props(nodes, elems)
    G = E / (2 * (1 + NU))
    A, I1, I2 = float(sp['A']), float(sp['I11']), float(sp['I22'])
    J, Cw, th = float(sp['J']), float(sp['Cw']), float(sp['phi'])
    cx, cy = float(sp['cx']), float(sp['cy'])
    x0, y0 = float(sp['x0']) - cx, float(sp['y0']) - cy
    c, s_ = np.cos(th), np.sin(th)
    u0, v0 = c * x0 + s_ * y0, -s_ * x0 + c * y0
    r0 = (I1 + I2) / A + u0 ** 2 + v0 ** 2
    P1, P2 = math.pi ** 2 * E * I1 / L ** 2, math.pi ** 2 * E * I2 / L ** 2
    Pz = (G * J + math.pi ** 2 * E * Cw / L ** 2) / r0
    roots = np.roots([r0 - u0 ** 2 - v0 ** 2, -r0 * (P1 + P2 + Pz) + u0 ** 2 * P2 + v0 ** 2 * P1,
                      r0 * (P1 * P2 + P1 * Pz + P2 * Pz), -r0 * P1 * P2 * Pz])
    P = float(np.sort(roots[np.abs(roots.imag) < 1e-9 * np.abs(roots).max()].real)[0])
    U1, U2 = P * v0 / (P2 - P), -P * u0 / (P1 - P)
    xi = np.column_stack([nodes[:, 1] - cx, nodes[:, 2] - cy]) @ np.array([[c, -s_], [s_, c]])
    d1 = U1 - (xi[:, 1] - v0)
    d2 = U2 + (xi[:, 0] - u0)
    d = np.column_stack([d1, d2]) @ np.array([[c, s_], [-s_, c]])
    return d / np.hypot(d[:, 0], d[:, 1]).max(), P / A


def shapes(x, r, s, L, h):
    xy, tags, nodes, elems = section(x, r, s, h)
    n = len(xy)
    a = A_LEG
    tr, _ = lb.troughs(nodes, elems, tags, a, 100, lo=0.3, hi=10.0, n=80, norms=())
    lips = [t for t in tr if t['lips'] > 0.5]
    lip = min(lips, key=lambda t: t['k'])
    sig, phi = lb.eig1(nodes, elems, lip['L_over_a'] * a)
    lip_shape = inplane(phi, n)
    counts = ct.class_counts(nodes, elems)
    for fac in (1.0, 1.003, 0.997, 1.01, 0.99, 1.03):
        try:
            ev, sh = ct.pure_class_eig(nodes, elems, 0.84 * fac * a, 'local', counts=counts)
            break
        except np.linalg.LinAlgError:  # isolated non-positive-definite reduced K
            continue
    cand = []
    for j in range(len(ev)):
        p = fsm.participation(sh[j], nodes, tags)[0]['lips']
        if p < 0.5:
            cand.append((ev[j], j, p))
    _, jl, p_leg = min(cand)
    leg_shape = inplane(sh[jl], n)
    g_shape, sig_ft = ft_shape(nodes, elems, L)
    return dict(xy=xy, tags=tags, nodes=nodes, elems=elems,
                lip=dict(shape=lip_shape, l=lip['L_over_a'] * a, k=lip['k'], P=lip['lips']),
                leg=dict(shape=leg_shape, l=0.84 * a, P=p_leg),
                glob=dict(shape=g_shape, sigma_ft=sig_ft))


def write_inp(name, x, r, s, L, fy, h=5.0, imp=1.0, lip_amp=0.94, leg_amp=0.34, glob_amp=1 / 1000):
    sh = shapes(x, r, s, L, h)
    xy, tags = sh['xy'], sh['tags']
    nc = len(xy)
    nz = 2 * math.ceil(L / h / 2)
    comps = [(lip_amp * T * imp, sh['lip']['shape'], max(1, round(L / sh['lip']['l']))),
             (leg_amp * T * imp, sh['leg']['shape'], max(1, round(L / sh['leg']['l']))),
             (glob_amp * L * imp, sh['glob']['shape'], 1)]
    node = lambda iz, j: iz * nc + j + 1  # noqa: E731
    out = ['*Heading', name, '*Node']
    for iz in range(nz + 1):
        z = L * iz / nz
        dxy = sum(A * S * math.sin(n * math.pi * z / L) for A, S, n in comps)
        out += ['%d, %.10g, %.10g, %.10g' % (node(iz, j), xy[j][0] + dxy[j, 0], xy[j][1] + dxy[j, 1], z)
                for j in range(nc)]
    out.append('*Element, type=S4R, elset=ALL')
    out += ['%d, %d, %d, %d, %d' % (iz * (nc - 1) + j + 1, node(iz, j), node(iz, j + 1),
                                     node(iz + 1, j + 1), node(iz + 1, j))
            for iz in range(nz) for j in range(nc - 1)]
    ends = [node(iz, j) for iz in (0, nz) for j in range(nc)]
    out.append('*Nset, nset=ENDS')
    out += [', '.join(map(str, ends[i:i + 12])) for i in range(0, len(ends), 12)]
    out += ['*Material, name=STEEL', '*Elastic', f'{E}, {NU}', '*Plastic', f'{fy}, 0.', f'{1.0001 * fy}, 0.2',
            '*Shell Section, elset=ALL, material=STEEL', f'{T}, 5']
    origin = min(range(nc), key=lambda i: xy[i][0] ** 2 + xy[i][1] ** 2)
    out += ['*Boundary', 'ENDS, 1, 2', 'ENDS, 6, 6', f'{node(0, origin)}, 3, 3']
    w = [0.0] * nc
    for j in range(nc - 1):
        hj = math.dist(xy[j], xy[j + 1]) / 2
        w[j] += hj
        w[j + 1] += hj
    out += ['*Step, name=RIKS, nlgeom=YES, inc=500', '*Static, riks', '0.02, 8.0, 1e-08, 0.06, 1.0',
            '*Cload']
    for iz, sign in ((0, 1), (nz, -1)):
        out += ['%d, 3, %.10g' % (node(iz, j), sign * wj * T * fy) for j, wj in enumerate(w)]
    out += ['*Output, field, frequency=1', '*Node Output', 'U,', '*End Step']
    folder = RUNROOT / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f'{name}.inp').write_text('\n'.join(out) + '\n')
    meta = dict(name=name, x=x, r=r, s=s, L=L, fy=fy, h=h, imp=imp, lip_amp=lip_amp, leg_amp=leg_amp,
                glob_amp=glob_amp, ncross=nc, nz=nz, area=sum(w) * T, tags=tags, xy=np.asarray(xy).tolist(),
                lip=dict(l=sh['lip']['l'], k=sh['lip']['k'], P=sh['lip']['P'], n=comps[0][2]),
                leg=dict(l=sh['leg']['l'], P=sh['leg']['P'], n=comps[1][2]),
                sigma_ft=sh['glob']['sigma_ft'])
    (folder / 'model.json').write_text(json.dumps(meta, indent=1))
    return folder, meta


def parse_sta(path):
    """Riks .sta rows: STEP INC ATT SEVERE EQUIL TOTAL [blank time] LPF dLPF R -> (inc, LPF)."""
    rows = []
    for line in path.read_text(errors='replace').splitlines():
        p = line.split()
        if len(p) >= 8 and p[-1] == 'R' and p[0].isdigit() and p[1].isdigit() and p[2].isdigit():
            try:
                rows.append((int(p[1]), float(p[6])))
            except ValueError:
                pass
    return rows


EXTRACT = r'''from odbAccess import openOdb
import json, sys
odbp, inc = sys.argv[-2], int(sys.argv[-1])
odb = openOdb(odbp, readOnly=True)
st = odb.steps[odb.steps.keys()[-1]]
fr = st.frames[-1]
for f in st.frames:
    if f.incrementNumber == inc:
        fr = f
u = [[int(v.nodeLabel)] + [float(q) for q in v.data] for v in fr.fieldOutputs['U'].values]
json.dump({'inc': fr.incrementNumber, 'U': u}, open(odbp[:-4] + '_peakU.json', 'w'))
odb.close()
'''


def run(name, cpus=4, solve=True):
    folder = RUNROOT / name
    t0 = time.time()
    if solve:
        with (folder / 'console.log').open('w') as log:
            subprocess.run(['cmd', '/c', ABQ, f'job={name}', 'interactive', f'cpus={cpus}', 'ask_delete=OFF'],
                           cwd=folder, stdout=log, stderr=subprocess.STDOUT)
    sta = folder / f'{name}.sta'
    rows = parse_sta(sta) if sta.exists() else []
    res = dict(name=name, wall_s=time.time() - t0, n_inc=len(rows))
    if rows:
        inc, lpf = max(rows, key=lambda q: q[1])
        res.update(lpf_max=lpf, inc_peak=inc, lpf_last=rows[-1][1],
                   post_peak=rows[-1][1] < 0.98 * lpf, curve=rows)
        (folder / 'extract_gmnia.py').write_text(EXTRACT)
        with (folder / 'extract.log').open('w') as log:
            subprocess.run(['cmd', '/c', ABQ, 'python', 'extract_gmnia.py', f'{name}.odb', str(inc)],
                           cwd=folder, stdout=log, stderr=subprocess.STDOUT)
    res.update(mode=failure_mode(folder))
    (folder / 'result.json').write_text(json.dumps(res, indent=1))
    ARCH.mkdir(exist_ok=True)
    dst = ARCH / name
    dst.mkdir(exist_ok=True)
    for f in ('model.json', 'result.json', f'{name}.inp', f'{name}.sta', f'{name}.msg'):
        if (folder / f).exists():
            shutil.copy2(folder / f, dst / f)
    return res


def failure_mode(folder):
    """Lip share and dominant half-wave count of the deformation added between the
    imperfect geometry and the peak load."""
    pj = next(folder.glob('*_peakU.json'), None)
    if pj is None:
        return None
    meta = json.loads((folder / 'model.json').read_text())
    U = np.array(json.loads(pj.read_text())['U'])
    nc, nz = meta['ncross'], meta['nz']
    U = U[np.argsort(U[:, 0])]
    uv = U[:, 1:3].reshape(nz + 1, nc, 2)
    mag2 = (uv ** 2).sum(axis=2)
    tags = np.array(meta['tags'] + [meta['tags'][-1]])
    lipmask = np.array([t.startswith('lip') for t in tags])
    P = float(mag2[:, lipmask].sum() / mag2.sum())

    def harmonic(j):
        sig = uv[:, j, :]
        sig = sig[:, np.argmax(np.abs(sig).max(axis=0))]
        z = np.linspace(0, 1, nz + 1)
        amps = [abs(np.trapezoid(sig * np.sin(k * np.pi * z), z)) for k in range(1, 40)]
        return int(np.argmax(amps) + 1)
    tip = nc - 1
    legmid = int(np.argmin([abs(xy[0] - 50.0) + abs(xy[1]) for xy in meta['xy']]))
    return dict(P_lip=P, n_lip_tip=harmonic(tip), n_leg_mid=harmonic(legmid),
                umax=float(np.sqrt(mag2.max())))


CASES = {}


def add(x=1.3, r=0.6, s=0.0, L=800.0, fy=250.0, h=5.0, imp=1.0, tag='', **kw):
    name = f'g_x{x}_r{r}_s{s:.2f}_L{int(L)}_fy{int(fy)}_h{h}_i{imp}{tag}'
    CASES[name] = dict(x=x, r=r, s=s, L=L, fy=fy, h=h, imp=imp, **kw)


for s in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16):          # A: primary
    add(s=s)
for L in (1200.0, 1800.0, 2400.0):                                       # B: length
    for s in (0.0, 0.02, 0.06, 0.10, 0.14):
        add(s=s, L=L)
for fy in (350.0, 450.0):                                                # C: yield stress
    for s in (0.0, 0.02, 0.08, 0.14):
        add(s=s, fy=fy)
for imp in (0.5, 2.0):                                                   # D: imperfections
    for s in (0.0, 0.02, 0.08, 0.14):
        add(s=s, imp=imp)
for s in (0.0, 0.02, 0.06, 0.10, 0.14):                                  # E: lip ratio 0.5
    add(r=0.5, s=s)
for s in (0.0, 0.14):                                                    # F: mesh
    add(s=s, h=2.5)
add(s=0.0, tag='_labelbased', lip_amp=0.34)                              # G: label-based amplitude


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'check':
        sh = shapes(1.3, 0.6, 0.08, 6000.0, 5.0)
        xy, tags, nodes, elems = section(1.3, 0.6, 0.08, 5.0)
        sig, phi = lb.eig1(nodes, elems, 6000.0)
        f = inplane(phi, len(xy)).ravel()
        g = sh['glob']['shape'].ravel()
        print('FT sigma classical', round(sh['glob']['sigma_ft'], 3), 'free', round(sig, 3),
              'MAC(rigid FT shape, FSM eigvec at 60a) =', round(float((f @ g) ** 2 / ((f @ f) * (g @ g))), 5))
        print('lip', {k: v for k, v in sh['lip'].items() if k != 'shape'},
              'leg', {k: v for k, v in sh['leg'].items() if k != 'shape'})
    elif mode == 'gen':
        names = [n for n in CASES if not sys.argv[2:] or any(k in n for k in sys.argv[2:])]
        for n in names:
            c = dict(CASES[n])
            folder, meta = write_inp(n, **c)
            print(n, meta['ncross'], meta['nz'], 'lip l/a', round(meta['lip']['l'] / 100, 3), 'n', meta['lip']['n'],
                  'leg n', meta['leg']['n'], 'P_leg', round(meta['leg']['P'], 3), flush=True)
    elif mode == 'post':
        for n in sys.argv[2:]:
            res = run(n, solve=False)
            print(n, res.get('lpf_max'), res.get('inc_peak'), res.get('n_inc'), res.get('post_peak'),
                  res.get('mode'), flush=True)
    elif mode == 'run':
        from concurrent.futures import ThreadPoolExecutor
        workers = int(sys.argv[2])
        names = [n for n in CASES if (len(sys.argv) <= 3 or any(k in n for k in sys.argv[3:]))
                 and not (ARCH / n / 'result.json').exists()]
        for n in names:
            if not (RUNROOT / n / f'{n}.inp').exists():
                write_inp(n, **CASES[n])
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for res in ex.map(run, names):
                print(res['name'], res.get('lpf_max'), res.get('inc_peak'), res.get('n_inc'),
                      res.get('post_peak'), round(res['wall_s']), res.get('mode'), flush=True)
