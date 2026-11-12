"""R19b second-family crossover: does a return-fold mode exchange occur in lipped channels?

Family U (unequal-flange channel of channel_family.py): web a, bottom flange a,
  top flange b=a/x, lips c=ra on both flanges, inward returns d=sc; a/t=100.
Family C (conventional lipped channel with complex edge stiffener): web h=2b,
  equal flanges b, lips c=rb, inward returns d=sc; b/t=50 (b=50 mm, t=1 mm).
For each case s is scanned upward on a wide wavelength probe and the first sign
change of k_lip - k_leg between two coexisting stationary minima is solved by
Brent (r19b_lib.crossover_scan).  The governing-minimum class (cFSM, vector norm)
is recorded at s=0 and at the root.
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r19env  # noqa: F401
import numpy as np  # noqa: E402
import r19b_lib as lb  # noqa: E402
from channel_family import build_channel  # noqa: E402

OUT = lb.DATA / 'channel_crossover.json'
SEG = ['lip_a_d', 'lip_a', 'leg_a', 'leg_web', 'leg_b', 'lip_b', 'lip_b_d']


def build_std_channel(r, s, bt=50.0, h_over_b=2.0, t=1.0, nper=12):
    b = bt * t
    h, c = h_over_b * b, r * b
    d = s * c
    pts = [(b - d, c), (b, c), (b, 0.0), (0.0, 0.0), (0.0, h), (b, h), (b, h - c), (b - d, h - c)]
    names = SEG
    if s == 0:
        pts, names = pts[1:-1], SEG[1:-1]
    P = [np.array(p, float) for p in pts]
    coords, tags = [], []
    for i, (p0, p1) in enumerate(zip(P[:-1], P[1:])):
        n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / (b / nper))))
        coords += [p0 + (p1 - p0) * j / n for j in range(n)]
        tags += [names[i]] * n
    coords.append(P[-1])
    n = len(coords)
    nodes = np.zeros((n, 8))
    nodes[:, 0] = np.arange(n)
    nodes[:, 1:3] = coords
    nodes[:, 3:8] = 1
    elems = np.column_stack([np.arange(n - 1), np.arange(n - 1), np.arange(1, n),
                             np.full(n - 1, t), np.full(n - 1, 100)])
    return nodes, elems, tags, b


CASES = ([('U', 1.3, r) for r in (0.6, 0.5)]
         + [('C', None, r) for r in (0.6, 0.5, 0.4, 0.3, 0.2)])


def label_governing(builder, s, at, lo, hi):
    nodes, elems, tags, a = builder(s)
    tr, _ = lb.troughs_id(nodes, elems, tags, a, at, lo, hi, 80)
    return tr


if __name__ == '__main__':
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for fam, x, r in CASES:
        key = f'{fam}_x{x}_r{r}'
        if key in res:
            continue
        t0 = time.time()
        if fam == 'U':
            builder = lambda s, x=x, r=r: build_channel(x, r, float(s), nmesh=24, at=100)  # noqa: E731
            at, lo, hi, grid = 100.0, 0.3, 12.0, np.round(np.arange(0.0, 0.3001, 0.005), 4)
        else:
            builder = lambda s, r=r: build_std_channel(r, float(s))  # noqa: E731
            at, lo, hi, grid = 50.0, 0.3, 20.0, np.round(np.arange(0.0, 1.0001, 0.02), 3)
        log = []
        try:
            out, log = lb.crossover_scan(builder, at, grid, lo=lo, hi=hi, n=80, log=log)
        except Exception as exc:
            out = dict(status='error', error=repr(exc))
        out['s0_troughs_labelled'] = label_governing(builder, 0.0, at, lo, hi)
        if out.get('s_star') is not None:
            out['root_troughs_labelled'] = label_governing(builder, out['s_star'], at, lo, hi)
            out['d_over_t'] = out['s_star'] * r * at
        out.update(family=fam, x=x, r=r, at=at, elapsed_s=time.time() - t0, scan=log)
        res[key] = out
        OUT.write_text(json.dumps(res, indent=1))
        print(key, out['status'], out.get('s_star'), out.get('d_over_t'), out.get('first_local_s'),
              [(round(t['L_over_a'], 2), round(t['k'], 3), t['dominant'], round(t['lips'], 2))
               for t in out['s0_troughs_labelled']], f"{out['elapsed_s']:.0f}s", flush=True)
