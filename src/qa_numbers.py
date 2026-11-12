"""R19b numerical QA: recompute every manuscript number that comes from the R19b data files.

Each check prints the recomputed value next to the value stated in main.tex and a PASS/FAIL
flag (tolerance = rounding of the stated value).  Output is also written to qa/qa_numbers.txt.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r19env  # noqa: F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'data'
J = lambda f: json.loads((D / f).read_text())  # noqa: E731
lines, fails = [], 0


def check(label, value, stated, tol):
    global fails
    ok = abs(value - stated) <= tol
    fails += 0 if ok else 1
    lines.append(f"{'PASS' if ok else 'FAIL'}  {label}: computed {value:.5g}, stated {stated}")


def rng(label, vals, lo, hi, tol):
    check(label + ' (min)', min(vals), lo, tol)
    check(label + ' (max)', max(vals), hi, tol)


# --- identification (Table 3, Sections 3 and 4.1-4.2) ---
idv = J('identify_v3.json')
simple = [idv[f'x1.3_r{r}_s0.0']['troughs'][0] for r in (0.2, 0.3, 0.4, 0.5, 0.6)]
rng('rho_L simple lips r=0.2-0.6', [t['rho_L'] for t in simple], 1.000, 1.003, 0.0005)
check('max internal fold translation, simple lips', max(t['internal_fold_max'] for t in simple), 0.01, 0.005)
t0, t2 = idv['x1.3_r0.6_s0.0']['troughs'][0], idv['x1.3_r0.6_s0.02']['troughs'][0]
check('k s=0 (r=0.6)', t0['k'], 2.425, 0.0005)
check('k s=0.02', t2['k'], 2.459, 0.0005)
check('first-fold elastic change (%)', 100 * (t2['k'] / t0['k'] - 1), 1.4, 0.05)
check('rho_L s=0.02', t2['rho_L'], 2.883, 0.0005)
check('rigid share s=0.02', t2['rigid_fraction'], 0.29, 0.005)
lip = [max(v['troughs'], key=lambda t: t['lips']) for k, v in idv.items()
       if k.startswith('x1.3_r0.6_s') and not k.endswith('s0.0')]
rng('rho_FT lip-return minima', [t['rho_FT'] for t in lip], 3.3, 660, 0.5)
nonlocal_rl = [t['rho_L'] for v in idv.values() if 'troughs' in v for t in v['troughs'] if t['cls'] == 'D']
rng('rho_L non-local minima', nonlocal_rl, 2.2, 9.2, 0.05)
legs = [min(v['troughs'], key=lambda t: t['lips']) for k, v in idv.items()
        if k.startswith('x1.3_r0.6_s') and len(v.get('troughs', [])) == 2]
rng('rho_L leg-local minima', [t['rho_L'] for t in legs], 1.001, 1.035, 0.0005)
rng('k leg-local minima', [t['k'] for t in legs], 5.06, 5.20, 0.005)
gc = [v for k, v in idv.items() if k.startswith('global_check')]
rng('rho_FT at 24a/60a', [v['rho_FT'] for v in gc], 1.000, 1.009, 0.0005)
rng('rigid share at 24a/60a', [v['rigid_fraction'] for v in gc], 0.999, 1.000, 0.0005)

# --- regime map (Section 4.2) ---
rm = J('regime_map_v3.json')
g = lambda r, s: rm[f'r{r:.3f}_s{s:.3f}']['governing']  # noqa: E731
S = sorted({v['s'] for v in rm.values()})
for r, lo, hi in ((0.15, 3.0, 3.0), (0.30, 13.8, 13.8)):
    check(f'gain r={r} over s (%)', 100 * (g(r, 0.2)['k'] / g(r, 0.0)['k'] - 1), lo, 0.1)
check('max P_l of governing minimum, r<=0.30, s>0',
      max(g(r, s)['lips'] for r in (0.15, 0.2, 0.25, 0.3) for s in S), 0.14, 0.005)
rng('rho_L at s=0.02, r<=0.30', [g(r, 0.02)['rho_L'] for r in (0.15, 0.2, 0.25, 0.3)], 1.18, 1.21, 0.006)
rng('P_l at s=0, r>=0.45', [g(r, 0.0)['lips'] for r in (0.45, 0.5, 0.55, 0.6)], 0.64, 0.91, 0.006)
dist = [g(r, s)['rho_L'] for r in (0.45, 0.5, 0.55, 0.6) for s in S if g(r, s)['rho_L'] > 1.5]
rng('rho_L lip-return governing, r>=0.45', dist, 1.6, 3.7, 0.05)
check('single minimum everywhere for r=0.35-0.40',
      max(rm[f'r{r:.3f}_s{s:.3f}']['n_minima'] for r in (0.35, 0.4) for s in S), 1, 0)
for r, st in ((0.6, 114), (0.5, 62), (0.4, 27)):
    check(f'gain r={r} (%)', 100 * (max(g(r, s)['k'] for s in S) / g(r, 0.0)['k'] - 1), st, 1)
post = {0.6: 0.10, 0.5: 0.08, 0.4: 0.08}
for r, s0 in post.items():
    check(f'post-exchange gain r={r} (%)', 100 * (g(r, 0.2)['k'] / g(r, s0)['k'] - 1),
          {0.6: 0.4, 0.5: 1.5, 0.4: 3.1}[r], 0.1)

# --- thickness scaling (Table 5) ---
for f, r in (('at_crossover.json', 0.6), ('at_crossover_r05.json', 0.5)):
    d = J(f)
    at = np.array([50, 75, 100, 150, 200])
    ss = np.array([d[f'x1.3_r{r}_at{a}']['s_star'] for a in at])
    dt = ss * r * at
    lines.append(f'INFO  r={r}: s* = {np.round(ss, 4).tolist()}, d*/t = {np.round(dt, 2).tolist()}')
    check(f'r={r} s* ratio a/t 50/200', ss[0] / ss[-1], 3.3, 0.06)
    check(f'r={r} d*/t growth (%)', 100 * (dt[-1] / dt[0] - 1), {0.6: 20, 0.5: 23}[r], 0.6)
    check(f'r={r} exponent s*', np.polyfit(np.log(at), np.log(ss), 1)[0], {0.6: -0.87, 0.5: -0.85}[r], 0.006)
    check(f'r={r} exponent d*/t', np.polyfit(np.log(at), np.log(dt), 1)[0], {0.6: 0.13, 0.5: 0.15}[r], 0.006)
    if r == 0.6:
        check('mesh-24 reference vs refined 0.0871 (%)', 100 * (1 - ss[2] / 0.0871), 0.3, 0.06)

# --- corner radius ---
cr = J('corner_radius.json')
sv = [cr[k]['s_star'] for k in cr]
check('corner spread (%)', 100 * (max(sv) - min(sv)) / cr['0.0']['s_star'], 1.6, 0.05)
rng('corner s* (rho>0)', [cr[k]['s_star'] for k in cr if k != '0.0'], 0.0849, 0.0857, 0.00005)
rng('corner d*/t', [cr[k]['d_over_t'] for k in cr], 5.09, 5.17, 0.005)
check('corner k rise at rho/t=4 (%)', 100 * (cr['4.0']['final']['leg']['k'] / cr['0.0']['final']['leg']['k'] - 1),
      4.3, 0.05)

# --- second family (Section 4.5) ---
ch = J('channel_crossover_v3.json')
ang = {0.6: J('at_crossover.json')['x1.3_r0.6_at100']['s_star'],
       0.5: J('at_crossover_r05.json')['x1.3_r0.5_at100']['s_star']}
for r, st in ((0.6, 14), (0.5, 16)):
    v = ch[f'U_x1.3_r{r}']
    check(f'channel U r={r} s*', v['s_star'], {0.6: 0.0747, 0.5: 0.0635}[r], 0.00005)
    check(f'channel U r={r} reduction vs angle (%)', 100 * (1 - v['s_star'] / ang[r]), st, 0.6)
    tr = v['root_troughs_labelled']
    lines.append(f"INFO  U r={r} root: " + ', '.join(f"k={t['k']:.3f} rhoL={t['rho_L']:.2f}" for t in tr))

# --- DSM (Section 4.6) ---
df = pd.read_csv(D / 'dsm_v3_strength.csv')
jumps, gains, switch = [], {0.6: [], 0.5: []}, {}
for (x, r, L, fy), u in df.groupby(['x', 'r', 'L_mm', 'fy']):
    u = u.sort_values('s').set_index('s')
    jumps.append(100 * (u.p_n_A.loc[0.01] / u.p_n_A.loc[0.0] - 1))
    gains[r].append(100 * (u.p_n_A.iloc[-1] / u.p_n_A.loc[0.0] - 1))
    ok = [s for s in u.index if s > 0 and (u.loc[s:].governing_A == 'local').all()]
    switch[(x, r, L, fy)] = min(ok) if ok else np.nan
rng('first-fold DSM change (%)', jumps, -30, 32, 0.6)
rng('DSM gain r=0.6 (%)', gains[0.6], 37, 59, 0.6)
rng('DSM gain r=0.5 (%)', gains[0.5], 28, 44, 0.6)
sw800 = [v for k, v in switch.items() if k[1] == 0.6 and k[2] == 800]
sw1200 = [v for k, v in switch.items() if k[1] == 0.6 and k[2] == 1200]
rng('DSM switch L=0.8 m, r=0.6', sw800, 0.13, 0.15, 1e-9)
rng('DSM switch L=1.2 m, r=0.6', sw1200, 0.11, 0.11, 1e-9)
u = df[(df.x == 1.3) & (df.r == 0.6) & (df.fy == 250)]
for L, st in ((800, 0.44), (1200, 0.40), (1800, 0.33), (2400, 0.25)):
    w = u[(u.L_mm == L) & (u.s == 0)]
    check(f'Pn/Py s=0 L={L}', float(w.p_n_A.iloc[0] / w.p_y.iloc[0]), st, 0.006)
w = u[(u.L_mm == 800)].set_index('s')
check('Pcrd/Pcrl at s=0.13 (L=0.8 m)', w.sig_d.loc[0.13] / w.sig_l_free.loc[0.13], 1.43, 0.006)
check('Pcrd/Pcrl at s=0.14 (L=0.8 m)', w.sig_d.loc[0.14] / w.sig_l_free.loc[0.14], 1.52, 0.006)
for x, sst in ((1.0, 0.0812), (1.3, 0.0871), (1.6, 0.0876)):
    for L in (800, 1200):
        vals = [switch[(x, 0.6, L, fy)] for fy in (250.0, 350.0, 450.0)]
        lines.append(f'INFO  DSM switch x={x} L={L}: {vals} -> +{100*(min(vals)/sst-1):.0f}% to +{100*(max(vals)/sst-1):.0f}% over s*')
vb, shift = 0, 0.0
for key, u in df.groupby(['x', 'r', 'L_mm', 'fy']):
    u = u.sort_values('s')
    if (u.governing_A.values != u.governing_B.values).any():
        vb += 1
        sw = []
        for col in ('governing_A', 'governing_B'):
            ok = [s for s in u.s if s > 0 and (u[u.s >= s][col] == 'local').all()]
            sw.append(min(ok) if ok else np.nan)
        shift = max(shift, abs(sw[1] - sw[0]))
lines.append(f'INFO  sequences where variant B differs from A: {vb} of {df.groupby(["x", "r", "L_mm", "fy"]).ngroups}')
check('largest A/B switch shift in s', shift, 0.01, 1e-9)
sw_pct = [100 * (switch[(x, 0.6, L, fy)] / sst - 1) for x, sst in ((1.0, 0.0812), (1.3, 0.0871), (1.6, 0.0876))
          for L in (800, 1200) for fy in (250.0, 350.0, 450.0)]
rng('DSM switch above elastic s* (%)', sw_pct, 26, 72, 0.6)

# --- Yao et al. (2022) geometries ---
yao = J('yao2022_check.json')
rl = [v['troughs_at_test'][0]['rho_L'] for v in yao.values()]
rng('Yao sections rho_L', rl, 1.03, 1.12, 0.006)
rng('Yao c/a', [v['r'] for v in yao.values()], 0.19, 0.33, 0.006)
v = yao['DLA6020_nominal']['troughs_at_test'][0]
sig = v['k'] * np.pi ** 2 * 210000 / (12 * 0.91) / 30.0 ** 2
check('DLA6020 elastic local / fy', sig / 403.0, 2.5, 0.06)

# --- GMNIA (Section 4.7, Table 4) ---
g = pd.read_csv(D / 'gmnia_results.csv')
g['tag'] = g.tag.fillna('')
base = g[(g.status == 'ok') & (g.h == 5.0) & (g.imp == 1.0) & (g.tag == '')]
lines.append(f'INFO  GMNIA cases ok/failed: {(g.status=="ok").sum()}/{(g.status=="failed").sum()}, '
             f'no post-peak: {(~base.post_peak.astype(bool)).sum()} of {len(base)} baseline')
p800 = base[(base.r == 0.6) & (base.L == 800) & (base.fy == 250)].set_index('s')
for s, pu, pn, pl in [(0.0, 0.222, 0.438, 0.85), (0.02, 0.223, 0.332, 0.88), (0.08, 0.297, 0.472, 0.93),
                     (0.14, 0.494, 0.583, 0.70), (0.16, 0.518, 0.584, 0.59)]:
    row = p800.loc[s]
    check(f'GMNIA Pu/Py s={s}', row.Pu_Py, pu, 0.001)
    check(f'GMNIA Pn/Py s={s}', row.Pn_Py, pn, 0.001)
    check(f'GMNIA P_lip s={s}', row.P_lip, pl, 0.005)
check('first-fold GMNIA change (%)', 100 * (p800.loc[0.02].Pu_Py / p800.loc[0.0].Pu_Py - 1), 0.5, 0.05)
fold = []
for (r_, L_, fy_), u in base.groupby(['r', 'L', 'fy']):
    u = u.set_index('s')
    if 0.0 in u.index and 0.02 in u.index:
        fold.append(100 * (u.loc[0.02].Pu_Py / u.loc[0.0].Pu_Py - 1))
allok = g[(g.status == 'ok') & (g.tag == '')]
fold2 = [100 * (u.set_index('s').loc[0.02].Pu_Py / u.set_index('s').loc[0.0].Pu_Py - 1)
         for _, u in allok.groupby(['r', 'L', 'fy', 'imp', 'h']) if {0.0, 0.02} <= set(u.s)]
rng('first-fold GMNIA change all pairs incl. imp variants (%)', fold2, -2.7, 4.3, 0.1)
check('stated bounds -3..5 cover all pairs', max(-min(fold2), max(fold2)) <= 5.0 and min(fold2) >= -3.0, 1, 1e-9)
check('Mixed baseline attained-load ratio mean (includes lower bounds)', base.ratio.mean(), 0.68, 0.005)
check('Mixed baseline attained-load ratio COV', base.ratio.std() / base.ratio.mean(), 0.18, 0.005)
rng('Pu/Pn at s=0', base[base.s == 0.0].ratio, 0.44, 0.65, 0.01)
check('Pu/Pn s=0.16 L800 fy250', p800.loc[0.16].ratio, 0.89, 0.005)
check('Pu/Pn s=0.14 L2400 fy250',
      base[(base.r == 0.6) & (base.L == 2400) & (base.fy == 250) & (base.s == 0.14)].ratio.iloc[0], 0.91, 0.005)
imp = g[(g.status == 'ok') & (g.r == 0.6) & (g.L == 800) & (g.fy == 250) & (g.h == 5.0) & (g.tag == '')].set_index(['imp', 's'])
check('imp x0.5 s=0.14', imp.loc[(0.5, 0.14)].Pu_Py, 0.525, 0.001)
check('imp x2.0 s=0.14', imp.loc[(2.0, 0.14)].Pu_Py, 0.433, 0.001)
check('imp x2 vs x1 change at s=0.14 (%)', 100 * (imp.loc[(2.0, 0.14)].Pu_Py / imp.loc[(1.0, 0.14)].Pu_Py - 1), -12, 0.5)
mesh = g[(g.status == 'ok') & (g.r == 0.6) & (g.L == 800) & (g.fy == 250) & (g.imp == 1.0) & (g.tag == '')].set_index(['h', 's'])
check('mesh fine s=0', mesh.loc[(2.5, 0.0)].Pu_Py, 0.220, 0.001)
check('mesh fine s=0.14', mesh.loc[(2.5, 0.14)].Pu_Py, 0.501, 0.001)
check('mesh effect s=0.14 (%)', abs(100 * (mesh.loc[(2.5, 0.14)].Pu_Py / mesh.loc[(5.0, 0.14)].Pu_Py - 1)), 1.4, 0.1)
lb = g[(g.status == 'ok') & (g.tag == 'labelbased')]
check('label-based s=0 Pu/Py', float(lb.Pu_Py.iloc[0]), 0.234, 0.001)
r5 = base[(base.r == 0.5) & (base.L == 800) & (base.fy == 250)].set_index('s')
rng('r=0.5 Pu/Pn range', r5.ratio, 0.54, 0.84, 0.005)
check('r=0.5 P_lip s=0.14', r5.loc[0.14].P_lip, 0.62, 0.005)
f450 = base[(base.r == 0.6) & (base.fy == 450)].ratio
rng('fy=450 Pu/Pn', f450, 0.44, 0.72, 0.005)

# R19c: check the censoring-aware statements against the actual 48-row campaign.
peak = base[base.post_peak.astype(bool)]
check('campaign count', len(g), 48, 0)
check('resolved campaign peaks', g.post_peak.astype(bool).sum(), 43, 0)
check('censored campaign count', (~g.post_peak.astype(bool)).sum(), 5, 0)
check('baseline count', len(base), 37, 0)
check('baseline resolved peaks', len(peak), 33, 0)
check('resolved baseline Pu/Pn mean', peak.ratio.mean(), 0.691, .0005)
check('resolved baseline Pu/Pn COV', peak.ratio.std()/peak.ratio.mean(), .178, .0005)
check('no baseline leg-dominated spatial state', bool((base.P_lip>.5).all()), 1, 0)
fold_peak=[]
for _,u in allok.groupby(['r','L','fy','imp','h']):
    if {0.,.02}<=set(u.s):
        u=u.set_index('s')
        if bool(u.loc[0.].post_peak) and bool(u.loc[.02].post_peak):
            fold_peak.append(100*(u.loc[.02].Pu_Py/u.loc[0.].Pu_Py-1))
check('resolved paired comparisons',len(fold_peak),6,0)
rng('resolved first-fold peak change (%)',fold_peak,.4,4.3,.05)
partial=g[~g.post_peak.astype(bool)]
check('censored doubled-imperfection short member',len(partial[(partial.L==800)&(partial.imp==2)&(partial.s==0)]),1,0)
check('no censored s=.06 L2400 member',len(partial[(partial.L==2400)&(partial.s==.06)]),0,0)
out = '\n'.join(lines) + f'\n\n{fails} FAIL(s)'
(ROOT / 'qa').mkdir(exist_ok=True)
(ROOT / 'qa' / 'qa_numbers.txt').write_text(out, encoding='utf-8')
print(out)
if fails:raise SystemExit(1)
