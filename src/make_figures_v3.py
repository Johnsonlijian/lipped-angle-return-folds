"""R19b figures built from the v3 evidence (regime map, threshold scaling, second family, DSM)."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r19env  # noqa: F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT / 'data', ROOT / 'figures'
C0, C1, C2, C3, C4 = '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00'
plt.rcParams.update({'font.size': 8.5, 'axes.titlesize': 8.5, 'axes.labelsize': 8.5,
                     'legend.fontsize': 7.2, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'lines.linewidth': 1.2, 'figure.dpi': 150, 'axes.spines.top': False,
                     'axes.spines.right': False})
J = lambda f: json.loads((DATA / f).read_text())  # noqa: E731


def pair_curves(scan):
    """(s, k_leg, k_lip) from a crossover scan log; NaN where a pair is absent."""
    out = []
    for e in scan:
        tr = e['troughs']
        if len(tr) >= 2:
            leg, lip = min(tr, key=lambda t: t['lips']), max(tr, key=lambda t: t['lips'])
            out.append((e['s'], leg['k'], lip['k']))
        elif len(tr) == 1:
            t = tr[0]
            out.append((e['s'], np.nan, t['k']) if t['lips'] > 0.5 else (e['s'], t['k'], np.nan))
    return np.array(out)


def fig_regime():
    d = J('regime_map_v3.json')
    rs = sorted({v['r'] for v in d.values()})
    ss = sorted({v['s'] for v in d.values()})
    P = np.array([[d[f'r{r:.3f}_s{s:.3f}']['governing']['lips'] for s in ss] for r in rs])
    K = np.array([[d[f'r{r:.3f}_s{s:.3f}']['governing']['k'] for s in ss] for r in rs])
    RL = np.array([[d[f'r{r:.3f}_s{s:.3f}']['governing']['rho_L'] for s in ss] for r in rs])
    N = np.array([[d[f'r{r:.3f}_s{s:.3f}']['n_minima'] for s in ss] for r in rs])
    fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.9), gridspec_kw=dict(width_ratios=[1.15, 1]))
    a = ax[0]
    ds, dr = 0.01, 0.025
    pm = a.pcolormesh(np.r_[np.array(ss) - ds, ss[-1] + ds], np.r_[np.array(rs) - dr, rs[-1] + dr], P,
                      cmap='cividis', vmin=0, vmax=1, shading='flat')
    cs = a.contour(ss, rs, K, levels=[3.0, 4.0, 4.6, 5.0], colors='w', linewidths=0.7)
    a.clabel(cs, fmt='k=%.1f', fontsize=6)
    for i, r in enumerate(rs):
        for j, s in enumerate(ss):
            if RL[i, j] > 1.5:
                a.plot(s, r, marker='x', color=C1, ms=4.5, mew=1.1)
            elif RL[i, j] > 1.05:
                a.plot(s, r, marker='.', color=C4, ms=4)
            if N[i, j] == 2:
                a.plot(s, r, marker='o', mfc='none', mec='w', ms=6, mew=0.6)
    for r, sst in ((0.5, 0.0760), (0.6, 0.0868)):
        a.plot(sst, r, marker='D', color=C3, ms=4.5, mec='k', mew=0.5)
    sx = np.linspace(0.005, 0.2, 200)
    for dt in (2, 4, 6):
        a.plot(sx, dt / (100 * sx), ls='--', color='0.85', lw=0.7)
        a.annotate(f'$d/t={dt}$', (dt / 55.0, 0.552), fontsize=6, color='w', ha='left', va='bottom',
                   rotation=-62)
    a.set_xlim(-ds, ss[-1] + ds)
    a.set_ylim(rs[0] - dr, rs[-1] + dr)
    a.set_xlabel(r'return ratio $s=d/c$')
    a.set_ylabel(r'lip ratio $r=c/a$')
    a.set_title('(a) lowest stationary minimum, $x=1.3$, $a/t=100$')
    cb = fig.colorbar(pm, ax=a, pad=0.02, fraction=0.05)
    cb.set_label(r'lip participation $P_\ell$')
    fig.legend(handles=[Line2D([], [], marker='x', ls='', color=C1, label=r'$\rho_L>1.5$ (distortional)'),
                        Line2D([], [], marker='.', ls='', color=C4, label=r'$1.05<\rho_L\leq1.5$ (coupled)'),
                        Line2D([], [], marker='o', ls='', mfc='none', mec='k', label='two stationary minima'),
                        Line2D([], [], marker='D', ls='', color=C3, mec='k', label=r'refined crossover $s_*$'),
                        Line2D([], [], ls='--', color='0.6', label=r'constant $d/t$')],
               loc='lower left', bbox_to_anchor=(0.06, 0.0), ncol=5, frameon=False, fontsize=6.2,
               handletextpad=0.3, columnspacing=1.0)
    b = ax[1]
    for r, c in ((0.2, C2), (0.4, C4), (0.5, C0), (0.6, C1)):
        i = rs.index(r)
        b.plot(ss, K[i], 'o-', color=c, ms=3, label=f'$r={r}$')
    b.axhline(5.19, color='0.5', ls=':', lw=0.8)
    b.annotate('leg-local level', (0.125, 5.32), fontsize=6.5, color='0.35')
    b.set_xlabel(r'return ratio $s=d/c$')
    b.set_ylabel(r'lowest stationary coefficient $k$')
    b.set_title('(b) the elastic gain saturates')
    b.legend(frameon=False, loc='lower right')
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / 'fig_regime.pdf')
    plt.close(fig)


def fig_scaling():
    d6, d5 = J('at_crossover.json'), J('at_crossover_r05.json')
    fig, ax = plt.subplots(1, 3, figsize=(6.9, 2.4))
    fits = {}
    for d, r, c, m in ((d6, 0.6, C1, 'o'), (d5, 0.5, C0, 's')):
        at = np.array(sorted(v['at'] for v in d.values()))
        ss = np.array([d[f'x1.3_r{r}_at{int(a)}']['s_star'] for a in at])
        dt = ss * r * at
        p = np.polyfit(np.log(at), np.log(ss), 1)
        q = np.polyfit(np.log(at), np.log(dt), 1)
        fits[r] = (p[0], q[0])
        ax[0].loglog(at, ss, m, color=c, ms=4, label=f'$r={r}$')
        xx = np.geomspace(45, 220, 50)
        ax[0].loglog(xx, np.exp(np.polyval(p, np.log(xx))), '-', color=c, lw=0.8)
        ax[1].semilogx(at, dt, m + '-', color=c, ms=4, label=f'$r={r}$')
    ax[0].set_xlabel(r'$a/t$')
    ax[0].set_ylabel(r'crossover return ratio $s_*=d_*/c$')
    ax[0].set_title('(a) $s_*$ falls with slenderness')
    ax[0].legend(frameon=False)
    ax[0].text(0.05, 0.06, '\n'.join(f'$r={r}$: $s_*\\propto(a/t)^{{{fits[r][0]:.2f}}}$' for r in fits),
               transform=ax[0].transAxes, fontsize=6.5)
    ax[1].set_xlabel(r'$a/t$')
    ax[1].set_ylabel(r'crossover return width $d_*/t$')
    ax[1].set_ylim(0, 7)
    ax[1].set_title('(b) $d_*/t$ nearly constant')
    ax[1].text(0.05, 0.06, '\n'.join(f'$r={r}$: $d_*/t\\propto(a/t)^{{{fits[r][1]:.2f}}}$' for r in fits),
               transform=ax[1].transAxes, fontsize=6.5)
    from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
    for a_ in ax[:2]:
        a_.xaxis.set_major_locator(FixedLocator([50, 75, 100, 150, 200]))
        a_.xaxis.set_major_formatter(FixedFormatter(['50', '75', '100', '150', '200']))
        a_.xaxis.set_minor_formatter(NullFormatter())
    ax[0].yaxis.set_major_locator(FixedLocator([0.04, 0.06, 0.08, 0.1, 0.15]))
    ax[0].yaxis.set_major_formatter(FixedFormatter(['0.04', '0.06', '0.08', '0.10', '0.15']))
    ax[0].yaxis.set_minor_formatter(NullFormatter())
    cr = J('corner_radius.json')
    rho = sorted(float(k) for k in cr)
    sv = [cr[f'{v:.1f}']['s_star'] for v in rho]
    ax[2].plot(rho, sv, 'o-', color=C2, ms=4)
    ax[2].set_ylim(0.083, 0.089)
    ax[2].set_xlabel(r'corner radius $\rho/t$')
    ax[2].set_ylabel(r'$s_*$ ($x=1.3$, $r=0.6$, $a/t=100$)')
    ax[2].set_title('(c) segmented-arc corners')
    fig.tight_layout()
    fig.savefig(FIG / 'fig_scaling.pdf')
    plt.close(fig)
    return fits


def fig_channel():
    ang = J('at_crossover.json')['x1.3_r0.6_at100']['scan']
    ang5 = J('at_crossover_r05.json')['x1.3_r0.5_at100']['scan']
    ch = J('channel_crossover_v3.json')
    fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.6))
    a = ax[0]
    for scan, c, lab in ((ang, C1, 'angle, $r=0.6$'), (ch['U_x1.3_r0.6']['scan'], C0, 'channel U, $r=0.6$'),
                         (ang5, C4, 'angle, $r=0.5$'), (ch['U_x1.3_r0.5']['scan'], C2, 'channel U, $r=0.5$')):
        pc = pair_curves(scan)
        a.plot(pc[:, 0], pc[:, 2], '-', color=c, label=lab)
        a.plot(pc[:, 0], pc[:, 1], '--', color=c, lw=0.9)
    for key, c in (('U_x1.3_r0.6', C0), ('U_x1.3_r0.5', C2)):
        a.axvline(ch[key]['s_star'], color=c, ls=':', lw=0.8)
    for key, d_, c in (('x1.3_r0.6_at100', 'at_crossover.json', C1), ('x1.3_r0.5_at100', 'at_crossover_r05.json', C4)):
        a.axvline(J(d_)[key]['s_star'], color=c, ls=':', lw=0.8)
    a.set_xlim(0, 0.095)
    a.set_ylim(2, 6.5)
    a.set_xlabel(r'return ratio $s=d/c$')
    a.set_ylabel(r'stationary coefficient $k_a$')
    a.set_title('(a) lip-return (solid) and leg/web-local (dashed) troughs')
    a.legend(frameon=False, loc='lower right', fontsize=6.2)
    b = ax[1]
    c25 = J('channel_crossover_c25.json')
    for fam, src, c, m in (('$b/t=50$', ch, C0, 'o'), ('$b/t=25$', c25, C1, 's')):
        rr, kl, kd = [], [], []
        for k, v in src.items():
            if not k.startswith('C'):
                continue
            tr = v['s0_troughs_labelled']
            loc = [t['k'] for t in tr if t['lips'] < 0.2]
            dis = [t['k'] for t in tr if t['lips'] >= 0.2]
            rr.append(v['r'])
            kl.append(min(loc) if loc else np.nan)
            kd.append(min(dis) if dis else np.nan)
        o = np.argsort(rr)
        rr, kl, kd = np.array(rr)[o], np.array(kl)[o], np.array(kd)[o]
        b.plot(rr, kl, m + '--', color=c, ms=3.5, mfc='none', label=f'{fam}: web-local')
        b.plot(rr, kd, m + '-', color=c, ms=3.5, label=f'{fam}: flange-lip')
    b.set_xlabel(r'lip ratio $c/b$ ($h=2b$, $s=0$)')
    b.set_ylabel(r'stationary coefficient $k_b$')
    b.set_title('(b) conventional lipped channel at $s=0$')
    b.legend(frameon=False, fontsize=6.2, loc='center right')
    fig.tight_layout()
    fig.savefig(FIG / 'fig_channel.pdf')
    plt.close(fig)


def fig_dsm():
    df = pd.read_csv(DATA / 'dsm_v3_strength.csv')
    fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.7))
    a = ax[0]
    t = df[(df.x == 1.3) & (df.r == 0.6) & (df.fy == 250)]
    for L, c in ((800, C0), (1200, C2), (1800, C4), (2400, C1)):
        u = t[t.L_mm == L].sort_values('s')
        y = u.p_n_A / u.p_y
        a.plot(u.s, y, '-', color=c, lw=1.0, label=f'$L={L}$ mm')
        for g, m in (('local', 'o'), ('distortional', '^'), ('global', 's')):
            w = u.governing_A == g
            a.plot(u.s[w], y[w], m, color=c, ms=3.2, mfc='w' if g == 'distortional' else c)
    a.axvline(0.0868, color='0.4', ls=':', lw=0.9)
    a.annotate('elastic $s_*$', (0.0885, 0.595), fontsize=6.5, color='0.35', va='top')
    a.set_xlabel(r'return ratio $s=d/c$')
    a.set_ylabel(r'$P_n/P_y$')
    a.set_title(r'(a) DSM strength, $x=1.3$, $r=0.6$, $f_y=250$ MPa')
    h = a.get_legend_handles_labels()[0][:4]
    a.legend(handles=[*h, Line2D([], [], marker='o', ls='', color='0.3', label='local governs'),
                      Line2D([], [], marker='^', ls='', mfc='w', color='0.3', label='distortional governs')],
             frameon=False, fontsize=6.0, loc='upper left', ncol=2, bbox_to_anchor=(0.0, 1.0))
    a.set_ylim(0.22, 0.75)
    b = ax[1]
    for x, m in ((1.0, 's'), (1.3, 'o'), (1.6, '^')):
        for fy, c in ((250, C0), (350, C2), (450, C1)):
            u = df[(df.x == x) & (df.r == 0.6) & (df.fy == fy)]
            Ls, sl = [], []
            for L, g in u.groupby('L_mm'):
                g = g.sort_values('s')
                ok = [s for s in g.s if s > 0 and (g[g.s >= s].governing_A == 'local').all()]
                Ls.append(L / 1000)
                sl.append(min(ok) if ok else np.nan)
            b.plot(np.array(Ls) + (x - 1.3) * 0.12, sl, m, color=c, ms=3.5,
                   label=f'$f_y={fy}$' if x == 1.3 else None)
    b.axhspan(0.0812, 0.0876, color=C3, alpha=0.18)
    b.annotate(r'elastic $s_*$, $x=1.0$-$1.6$', (1.45, 0.091), fontsize=6.5, color=C3)
    b.set_xlabel('member length $L$ (m)')
    b.set_ylabel('return ratio beyond which\nlocal strength governs')
    b.set_title('(b) DSM switch vs elastic crossover ($r=0.6$)')
    b.legend(frameon=False, fontsize=6.2, loc='upper right', title='squares/circles/triangles:\n$x=1.0/1.3/1.6$',
             title_fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / 'fig_dsm.pdf')
    plt.close(fig)


if __name__ == '__main__':
    fig_regime()
    print('regime ok')
    print('scaling fits', fig_scaling())
    fig_channel()
    print('channel ok')
    fig_dsm()
    print('dsm ok')
