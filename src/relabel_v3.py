"""R19b: re-identify stored free troughs with the v3 protocol (r19b_lib.identify).

The free stationary minima, crossover roots and participation values in
regime_map.json, channel_crossover.json and dsm_v2_elastic.json come from the free
FSM and remain valid; only their G/D/L/O labels (vendored cFSM bases) are replaced.
Outputs: regime_map_v3.json, channel_crossover_v3.json, dsm_v3_elastic.json.
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r19env  # noqa: F401
import research_scan as fsm  # noqa: E402
import cfsm_tools as ct  # noqa: E402
import r19b_lib as lb  # noqa: E402
from channel_family import build_channel  # noqa: E402
from channel_crossover import build_std_channel  # noqa: E402

OLD = ('G', 'D', 'L', 'O', 'dominant')


def relabel(tr, nodes, elems, a):
    counts = ct.class_counts(nodes, elems)
    for t in tr:
        for k in OLD:
            t.pop(k, None)
        sigma, phi = lb.eig1(nodes, elems, t['L_over_a'] * a)
        t.update(lb.identify(nodes, elems, a, t['L_over_a'], sigma, phi, counts))
    return tr


if __name__ == '__main__':
    what = sys.argv[1]
    t0 = time.time()
    if what == 'regime':
        d = json.loads((lb.DATA / 'regime_map.json').read_text())
        for key, v in d.items():
            nodes, elems, tags, a = fsm.build(1.3, v['r'], v['s'], nmesh=24, at=100)
            relabel(v['troughs'], nodes, elems, a)
            v['governing'] = min(v['troughs'], key=lambda t: t['k']) if v['troughs'] else None
            print(key, [(round(t['L_over_a'], 2), round(t['k'], 3), t['cls'], round(t['rho_L'], 2))
                        for t in v['troughs']], f'{time.time()-t0:.0f}s', flush=True)
        (lb.DATA / 'regime_map_v3.json').write_text(json.dumps(d, indent=1))
    elif what == 'channel':
        d = json.loads((lb.DATA / 'channel_crossover.json').read_text())
        for key, v in d.items():
            if v['family'] == 'U':
                build = lambda s, v=v: build_channel(v['x'], v['r'], float(s), nmesh=24, at=100)  # noqa: E731
                at, lo, hi = 100.0, 0.3, 12.0
            else:
                build = lambda s, v=v: build_std_channel(v['r'], float(s))  # noqa: E731
                at, lo, hi = 50.0, 0.3, 20.0
            for tag, s in (('s0', 0.0), ('root', v.get('s_star'))):
                if s is None:
                    continue
                nodes, elems, tags, a = build(s)
                tr, _ = lb.troughs_id(nodes, elems, tags, a, at, lo, hi, 80)
                v[f'{tag}_troughs_labelled'] = tr
            print(key, v['status'], v.get('s_star'),
                  [(round(t['L_over_a'], 2), round(t['k'], 3), t['cls'], round(t['rho_L'], 2),
                    round(t['rho_FT'], 1)) for t in v.get('root_troughs_labelled', v['s0_troughs_labelled'])],
                  f'{time.time()-t0:.0f}s', flush=True)
        (lb.DATA / 'channel_crossover_v3.json').write_text(json.dumps(d, indent=1))
    elif what == 'dsm':
        d = json.loads((lb.DATA / 'dsm_v2_elastic.json').read_text())
        for key, e in d.items():
            nodes, elems, tags, a = fsm.build(e['x'], e['r'], e['s'], nmesh=24, at=100)
            relabel(e['troughs'], nodes, elems, a)
            sp = ct.sect_props(nodes, elems)
            e['ft'] = {L: lb.classical_ft(sp, float(L)) for L in e['glob']}
            e.pop('pure_dist', None)  # unvalidated distortional subspace
            for L in e['glob']:
                e['glob'][L].pop('pure', None)
                e['glob'][L].pop('G_free', None)
                e['glob'][L].pop('free_dominant', None)
            print(key, [(round(t['L_over_a'], 2), round(t['sigma'], 2), t['cls']) for t in e['troughs']],
                  'FT800', round(e['ft']['800']['FT'], 1), f'{time.time()-t0:.0f}s', flush=True)
        (lb.DATA / 'dsm_v3_elastic.json').write_text(json.dumps(d, indent=1))
