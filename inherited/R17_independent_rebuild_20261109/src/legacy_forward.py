"""R17 — independent forward finite-strip scan of the lipped angle.

Independent re-implementation of the r03 reconstruction contract in
pycufsm 0.2.0 (numpy<2 env). Geometry replicates the frozen spec
(external_sources/zhang_reconstruction/r03_zhang_cufsm_run.m):

  simple edge (Table 4):  P = [c,b; 0,b; 0,0; a,0; a,c]
  complex edge (Table 8): P = [c,b-d; c,b; 0,b; 0,0; a,0; a,c; a-d,c], d = c/2

Both legs carry an inward lip c at the tip (a = long leg, b = short leg).
Uniform unit compression on all nodes; signature curve over
a*logspace(log10(0.30), log10(1.60), 32); first eigenvalue minimum with
quadratic interpolation in log wavelength -> sigma_cr, L_cr.

pycufsm notes:
  * node row = [id, x, y, dof_x, dof_y, dof_z, dof_r, stress]
    global DOF order = [u_j=2j, v_j=2j+1 (first 2n) | w_j=2n+2j,
    th_j=2n+2j+1 (last 2n)]: dof_x=u, dof_z=v_axial, dof_y=w, dof_r=th.
  * 1 = free, 0 = restrained. No restraints needed: the open section is
    self-supporting via shared corner/junction nodes.
  * venv library patched: cfsm.constr_user off-by-one dropped the last
    kept DOF and left a stale identity column that silently re-freed the
    final restrained DOF (upstream bug in pycufsm 0.2.0; documented in
    the package reproducibility note).

Stage 1: reproduce all 630 published cells -> RMSE vs source table.
Stage 2: fine c/a scan with L_cr and mode-participation tracking.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import r19env
import pycufsm
import pycufsm.fsm as _fsm
import pycufsm.solve.analysis_p as _analysis_p

# pycufsm 0.2.0's compiled analysis_c crashes on int(ndarray) in this
# numpy version; use the bundled pure-Python solver (identical math).
_fsm.analysis = _analysis_p

E_MPA = 217400.0
NU = 0.30
T_MM = 2.0
MESH_TARGET_MM = 4.0
N_LENGTHS = 32
LENGTH_FACTOR = (0.30, 1.60)
N_EIGS = 4

GBT = {"glob": [0], "dist": [0], "local": [0], "other": [0],
       "o_space": 1, "couple": 1, "orth": 2, "norm": 1}
PROPS = np.array([[100.0, E_MPA, E_MPA, NU, NU, E_MPA / (2.0 * (1.0 + NU))]])

ROUND = Path(__file__).resolve().parent
OUT = ROUND


def section_polyline(a, b, c, d, is_complex):
    """Return the node polyline (list of (x, y)) replicating r03_make_model."""
    if is_complex:
        P = [(c, b - d), (c, b), (0.0, b), (0.0, 0.0), (a, 0.0), (a, c), (a - d, c)]
    else:
        P = [(c, b), (0.0, b), (0.0, 0.0), (a, 0.0), (a, c)]
    return P


def segment_key_points(a, b, c, d, is_complex):
    """Anchor coordinates marking segment boundaries, for mode attribution."""
    P = section_polyline(a, b, c, d, is_complex)
    return P


def build_section(a_over_t, a_over_b, c_over_a, is_complex, target=MESH_TARGET_MM):
    a = T_MM * a_over_t
    b = a / a_over_b
    c = a * c_over_a
    d = 0.5 * c
    P = section_polyline(a, b, c, d, is_complex)
    coords = []
    seg_of_node = []  # which polyline segment each node belongs to
    for s in range(len(P) - 1):
        p0 = np.array(P[s], dtype=float)
        p1 = np.array(P[s + 1], dtype=float)
        n = max(1, int(np.ceil(np.linalg.norm(p1 - p0) / target)))
        for j in range(n):
            coords.append(p0 + (p1 - p0) * (j / n))
            seg_of_node.append(s)
    coords.append(np.array(P[-1], dtype=float))
    seg_of_node.append(len(P) - 2)
    nn = len(coords)
    nodes = np.zeros((nn, 8))
    nodes[:, 0] = np.arange(nn)
    nodes[:, 1] = [pt[0] for pt in coords]
    nodes[:, 2] = [pt[1] for pt in coords]
    nodes[:, 3:7] = 1.0  # all DOFs free
    nodes[:, 7] = 1.0    # unit compressive stress
    elems = np.zeros((nn - 1, 5))
    elems[:, 0] = np.arange(nn - 1)
    elems[:, 1] = np.arange(nn - 1)
    elems[:, 2] = np.arange(1, nn)
    elems[:, 3] = T_MM
    elems[:, 4] = 100.0
    return nodes, elems, np.array(seg_of_node)


def signature(nodes, elems, lengths):
    m_all = np.ones((len(lengths), 1))
    sig, curve, shapes = _fsm.strip(
        props=PROPS, nodes=nodes, elements=elems, lengths=lengths,
        springs=np.array([]), constraints=np.array([]), GBT_con=GBT,
        B_C="S-S", m_all=m_all, n_eigs=N_EIGS, sect_props=None,
    )
    return np.asarray(sig, dtype=float), np.asarray(curve), np.asarray(shapes)


def refined_minimum(lengths, first):
    """Minimum of first eigenvalue + quadratic interpolation in log L."""
    first = np.asarray(first, dtype=float)
    ix = int(np.argmin(first))
    sig_star, L_star = first[ix], lengths[ix]
    if 0 < ix < len(first) - 1:
        xx = np.log(lengths[ix - 1 : ix + 2])
        yy = first[ix - 1 : ix + 2]
        pp = np.polyfit(xx, yy, 2)
        if pp[0] > 0:
            x0 = -pp[1] / (2 * pp[0])
            if xx[0] <= x0 <= xx[2]:
                sig_star = float(np.polyval(pp, x0))
                L_star = float(np.exp(x0))
    return sig_star, L_star, ix


SEG_NAMES_SIMPLE = ["lip_b", "leg_b", "leg_a", "lip_a"]
SEG_NAMES_COMPLEX = ["lip_b_d", "lip_b", "leg_b", "leg_a", "lip_a", "lip_a_d"]


def mode_participation(shapes_row, seg_of_node, n_nodes):
    """w-DOF energy share per polyline segment for the first mode."""
    vec = np.asarray(shapes_row)[0]  # first mode
    w = vec[2 * n_nodes : 4 * n_nodes : 2]
    th = vec[2 * n_nodes + 1 : 4 * n_nodes : 2]
    amp = w ** 2 + th ** 2
    names = SEG_NAMES_SIMPLE if len(np.unique(seg_of_node)) <= 4 else SEG_NAMES_COMPLEX
    share = {}
    total = amp.sum()
    for s in np.unique(seg_of_node):
        share[names[int(s)]] = float(amp[seg_of_node == s].sum() / total) if total > 0 else 0.0
    return share


def run_case(a_over_t, a_over_b, c_over_a, is_complex, want_shapes=False):
    nodes, elems, seg = build_section(a_over_t, a_over_b, c_over_a, is_complex)
    a = T_MM * a_over_t
    lengths = a * np.logspace(np.log10(LENGTH_FACTOR[0]), np.log10(LENGTH_FACTOR[1]), N_LENGTHS)
    sig, curve, shapes = signature(nodes, elems, lengths)
    sig_star, L_star, ix = refined_minimum(lengths, sig)
    out = {
        "sigma_cr": sig_star,
        "L_cr": L_star,
        "ix_min": ix,
        "node_count": len(nodes),
        "element_count": len(elems),
        "min_at_edge": ix in (0, len(lengths) - 1),
    }
    if want_shapes:
        out["participation"] = mode_participation(
            shapes[ix], seg, len(nodes)
        )
        out["sig_curve"] = sig.tolist()
    return out


def stage_reproduce(df_cells, path_csv):
    rows = []
    t0 = time.time()
    hdr = not Path(path_csv).exists()
    for i, r in df_cells.iterrows():
        res = run_case(r.a_over_t, r.a_over_b, r.c_over_a, bool(r.is_complex))
        err = (res["sigma_cr"] / r.source_stress_mpa - 1.0) * 100.0
        rec = {
            "case_id": r.case_id, "table": r.table, "is_complex": r.is_complex,
            "a_over_t": r.a_over_t, "a_over_b": r.a_over_b, "c_over_a": r.c_over_a,
            "source_stress_mpa": r.source_stress_mpa,
            "pycufsm_stress_mpa": res["sigma_cr"],
            "pycufsm_L_cr_mm": res["L_cr"],
            "matlab_L_cr_mm": r.length_at_min_mm,
            "node_count": res["node_count"],
            "relative_error_pct": err,
            "min_at_edge": res["min_at_edge"],
        }
        rows.append(rec)
        pd.DataFrame([rec]).to_csv(path_csv, mode="a", header=hdr, index=False)
        hdr = False
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(df_cells)}  elapsed {time.time()-t0:.0f}s", flush=True)
    out = pd.DataFrame(rows)
    return out


def stage_ca_scan(ca_values, at_values=(70.0, 110.0), ab=1.3):
    rows = []
    for is_complex, tbl in [(0, "Table 4"), (1, "Table 8")]:
        for at in at_values:
            for ca in ca_values:
                res = run_case(at, ab, ca, bool(is_complex), want_shapes=True)
                part = res.pop("participation")
                curve = res.pop("sig_curve")
                row = {
                    "table": tbl, "is_complex": is_complex, "a_over_t": at,
                    "a_over_b": ab, "c_over_a": ca,
                    "sigma_cr": res["sigma_cr"], "L_cr": res["L_cr"],
                    "ix_min": res["ix_min"], "min_at_edge": res["min_at_edge"],
                    **{f"part_{k}": v for k, v in part.items()},
                }
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "reproduce"
    if mode == "reproduce":
        cells = pd.read_csv(OUT / "r15_holdout_row_classification.csv")
        df = stage_reproduce(cells, OUT / "r17_pycufsm_reproduction.csv")
        ae = df["relative_error_pct"].abs()
        summary = {
            "stage": "reproduce_630_cells",
            "solver": "pycufsm 0.2.0 (patched constr_user + uniform assembly), numpy<2 venv",
            "contract": "r03 frozen: E=217400 nu=0.30 t=2mm mesh=4mm 32 log L in [0.30a,1.60a]",
            "rows": int(len(df)),
            "rmse_pct": float(np.sqrt(np.mean(df["relative_error_pct"] ** 2))),
            "mean_bias_pct": float(df["relative_error_pct"].mean()),
            "q95_abs_pct": float(np.quantile(ae, 0.95)),
            "max_abs_pct": float(ae.max()),
            "within_5pct": float((ae <= 5.0).mean()),
            "min_at_edge_rows": int(df["min_at_edge"].sum()),
        }
        by = df.groupby("table")["relative_error_pct"]
        for tbl, g in by:
            summary[f"{tbl}_rmse_pct"] = float(np.sqrt(np.mean(g ** 2)))
            summary[f"{tbl}_within5"] = float((g.abs() <= 5.0).mean())
        Path(OUT / "r17_pycufsm_reproduction_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    elif mode == "reproduce_shard":
        idx, n_shards = int(sys.argv[2]), int(sys.argv[3])
        part = OUT / f"r17_pycufsm_reproduction_part{idx:02d}.csv"
        cells = pd.read_csv(OUT / "r15_holdout_row_classification.csv")
        shard = cells.iloc[idx::n_shards]
        if part.exists():
            done = pd.read_csv(part)["case_id"].tolist()
            shard = shard[~shard["case_id"].isin(done)]
            print(f"shard {idx}: resuming, {len(done)} done, {len(shard)} left", flush=True)
        df = stage_reproduce(shard, part)
        print(f"shard {idx}/{n_shards}: {len(df)} rows done", flush=True)
    elif mode == "scan":
        ca = np.round(np.arange(0.15, 0.601, 0.01), 3)
        df = stage_ca_scan(ca)
        df.to_csv(OUT / "r17_pycufsm_ca_scan.csv", index=False)
        print(f"scan rows: {len(df)}")
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
