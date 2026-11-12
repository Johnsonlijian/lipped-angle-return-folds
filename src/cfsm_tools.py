"""R19b constrained-FSM tools: full-class pure-mode solves and G/D/L/O modal participation.

Why this module exists (independent-review finding, 2026-11-11):
  * cfsm_labels.py passed one-element class lists (e.g. local=[1]); pycufsm's
    mode_select then keeps only the FIRST base vector of the class.  With the
    modal (orth=2) basis this may or may not equal the class minimum, so the
    full class space is re-solved here and compared.
  * pycufsm 0.2.0 classify()/mode_class() are a literal MATLAB port with
    broken indexing (tuple slicing, 1-based offsets) and cannot run.  The
    participation below re-implements the CUFSM definition: express the
    eigenvector in the cFSM base (columns ordered G, D, L, O), take the L2
    norm of each class's coefficient block, normalise the four norms to 100.
"""
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r19env  # noqa: F401
import numpy as np  # noqa: E402
import research_scan as fsm  # noqa: E402
from pycufsm.pre import cutwp  # noqa: E402
from pycufsm.solve import cfsm  # noqa: E402
from pycufsm.solve.analysis import analysis  # noqa: E402

CLASSES = ('glob', 'dist', 'local', 'other')


def sect_props(nodes, elems):
    return cutwp.prop2(nodes[:, 1:3], elems[:, 1:4])


def class_counts(nodes, elems):
    """Number of cFSM base vectors per class (G, D, L, O) for one longitudinal term."""
    nb = deepcopy(nodes)
    nb[:, 7] = 1.0
    (main_nodes, meta_elements, node_props, n_main, n_corner, n_sub,
     n_dist, n_local, dof_perm) = cfsm.base_properties(nodes=nb, elements=elems)
    r_x, r_z, r_yd, r_ys, r_ud = cfsm.mode_constr(
        nodes=nb, elements=elems, node_props=node_props,
        main_nodes=main_nodes, meta_elements=meta_elements)
    d_y, n_glob = cfsm.y_dofs(
        nodes=nb, elements=elems, main_nodes=main_nodes, n_main_nodes=n_main,
        n_dist_modes=n_dist, r_yd=r_yd, r_ud=r_ud,
        sect_props=sect_props(nodes, elems),
        el_props=analysis.elem_prop(nodes=nodes, elements=elems))
    n_glob, n_dist, n_local = int(n_glob), int(n_dist), int(n_local)
    return dict(glob=n_glob, dist=n_dist, local=n_local,
                other=4 * len(nodes) - n_glob - n_dist - n_local)


def free_eig(nodes, elems, L_mm, n_eigs=4):
    sig, curve, shapes = fsm.solver.signature(nodes, elems, np.array([L_mm]))
    return np.atleast_2d(np.asarray(curve))[0], np.asarray(shapes)[0]


def pure_class_eig(nodes, elems, L_mm, cls, first_only=False, n_eigs=4,
                   orth=2, norm=1, counts=None):
    """Eigenvalues with the deformation restricted to one cFSM class.

    The vendored strip() sizes the reduced basis by sum(list), so each class
    list must have exactly the class dimension (no padding)."""
    counts = counts or class_counts(nodes, elems)
    gbt = {}
    for k in CLASSES:
        n = counts[k]
        if k != cls:
            gbt[k] = [0] * n
        elif first_only:
            gbt[k] = [1] + [0] * (n - 1)
        else:
            gbt[k] = [1] * n
    gbt.update(o_space=1, couple=1, orth=orth, norm=norm)
    sig, curve, shapes = fsm.solver._fsm.strip(
        props=fsm.solver.PROPS, nodes=nodes, elements=elems,
        lengths=np.array([L_mm]), springs=np.array([]),
        constraints=np.array([]), GBT_con=gbt, B_C='S-S',
        m_all=np.ones((1, 1)), n_eigs=n_eigs,
        sect_props=sect_props(nodes, elems))
    return np.atleast_2d(np.asarray(curve))[0], np.asarray(shapes)[0]


def gdlo_basis(nodes, elems, L_mm, orth=2, norm=1):
    """cFSM base vectors (columns G|D|L|O) for one half-wave, single term m=1."""
    props = fsm.solver.PROPS
    m_a = np.array([1.0])
    sp = sect_props(nodes, elems)
    el_props = analysis.elem_prop(nodes=nodes, elements=elems)
    nb = deepcopy(nodes)
    nb[:, 7] = 1.0
    (main_nodes, meta_elements, node_props, n_main, n_corner, n_sub,
     n_dist, n_local, dof_perm) = cfsm.base_properties(nodes=nb, elements=elems)
    r_x, r_z, r_yd, r_ys, r_ud = cfsm.mode_constr(
        nodes=nb, elements=elems, node_props=node_props,
        main_nodes=main_nodes, meta_elements=meta_elements)
    d_y, n_glob = cfsm.y_dofs(
        nodes=nb, elements=elems, main_nodes=main_nodes, n_main_nodes=n_main,
        n_dist_modes=n_dist, r_yd=r_yd, r_ud=r_ud, sect_props=sp,
        el_props=el_props)
    b_v_l = cfsm.base_column(
        nodes_base=nb, elements=elems, props=props, length=L_mm, B_C='S-S',
        m_a=m_a, el_props=el_props, node_props=node_props, n_main_nodes=n_main,
        n_corner_nodes=n_corner, n_sub_nodes=n_sub, n_global_modes=n_glob,
        n_dist_modes=n_dist, n_local_modes=n_local, dof_perm=dof_perm,
        r_x=r_x, r_z=r_z, r_ys=r_ys, d_y=d_y)
    # pycufsm's norm=2/3 branches are broken (MATLAB right-division ported as
    # lstsq on 1-D arrays), so orthogonalise with norm=0 and normalise here.
    gbt = dict(glob=[0], dist=[0], local=[0], other=[0], o_space=1, couple=1,
               orth=orth, norm=0)
    b_v = np.asarray(cfsm.base_update(
        GBT_con=gbt, b_v_l=b_v_l, length=L_mm, m_a=m_a, nodes=nodes,
        elements=elems, props=props, n_global_modes=n_glob,
        n_dist_modes=n_dist, n_local_modes=n_local, B_C='S-S',
        el_props=el_props), dtype=float)
    if norm in (2, 3):
        K, Kg = analysis.k_kg_global(nodes=nb, elements=elems, el_props=el_props,
                                     props=props, length=L_mm, B_C='S-S',
                                     m_a=[1])
        M = np.asarray(K if norm == 2 else Kg, dtype=float)
        scale = np.sqrt(np.abs(np.einsum('ij,ik,kj->j', b_v, M, b_v)))
    else:
        scale = np.linalg.norm(b_v, axis=0)
    scale[scale < 1e-300] = 1.0
    return b_v / scale, (int(n_glob), int(n_dist), int(n_local))


def participation_gdlo(b_v, counts, phi):
    """CUFSM modal participation (percent) of eigenvector phi in G, D, L, O."""
    ng, nd, nl = counts
    coef = np.linalg.lstsq(b_v, np.asarray(phi, float), rcond=None)[0]
    cuts = [0, ng, ng + nd, ng + nd + nl, len(coef)]
    norms = np.array([np.linalg.norm(coef[cuts[i]:cuts[i + 1]]) for i in range(4)])
    return 100.0 * norms / norms.sum()


def classify_free_mode(nodes, elems, L_mm, norms=(1, 2, 3), mode=0):
    """Free eigenpair at L plus its G/D/L/O participation under several norms."""
    eigs, shapes = free_eig(nodes, elems, L_mm)
    phi = shapes[mode]
    out = dict(sigma=float(eigs[mode]))
    for nrm in norms:
        b_v, counts = gdlo_basis(nodes, elems, L_mm, orth=2, norm=nrm)
        out[f'gdlo_norm{nrm}'] = participation_gdlo(b_v, counts, phi).tolist()
    out['counts'] = counts
    return out, phi


if __name__ == '__main__':
    import time
    import r19lib
    kf = r19lib.kfactor(100)
    nodes, elems, tags, a = fsm.build(1.3, 0.6, 0.0871, nmesh=24)
    t0 = time.time()
    free_eig(nodes, elems, 0.842 * a)
    print('free solve s:', round(time.time() - t0, 3))
    for La in (0.842, 2.875):
        L = La * a
        ev, _ = free_eig(nodes, elems, L)
        print(f'\nL/a={La}: free k = {ev[0]*kf:.4f}')
        for cls in ('glob', 'dist', 'local'):
            t0 = time.time()
            e1, _ = pure_class_eig(nodes, elems, L, cls, first_only=True)
            t1 = time.time()
            eA, _ = pure_class_eig(nodes, elems, L, cls, first_only=False)
            t2 = time.time()
            print(f'  {cls:5s} first-vector k={e1[0]*kf:9.4f} ({t1-t0:.2f}s)  '
                  f'full-class k={eA[0]*kf:9.4f} ({t2-t1:.2f}s)')
        t0 = time.time()
        res, phi = classify_free_mode(nodes, elems, L)
        print('  GDLO % (vector, strain-energy, work):',
              [np.round(res[f"gdlo_norm{n}"], 1).tolist() for n in (1, 2, 3)],
              'counts', res['counts'], f'({time.time()-t0:.2f}s)')
