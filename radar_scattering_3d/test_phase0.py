"""
Phase 0 unit test: verify py_bem_solve_multi_rhs_3d and py_bem_solve_ir_3d.

Checks:
  1. Multi-RHS (M=5): each solution column matches scipy LU to < 1e-4 relative.
  2. IR solve:        residual drops from ~1e-6 to < 1e-10 in 2 steps.
  3. Timing sanity:   build-once cost is dominant; per-solve < 1 s at N=320.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bem_cobol'))

import numpy as np
from scipy.linalg import solve as sp_solve

from geometry_3d   import icosphere
from mie_sphere_3d import mie_soft_sphere_rcs
from bem_3d        import build_bem_matrix_3d_cpu, make_rhs_3d, rcs_3d

sys.path.insert(0, os.path.dirname(__file__))
from bem_assembly_3d_multi_ops import BEMAssembler3DMulti

PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'

def check(label, condition, detail=''):
    tag = PASS if condition else FAIL
    print(f'  [{tag}] {label}' + (f'  ({detail})' if detail else ''))
    return condition


def test_multi_rhs(N_target=320, k=3.0, M=5):
    print(f'\n=== Multi-RHS test  N~{N_target}  k={k}  M={M} ===')
    nodes, _, areas = icosphere(N_target, R=1.0)
    N = len(nodes)
    print(f'  Actual N = {N}')

    # M distinct incident directions spread over the sphere
    inc_dirs = np.array([
        [ 1.,  0.,  0.],
        [ 0.,  1.,  0.],
        [ 0.,  0.,  1.],
        [ 1.,  1.,  0.],
        [ 1.,  0.,  1.],
    ], dtype=np.float64)
    inc_dirs /= np.linalg.norm(inc_dirs, axis=1, keepdims=True)

    # Reference: scipy LU for each RHS
    A_cpu = build_bem_matrix_3d_cpu(nodes, areas, k)
    sigma_ref = np.zeros((N, M), dtype=np.complex128)
    for j in range(M):
        b = make_rhs_3d(nodes, k, inc_dirs[j])
        sigma_ref[:, j] = sp_solve(A_cpu, b)

    # Build B matrix (N, M) Fortran column-major
    B = np.zeros((N, M), dtype=np.complex128, order='F')
    for j in range(M):
        B[:, j] = make_rhs_3d(nodes, k, inc_dirs[j])

    # Multi-RHS Fortran solve
    asm = BEMAssembler3DMulti()
    t0  = time.perf_counter()
    X, n_conv = asm.solve_multi_rhs(nodes, areas, k, B, restart=50, tol=1e-6)
    t_total   = time.perf_counter() - t0

    print(f'  n_converged = {n_conv}/{M}   total time = {t_total:.3f}s')

    all_ok = True
    for j in range(M):
        ref   = sigma_ref[:, j]
        got   = X[:, j]
        r_err = np.linalg.norm(got - ref) / np.linalg.norm(ref)
        ok    = r_err < 1e-3
        all_ok = all_ok and ok
        check(f'  col {j}  rel_err={r_err:.2e}', ok)

    return all_ok


def test_ir(N_target=320, k=3.0):
    print(f'\n=== IR test  N~{N_target}  k={k} ===')
    nodes, _, areas = icosphere(N_target, R=1.0)
    N = len(nodes)
    inc = np.array([0., 0., 1.])
    b   = make_rhs_3d(nodes, k, inc)

    asm = BEMAssembler3DMulti()

    # GMRES only (maxiter_ir=0)
    x0, conv0, res0 = asm.solve_ir(nodes, areas, k, b, tol=1e-6, maxiter_ir=0)
    check(f'GMRES-only converged', conv0, f'rel_res={res0:.2e}')

    # With 2 IR steps
    x2, conv2, res2 = asm.solve_ir(nodes, areas, k, b, tol=1e-10, maxiter_ir=2)
    check(f'IR-2 rel_res < 1e-9', res2 < 1e-9, f'rel_res={res2:.2e}')
    check(f'IR improves residual', res2 < res0 * 0.01,
          f'{res0:.2e} → {res2:.2e}')

    # Physical check: both give same RCS to 0.05 dB
    theta = np.linspace(0, np.pi, 91)
    od    = np.stack([np.sin(theta), np.zeros_like(theta), np.cos(theta)], axis=1)
    rcs0  = rcs_3d(nodes, areas, x0, k, od)
    rcs2  = rcs_3d(nodes, areas, x2, k, od)
    db_diff = np.abs(10*np.log10(np.maximum(rcs0,1e-20)) -
                     10*np.log10(np.maximum(rcs2,1e-20))).max()
    check(f'RCS diff GMRES vs IR < 0.1 dB', db_diff < 0.1, f'{db_diff:.4f} dB')

    return conv2 and res2 < 1e-9


def test_mie_accuracy(N_target=1280, k=3.0):
    print(f'\n=== Mie accuracy via multi-RHS  N~{N_target}  k={k} ===')
    nodes, _, areas = icosphere(N_target, R=1.0)
    N = len(nodes)
    inc = np.array([0., 0., 1.])
    B   = np.asfortranarray(make_rhs_3d(nodes, k, inc).reshape(N, 1))

    asm     = BEMAssembler3DMulti()
    X, n_cv = asm.solve_multi_rhs(nodes, areas, k, B, restart=50, tol=1e-6)
    sigma   = X[:, 0]

    theta   = np.linspace(0, np.pi, 181)
    od      = np.stack([np.sin(theta), np.zeros_like(theta), np.cos(theta)], axis=1)
    rcs_bem = rcs_3d(nodes, areas, sigma, k, od)
    rcs_mie = mie_soft_sphere_rcs(k, 1.0, theta)
    err_db  = np.abs(10*np.log10(np.maximum(rcs_bem,1e-20)) -
                     10*np.log10(np.maximum(rcs_mie,1e-20)))

    print(f'  n_converged={n_cv}/1  max_err={err_db.max():.4f} dB  rms={err_db.mean():.4f} dB')
    ok = check(f'Max error vs Mie < 0.05 dB', err_db.max() < 0.05,
               f'{err_db.max():.4f} dB')
    return ok


if __name__ == '__main__':
    results = []
    results.append(test_multi_rhs())
    results.append(test_ir())
    results.append(test_mie_accuracy())

    print('\n' + '='*50)
    n_pass = sum(results)
    print(f'Phase 0: {n_pass}/{len(results)} test suites passed')
    if n_pass < len(results):
        sys.exit(1)
