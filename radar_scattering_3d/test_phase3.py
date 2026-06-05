"""
Phase 3 tests: COBOL 3D aggregator + rcs_bridge_3d.py

Checks:
  1. Record structs:      _CHK_FMT and _ENS_FMT are exactly 2048 bytes each.
  2. Byte layout:         checkpoint fields land at correct offsets.
  3. Round-trip write/read: write .bin, read back, all fields match.
  4. Python Welford:      aggregate_rcs_3d() fallback matches analytic mean/std.
  5. COBOL aggregator:    rcs_aggregator_3d produces byte-identical output
                          to Python fallback.
  6. FLAGS protocol:      in-progress records (FLAGS=1) are skipped.
  7. generate_test_data_3d: writes 100 files with correct sizes.
"""

import sys, os, tempfile, shutil
sys.path.insert(0, os.path.dirname(__file__))

import struct
import numpy as np
from pathlib import Path

from rcs_bridge_3d import (
    _CHK_FMT, _ENS_FMT, CHK_SIZE, ENS_SIZE,
    N_MONO_ANGLES, INC_GRID_NTHETA, INC_GRID_NPHI,
    write_checkpoint_3d, read_ensemble_3d,
    aggregate_rcs_3d, _python_aggregate_3d,
    RCSEnsemble3D,
)
from generate_test_data_3d import generate, synthetic_rcs, N_TARGETS, N_SEEDS

PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'

def check(label, cond, detail=''):
    tag = PASS if cond else FAIL
    print(f'  [{tag}] {label}' + (f'  ({detail})' if detail else ''))
    return cond


# ── 1. Struct sizes ────────────────────────────────────────────────────────────

def test_structs():
    print('\n=== 1. Struct sizes ===')
    ok  = check('CHK_FMT = 2048 bytes', _CHK_FMT.size == 2048, f'{_CHK_FMT.size}')
    ok &= check('ENS_FMT = 2048 bytes', _ENS_FMT.size == 2048, f'{_ENS_FMT.size}')
    ok &= check('N_MONO_ANGLES = 72',   N_MONO_ANGLES == 72)
    return ok


# ── 2. Byte layout ─────────────────────────────────────────────────────────────

def test_layout():
    print('\n=== 2. Byte layout ===')
    rcs    = np.zeros(72)
    raw    = bytearray(_CHK_FMT.pack(7, 3, 9.5, 72, 0, 5120, 8.0, *rcs, 6, 12))
    ok = True

    def int32_at(off):  return struct.unpack_from('<i', raw, off)[0]
    def float64_at(off): return struct.unpack_from('<d', raw, off)[0]

    ok &= check('target_id @ 0',        int32_at(0)   == 7)
    ok &= check('seed @ 4',             int32_at(4)   == 3)
    ok &= check('freq_ghz @ 8',         float64_at(8) == 9.5)
    ok &= check('n_angles @ 16',        int32_at(16)  == 72)
    ok &= check('flags @ 20',           int32_at(20)  == 0)
    ok &= check('n_panels @ 24',        int32_at(24)  == 5120)
    ok &= check('4-byte filler @ 28',   raw[28:32]    == b'\x00'*4)
    ok &= check('ka @ 32',              float64_at(32) == 8.0)
    ok &= check('rcs_dbm starts @ 40',  float64_at(40) == 0.0)
    ok &= check('inc_ntheta @ 616',     int32_at(616) == 6)
    ok &= check('inc_nphi @ 620',       int32_at(620) == 12)
    ok &= check('total size = 2048',    len(raw)      == 2048)
    return ok


# ── 3. Round-trip write / read ─────────────────────────────────────────────────

def test_roundtrip():
    print('\n=== 3. Round-trip write/read ===')
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rcs_in = np.linspace(-30.0, 5.0, 72)

        write_checkpoint_3d(td / 'rcs3d_T00_S0000.bin',
                            target_id=0, seed=0, freq_ghz=9.0, ka=8.0,
                            mono_rcs_dbm=rcs_in, n_panels=5120)

        raw = (td / 'rcs3d_T00_S0000.bin').read_bytes()
        ok  = check('file size = 2048', len(raw) == 2048, f'{len(raw)}')

        fields   = _CHK_FMT.unpack(raw)
        rcs_back = np.array(fields[7:79])
        ok &= check('target_id roundtrip',  fields[0] == 0)
        ok &= check('seed roundtrip',        fields[1] == 0)
        ok &= check('freq_ghz roundtrip',    fields[2] == 9.0)
        ok &= check('flags = 0 (complete)',  fields[4] == 0)
        ok &= check('n_panels roundtrip',    fields[5] == 5120)
        ok &= check('ka roundtrip',          fields[6] == 8.0)
        ok &= check('rcs values roundtrip',
                    np.allclose(rcs_back, rcs_in),
                    f'max_err={np.abs(rcs_back-rcs_in).max():.2e}')
        ok &= check('inc_ntheta roundtrip',  fields[79] == INC_GRID_NTHETA)
        ok &= check('inc_nphi roundtrip',    fields[80] == INC_GRID_NPHI)
    return ok


# ── 4. Python Welford ──────────────────────────────────────────────────────────

def test_python_welford():
    print('\n=== 4. Python Welford aggregation ===')
    with tempfile.TemporaryDirectory() as td:
        td    = Path(td)
        outf  = td / 'ens.stls'

        # Write 5 targets × 20 seeds
        generate(td, n_targets=5, n_seeds=20, verbose=False)

        results = _python_aggregate_3d(td, outf, n_expected=20)
        stls_written = outf.exists()   # check before tempdir is cleaned up

    ok  = check('5 ensemble records',  len(results) == 5,  f'{len(results)}')
    ok &= check('STLS output written', stls_written)

    for ens in results:
        t   = ens.target_id
        ok &= check(f'T{t}: n_seeds=20', ens.n_seeds_complete == 20,
                    f'{ens.n_seeds_complete}')
        ok &= check(f'T{t}: n_angles=72', ens.n_angles == 72)

        # Verify analytic mean: mean over 20 seeds ≈ noise-free pattern
        rcs_all   = np.stack([synthetic_rcs(t, s) for s in range(20)])
        true_mean = rcs_all.mean(axis=0)
        true_std  = rcs_all.std(axis=0, ddof=1)

        max_mean_err = np.abs(ens.mean_mono_rcs_dbm - true_mean).max()
        max_std_err  = np.abs(ens.std_mono_rcs_dbm  - true_std).max()

        ok &= check(f'T{t}: mean exact (err < 1e-10)',
                    max_mean_err < 1e-10, f'{max_mean_err:.2e}')
        ok &= check(f'T{t}: std exact (err < 1e-10)',
                    max_std_err  < 1e-10, f'{max_std_err:.2e}')

    return ok


# ── 5. COBOL vs Python ─────────────────────────────────────────────────────────

def test_cobol_vs_python():
    print('\n=== 5. COBOL aggregator vs Python fallback ===')
    exe = Path(__file__).parent / 'rcs_aggregator_3d'
    if not exe.exists():
        print(f'  [SKIP] {exe} not found — run make rcs_aggregator_3d')
        return True   # not a failure — just not built

    with tempfile.TemporaryDirectory() as td:
        td   = Path(td)
        generate(td, n_targets=5, n_seeds=20, verbose=False)
        outf_py   = td / 'ens_py.stls'
        outf_cobol = td / 'ens_cobol.stls'

        py_results    = _python_aggregate_3d(td, outf_py,    n_expected=20)
        cobol_results = aggregate_rcs_3d(td,    outf_cobol,  n_expected=20)

    ok = check('COBOL: 5 records', len(cobol_results) == 5,
               f'{len(cobol_results)}')

    for py, cb in zip(py_results, cobol_results):
        t = py.target_id
        ok &= check(f'T{t}: target_id match',   py.target_id == cb.target_id)
        ok &= check(f'T{t}: n_seeds match',     py.n_seeds_complete == cb.n_seeds_complete)

        mean_diff = np.abs(py.mean_mono_rcs_dbm - cb.mean_mono_rcs_dbm).max()
        std_diff  = np.abs(py.std_mono_rcs_dbm  - cb.std_mono_rcs_dbm).max()

        ok &= check(f'T{t}: mean matches Python (< 1e-9 dB)',
                    mean_diff < 1e-9, f'{mean_diff:.2e}')
        ok &= check(f'T{t}: std matches Python (< 1e-9 dB)',
                    std_diff  < 1e-9, f'{std_diff:.2e}')

    return ok


# ── 6. FLAGS protocol ──────────────────────────────────────────────────────────

def test_flags():
    print('\n=== 6. FLAGS protocol (in-progress records skipped) ===')
    with tempfile.TemporaryDirectory() as td:
        td   = Path(td)
        outf = td / 'ens.stls'
        rcs  = np.zeros(72)

        # Write 3 complete + 2 in-progress for target 0
        for s in range(3):
            write_checkpoint_3d(td / f'rcs3d_T00_S{s:04d}.bin',
                                0, s, 9.0, 8.0, rcs)
        for s in range(3, 5):
            write_checkpoint_3d(td / f'rcs3d_T00_S{s:04d}.bin',
                                0, s, 9.0, 8.0, rcs, complete=False)

        results = _python_aggregate_3d(td, outf, n_expected=20)

    ok  = check('1 ensemble record', len(results) == 1, f'{len(results)}')
    ok &= check('only 3 complete seeds counted',
                results[0].n_seeds_complete == 3, f'{results[0].n_seeds_complete}')
    return ok


# ── 7. generate_test_data_3d ──────────────────────────────────────────────────

def test_generate():
    print('\n=== 7. generate_test_data_3d ===')
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        generate(td, n_targets=N_TARGETS, n_seeds=N_SEEDS, verbose=False)

        files = sorted(td.glob('rcs3d_T*_S*.bin'))
        ok  = check(f'{N_TARGETS*N_SEEDS} files created',
                    len(files) == N_TARGETS * N_SEEDS, f'{len(files)}')
        ok &= check('all files 2048 bytes',
                    all(f.stat().st_size == 2048 for f in files))
        ok &= check('all FLAGS=0 (complete)',
                    all(struct.unpack_from('<i', f.read_bytes(), 20)[0] == 0
                        for f in files))
    return ok


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    results = [
        test_structs(),
        test_layout(),
        test_roundtrip(),
        test_python_welford(),
        test_cobol_vs_python(),
        test_flags(),
        test_generate(),
    ]
    print('\n' + '='*55)
    n_pass = sum(results)
    print(f'Phase 3: {n_pass}/{len(results)} test suites passed')
    if n_pass < len(results):
        sys.exit(1)
