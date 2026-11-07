"""Generate key-rate data for the NN audit.

For each (L, xi, alpha): cold-start solve (Protocol 6/7/8 solver, final 1e-6 gap), giving
  float_cert : float certificate at the final state (the 'numerical method' label, as in Liu et al.)
  U          : approximate primal objective at the final state (approximate upper reference for f*)
  leak       : delta_EC in bits (beta = 0.95)
and, if certify=True, the ball-arithmetic certified lower bound A (route A, FastChecker2).
NN inputs follow Liu et al. 2022 (29 variables): 16 Bob moments conditioned on Alice's
symbol, 12 off-diagonal Alice-marginal terms, and the excess noise xi.
Usage: python gen_data.py <seed_a> <seed_b> <n> <certify 0/1> <out.npz> [workers]
"""
import sys, time
import numpy as np
from multiprocessing import Pool

RANGES = ([5., 150.], [0.0045, 0.0405], [0.62, 0.72])


def points(seed_a, seed_b, n):
    u = np.random.default_rng(np.random.SeedSequence([20261103, seed_a, seed_b])).random((n, 3))
    lo = np.array([r[0] for r in RANGES]); hi = np.array([r[1] for r in RANGES])
    return lo + u * (hi - lo)


_m = _fc = None


def init(certify):
    global _m, _fc
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    from optimized_solver import ReflectedModel
    _m = ReflectedModel(8)
    if certify:
        from fastcheck import FastChecker2
        _fc = FastChecker2(8, 400)


def features(m, p):
    g = np.asarray(m.full_model.gamma(*p), float)  # 31 = 16 Bob moments + 15 Alice terms
    alice = g[16:]
    # 15 Alice terms = 3 diagonal (one dropped by the trace) + 6 real + 6 imaginary off-diagonal
    return np.r_[g[:16], alice[3:], p[1]]


def work(p):
    from optimized_solver import solve_optimized
    from keyrate import leakage_bits
    t = time.perf_counter()
    r = solve_optimized(_m, _m.gamma(*p), np.zeros((_m.dim, _m.dim)))
    tr = r['trajectory'][-1]
    out = {'x': features(_m, p), 'float_cert': tr['cert'], 'U': tr['approx_objective'], 'leak': leakage_bits(*p)[0],
           'ok': bool(r['stopped_at_target'] and not r['failed_inner_steps']), 'solve_s': time.perf_counter() - t, 'A': np.nan, 'A_s': np.nan}
    if _fc is not None:
        st = {**r['state'], 'lambda': _m.lift[1:] @ r['state']['lambda']}
        try:
            v = _fc.check(st, p); out['A'] = v['certified_lower_bound']; out['A_s'] = v['seconds']
        except Exception:
            pass
    return out


if __name__ == '__main__':
    sa, sb, n, cert, outp = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), bool(int(sys.argv[4])), sys.argv[5]
    workers = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    P = points(sa, sb, n)
    t0 = time.perf_counter()
    with Pool(workers, initializer=init, initargs=(cert,)) as pool:
        res = pool.map(work, list(P), chunksize=8)
    keys = ['float_cert', 'U', 'leak', 'ok', 'solve_s', 'A', 'A_s']
    np.savez_compressed(outp, params=P, X=np.array([r['x'] for r in res]), **{k: np.array([r[k] for r in res]) for k in keys})
    print('done', n, 'points in %.0f s' % (time.perf_counter() - t0), 'failed solves', int(sum(not r['ok'] for r in res)), flush=True)
