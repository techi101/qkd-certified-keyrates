"""Protocol 10R runner: identical computation to run_p10.py (Protocol 10), plus witness export, outward leakage
endpoints and Arb net rates. Writes out_p10r.json, then runs verify_witness.py on it."""
import sys, os, json, time, hashlib
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']: os.environ[k] = '1'
from pathlib import Path
import numpy as np
from multiprocessing import Pool

SMOKE = '--smoke' in sys.argv
COH = {'A': (0, 80, [5., .005, .5], [55., .035, .4], 8),
       'B': (1, 40, [60., .01, .5], [40., .04, .4], 8),
       'N12': (2, 40, [5., .005, .5], [55., .035, .4], 12)}
EPS = (1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-8)
ph = lambda p: hashlib.sha256(np.asarray(p, dtype='<f8').tobytes()).hexdigest()
cm = lambda M: {'re': np.real(M).tolist(), 'im': np.imag(M).tolist()}


def points():
    out = []
    for c, (sd, n, lo_, span, nc) in COH.items():
        P = np.random.default_rng(np.random.SeedSequence([20261105, 10, sd])).random((n, 3)) * span + lo_
        out += [(c, i, P[i], nc) for i in range(n)]
    return out


_M = {}
def work(task):
    c, i, p, Nc = task
    from flint import ctx, arb
    from optimized_solver import ReflectedModel, solve_optimized
    from fastcheck import FastChecker2, lo, up
    from enclosed_leak import leakage_bits
    if Nc not in _M:
        _M[Nc] = (ReflectedModel(Nc), FastChecker2(Nc, 400))
    m, fc = _M[Nc]
    t0 = time.perf_counter()
    r = solve_optimized(m, m.gamma(*p), np.zeros((m.dim, m.dim))); s = r['state']
    st = {'w': s['w'], 'V': s['V'], 'lambda': m.lift[1:] @ s['lambda']}
    rec = {'cohort': c, 'i': i, 'Nc': Nc, 'params': p.tolist(), 'float_cert': r['trajectory'][-1]['cert'],
           'solve_s': r['elapsed_seconds'], 'stopped': r['stopped_at_target'], 'failed_inner': r['failed_inner_steps'],
           'witness': {'w': np.asarray(s['w']).tolist(), 'V': cm(s['V']), 'lambda': np.asarray(st['lambda']).tolist()}}
    v = fc.check(st, p); rec.update(A=v['certified_lower_bound'], A_s=v['seconds'])
    dense = []
    if Nc == 8:
        import t1_standard_float as T1
        from qkd import Model
        if 'dm' not in _M: _M['dm'] = Model(8)
        dm = _M['dm']
        w, V, lam = st['w'], st['V'], st['lambda']
        lw = w - (w.max() + np.log(np.exp(w - w.max()).sum()))
        sig = (V * np.exp(lw)) @ V.conj().T; sig = (sig + sig.conj().T) / 2; gam = dm.gamma(*p)
        for eps in EPS:
            S = dm.regular(sig, eps); fv = float(dm.cert(dm.grad(S), lam, gam))
            e = T1.exact_bound(S, lam, p, 400); e.update(eps=eps, float_std=fv, S=cm(S)); dense.append(e)
    # leakage and net rates in Arb (128 bits), outward export
    ctx.prec = 128
    leak, _ = leakage_bits(*p); ln2 = arb(2).log()
    rec['leak_lo'] = lo(leak); rec['leak_hi'] = up(leak); rec['leak_mid'] = float(leak.mid())
    rec['rateA_lo'] = lo(arb(rec['A']) / ln2 - leak)
    for e in dense:
        e['float_claim'] = e['float_std'] / np.log(2) - rec['leak_mid']
        if e['status'] == 'ok':
            e['rate_lo'] = lo(arb(e['lo']) / ln2 - leak); e['rate_hi'] = up(arb(e['hi']) / ln2 - leak)
    if dense: rec['dense'] = dense
    rec['total_s'] = time.perf_counter() - t0
    return rec


def atomic(obj, name='out_p10r.json'):
    Path(name + '.tmp').write_text(json.dumps(obj)); os.replace(name + '.tmp', name)


if __name__ == '__main__':
    hashes = json.loads(Path('SOURCE_HASHES_10R.json').read_text())
    for name, h in hashes.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != h: raise RuntimeError('Source hash mismatch: ' + name)
    known = set(json.loads(Path('KNOWN_POINT_HASHES.json').read_text())['hashes'])
    tasks = points()
    hs = [ph(t[2]) for t in tasks]
    if len(set(hs)) != len(hs) or any(h in known for h in hs): raise RuntimeError('Known point reuse')
    if SMOKE: tasks = [t for t in tasks if t[1] == 0]
    import flint
    out = {'complete': False, 'smoke': SMOKE, 'protocol_sha256': hashes['PROTOCOL_10R.md'],
           'flint_version': flint.__version__, 'numpy': np.__version__, 'source_hashes': hashes,
           'n_tasks': len(tasks), 'res': []}
    atomic(out); t0 = time.perf_counter()
    tasks.sort(key=lambda t: t[3] != 12)
    with Pool(4) as pool:
        for rec in pool.imap_unordered(work, tasks):
            out['res'].append(rec); atomic(out)
            print(json.dumps({k: rec[k] for k in ('cohort', 'i', 'A', 'A_s', 'total_s')}), flush=True)
    out['complete'] = len(out['res']) == len(tasks); out['wall_s'] = time.perf_counter() - t0; atomic(out)
    print('DONE', out['complete'], out['wall_s'], flush=True)
    import subprocess
    subprocess.run([sys.executable, 'verify_witness.py', 'out_p10r.json'], check=False)
