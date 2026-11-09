"""Frozen Protocol 7; no resume from partial output."""
import hashlib, json, time, traceback, os
from pathlib import Path
import numpy as np
from optimized_solver import ReflectedModel, solve_optimized
from model_store import load_model
from verify_v4 import Checker
from fastcheck import FastChecker2

EXPECTED = 'e7bd0362952ba09d5f9dc6d18acda7f851f6f2930f2c0efd19b51de7390da354'
NAMES = ['R6_cold', 'R6_ridge']
OUT = Path('out_verify7.json')


def atomic(obj):
    tmp = OUT.with_suffix('.tmp'); tmp.write_text(json.dumps(obj, allow_nan=False), encoding='utf8'); os.replace(tmp, OUT)


def main():
    if OUT.exists(): raise RuntimeError('Existing result; refusing ambiguous resume')
    hashes = json.loads(Path('SOURCE_HASHES.json').read_text())
    for name, h in hashes.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != h: raise RuntimeError('Source hash mismatch: ' + name)
    digest = hashlib.sha256(Path('ridge_model.npz').read_bytes()).hexdigest()
    if digest != EXPECTED: raise RuntimeError('Frozen model hash mismatch')
    import flint
    out = {'complete': False, 'protocol_sha256': hashes['PROTOCOL_7_FROZEN.md'], 'model_sha256': digest,
           'flint_version': flint.__version__, 'source_hashes': hashes, 'res': [], 'failures': []}
    start = time.perf_counter()
    m = ReflectedModel(8); rid = load_model('ridge_model.npz')
    fc = FastChecker2(8, 400); old = Checker(8)
    pts = np.random.default_rng(np.random.SeedSequence([20261102, 9, 0])).random((80, 3)) * [55., .035, .4] + [5., .005, .5]
    known = set(json.loads(Path('KNOWN_POINT_HASHES.json').read_text())['hashes'])
    ph = lambda p: hashlib.sha256(np.asarray(p, dtype='<f8').tobytes()).hexdigest()
    if len({ph(p) for p in pts}) != 80 or any(ph(p) in known for p in pts): raise RuntimeError('Known point reuse')
    out['points'] = pts.tolist()
    from physical_moments import sample_inputs
    dev = (20., .02, .7); dx = sample_inputs(*dev, None, None); zero = np.zeros((m.dim, m.dim))
    for learned in (False, True):
        r = solve_optimized(m, m.gamma(*dev), zero, model=rid if learned else None, x=dx)
        fc.check({**r['state'], 'lambda': m.lift[1:] @ r['state']['lambda']}, dev)
    order_rng = np.random.default_rng(np.random.SeedSequence([20261102, 9, 2]))
    atomic(out)
    try:
        for i, p in enumerate(pts):
            x = sample_inputs(*p, None, None); gam = m.gamma(*p); reps = []; states = {}
            for rep in range(3):
                order = list(order_rng.permutation(NAMES)); row = {'order': order}
                for name in order:
                    r = solve_optimized(m, gam, zero, model=None if name == 'R6_cold' else rid, x=x)
                    tr = r['trajectory']
                    row[name] = {'elapsed_seconds': r['elapsed_seconds'], 'inference_seconds': r['inference_seconds'],
                                 'final_cert': tr[-1]['cert'], 'final_objective': tr[-1]['approx_objective'],
                                 'stopped_at_target': r['stopped_at_target'], 'failed_inner_steps': r['failed_inner_steps'],
                                 'max_trace_error': max(t['trace_error'] for t in tr), 'final_residual': tr[-1]['residual']}
                    if not r['stopped_at_target'] or r['failed_inner_steps'] or row[name]['max_trace_error'] > 1e-10 or tr[-1]['residual'] > 1e-7:
                        out['failures'].append({'point': i, 'rep': rep, 'method': name, 'kind': 'numerical_gate'})
                    if rep == 0:
                        states[name] = ({**r['state'], 'lambda': m.lift[1:] @ r['state']['lambda']}, tr[-1]['cert'])
                reps.append(row)
            checks = {}
            for name in NAMES:
                st, c = states[name]
                rec = {'float_cert': c}
                try:
                    times = []
                    for k in range(3):
                        v = fc.check(st, p); times.append(v['seconds'])
                        if k == 0: rec.update(certified_lower_bound=v['certified_lower_bound'], used=list(fc.used), shift=v['shift'])
                    rec.update(fast_times=times, fast_median=float(np.median(times)),
                               diff_nats=rec['certified_lower_bound'] - c)
                    if abs(rec['diff_nats']) > 1e-6: out['failures'].append({'point': i, 'method': name, 'kind': 'agreement', 'diff': rec['diff_nats']})
                except Exception as exc:
                    rec['error'] = repr(exc); out['failures'].append({'point': i, 'method': name, 'kind': 'certification', 'detail': repr(exc)})
                if i < 6:
                    try:
                        o = old.check(st, p, c)
                        rec.update(old_seconds=o['verification_seconds_including_setup'], old_value=o['certificate_recomputed'])
                    except Exception as exc:
                        rec['old_error'] = repr(exc)
                checks[name] = rec
            out['res'].append({'index': i, 'params': p.tolist(), 'reps': reps, 'checks': checks}); atomic(out)
            print('point', i + 1, '/80', flush=True)
        out.update(complete=True, total_job_seconds=time.perf_counter() - start); atomic(out)
        print('COMPLETE; failures:', len(out['failures']), flush=True)
    except Exception as exc:
        out.update(error=repr(exc), traceback=traceback.format_exc(), complete=False); atomic(out); raise


if __name__ == '__main__':
    main()
