"""Frozen Protocol 8; no resume from partial output."""
import hashlib, json, time, traceback, os
from pathlib import Path
import numpy as np
from optimized_solver import ReflectedModel, solve_optimized
from verify_v4 import Checker
from fastcheck import FastChecker, FastChecker2, LoewnerCrossCheck
from keyrate import leakage_bits

OUT = Path('out_verify8.json')


def atomic(obj):
    tmp = OUT.with_suffix('.tmp'); tmp.write_text(json.dumps(obj, allow_nan=False), encoding='utf8'); os.replace(tmp, OUT)


def float_cert_time(m, state, gam):
    ts = []
    for _ in range(3):
        t = time.perf_counter(); G, H, F = m.gradient(state['w'], state['V']); m.cert(G, state['lam_red'], gam); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def main():
    if OUT.exists(): raise RuntimeError('Existing result; refusing ambiguous resume')
    hashes = json.loads(Path('SOURCE_HASHES.json').read_text())
    for name, h in hashes.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != h: raise RuntimeError('Source hash mismatch: ' + name)
    import flint
    out = {'complete': False, 'protocol_sha256': hashes['PROTOCOL_8_FROZEN.md'], 'flint_version': flint.__version__,
           'source_hashes': hashes, 'res': [], 'failures': []}
    start = time.perf_counter()
    t = time.perf_counter(); m = ReflectedModel(8); out['setup_model_s'] = time.perf_counter() - t
    t = time.perf_counter(); fc = FastChecker2(8, 400); out['setup_fast_s'] = time.perf_counter() - t
    t = time.perf_counter(); lc = LoewnerCrossCheck(8, 400); out['setup_cross_s'] = time.perf_counter() - t
    t = time.perf_counter(); v1 = FastChecker(8, 400); out['setup_v1_s'] = time.perf_counter() - t
    old = Checker(8)
    A = np.random.default_rng(np.random.SeedSequence([20261102, 10, 0])).random((80, 3)) * [55., .035, .4] + [5., .005, .5]
    B = np.random.default_rng(np.random.SeedSequence([20261102, 10, 1])).random((40, 3)) * [40., .04, .4] + [60., .01, .5]
    known = set(json.loads(Path('KNOWN_POINT_HASHES.json').read_text())['hashes'])
    ph = lambda p: hashlib.sha256(np.asarray(p, dtype='<f8').tobytes()).hexdigest()
    allp = list(A) + list(B)
    if len({ph(p) for p in allp}) != 120 or any(ph(p) in known for p in allp): raise RuntimeError('Known point reuse')
    zero = np.zeros((m.dim, m.dim))
    dev = (20., .02, .7); r = solve_optimized(m, m.gamma(*dev), zero)
    st = {**r['state'], 'lambda': m.lift[1:] @ r['state']['lambda']}; fc.check(st, dev); lc.cross(st, dev)
    atomic(out)
    try:
        for idx, (cohort, p) in enumerate([('A', p) for p in A] + [('B', p) for p in B]):
            gam = m.gamma(*p)
            r = solve_optimized(m, gam, zero)
            tr = r['trajectory']; s = r['state']
            st = {'w': s['w'], 'V': s['V'], 'lambda': m.lift[1:] @ s['lambda'], 'lam_red': s['lambda']}
            rec = {'cohort': cohort, 'params': p.tolist(), 'solve_s': r['elapsed_seconds'], 'float_cert': tr[-1]['cert'],
                   'stopped': r['stopped_at_target'], 'failed_inner': r['failed_inner_steps'],
                   'state': {'w': s['w'].tolist(), 'V_re': s['V'].real.tolist(), 'V_im': s['V'].imag.tolist(),
                             'lambda_source': st['lambda'].tolist()}}
            if not r['stopped_at_target'] or r['failed_inner_steps']:
                out['failures'].append({'i': idx, 'kind': 'numerical_gate'})
            rec['float_cert_s'] = float_cert_time(m, st, gam)
            try:
                ts = []
                for k in range(3):
                    v = fc.check(st, p); ts.append(v['seconds'])
                    if k == 0: rec.update(A=v['certified_lower_bound'], A_upper=v['certified_upper_end'], A_used=list(fc.used))
                rec['A_s'] = float(np.median(ts)); rec['A_minus_float'] = rec['A'] - rec['float_cert']
                if abs(rec['A_minus_float']) > 1e-6: out['failures'].append({'i': idx, 'kind': 'agreement'})
            except Exception as exc:
                rec['A_error'] = repr(exc); out['failures'].append({'i': idx, 'kind': 'routeA', 'detail': repr(exc)})
            try:
                v = lc.cross(st, p); rec.update(B=v['cross_lower_bound'], B_s=v['seconds'])
                if 'A' in rec and not (0 <= rec['A'] - rec['B'] <= 1e-9 + 1e-12):
                    out['failures'].append({'i': idx, 'kind': 'routeB_mismatch', 'diff': rec['A'] - rec['B']})
            except Exception as exc:
                rec['B_error'] = repr(exc); out['failures'].append({'i': idx, 'kind': 'routeB', 'detail': repr(exc)})
            leak, _ = leakage_bits(*p); rec['leak_bits'] = leak
            if 'A' in rec: rec['rate_bits_certified'] = rec['A'] / np.log(2) - leak
            rec['rate_bits_float'] = rec['float_cert'] / np.log(2) - leak
            if cohort == 'A' and idx < 10:
                try:
                    v = v1.check(st, p); rec['v1_s'] = v['seconds']; rec['v1_value'] = v['certified_lower_bound']
                except Exception as exc:
                    rec['v1_error'] = repr(exc)
            if cohort == 'A' and idx < 6:
                try:
                    o = old.check(st, p, rec['float_cert']); rec['old_s'] = o['verification_seconds_including_setup']
                except Exception as exc:
                    rec['old_error'] = repr(exc)
            out['res'].append(rec); atomic(out); print('state', idx + 1, '/120', flush=True)
        out.update(complete=True, total_job_seconds=time.perf_counter() - start); atomic(out)
        print('COMPLETE; failures:', len(out['failures']), flush=True)
    except Exception as exc:
        out.update(error=repr(exc), traceback=traceback.format_exc(), complete=False); atomic(out); raise


if __name__ == '__main__':
    main()
