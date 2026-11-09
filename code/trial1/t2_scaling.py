"""Trial 2: does the sub-second enclosure survive larger photon-number cutoffs?
Fresh points (seed [20261105,2,Nc]) in cohort-A ranges; cold-start solver; route A (FastChecker2);
float certificate; same statistics as Protocol 8. Development-style trial, not a frozen protocol."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from optimized_solver import ReflectedModel, solve_optimized
from fastcheck import FastChecker2
Nc = int(sys.argv[1]); npts = int(sys.argv[2])
t = time.perf_counter(); m = ReflectedModel(Nc); t_model = time.perf_counter() - t
t = time.perf_counter(); fc = FastChecker2(Nc, 400); t_fc = time.perf_counter() - t
P = np.random.default_rng(np.random.SeedSequence([20261105, 2, Nc])).random((npts, 3)) * [55., .035, .4] + [5., .005, .5]
out = {'Nc': Nc, 'dim': m.dim, 'setup_model_s': t_model, 'setup_fc_s': t_fc, 'res': []}
for p in P:
    gam = m.gamma(*p)
    r = solve_optimized(m, gam, np.zeros((m.dim, m.dim)))
    s = r['state']
    st = {'w': s['w'], 'V': s['V'], 'lambda': m.lift[1:] @ s['lambda']}
    lmin = float((s['w'] - (s['w'].max() + np.log(np.exp(s['w'] - s['w'].max()).sum()))).min())
    rec = {'params': p.tolist(), 'solve_s': r['elapsed_seconds'], 'stopped': r['stopped_at_target'],
           'float_cert': r['trajectory'][-1]['cert'], 'min_log_eig': lmin}
    try:
        v = fc.check(st, p)
        rec.update(A=v['certified_lower_bound'], A_s=v['seconds'], prec=list(fc.used), A_minus_float=v['certified_lower_bound'] - rec['float_cert'])
    except Exception as exc:
        rec['A_error'] = repr(exc)[:120]
    out['res'].append(rec); print(json.dumps(rec), flush=True)
json.dump(out, open(sys.argv[3], 'w'), indent=1)
