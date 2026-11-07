"""Run T-W2 over a JSON of saved states (format of research_05nov_qkd_gap/data_out_verify8.json:
d['res'][k] = {cohort, params, state: {w, V_re, V_im, lambda_source}, ...}).
Usage: run_tw2.py STATES.json OUT_DIR [--n N] [--procs 4] [--sanity]
Writes OUT_DIR/state_<k>.json per state and OUT_DIR/tw2_all.json at the end.
--sanity: floor disabled, eps = 1e-6 (float must lie in, or within 1e-9 of, the exact enclosure)."""
import os, sys, json, time, argparse, hashlib
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

def job(args):
    k, rec, sanity = args
    import numpy as np
    import tw2
    from flint import ctx
    from enclosed_leak import leakage_bits            # from P10 src (path set by tw2)
    s = rec['state']
    V = np.array(s['V_re']) + 1j * np.array(s['V_im'])
    t = time.perf_counter()
    if sanity:
        r = tw2.tw2_state(s['w'], V, s['lambda_source'], rec['params'], floor=False, eps_override=1e-6)
    else:
        r = tw2.tw2_state(s['w'], V, s['lambda_source'], rec['params'])
    ctx.prec = 128
    leak, _ = leakage_bits(*rec['params'])
    r.update(k=k, cohort=rec.get('cohort'), params=rec['params'], leak_lo=float(leak.lower()),
             leak_hi=float(leak.upper()), leak_mid=float(leak.mid()), total_s=time.perf_counter() - t,
             state_sha=hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()[:16])
    return r

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('states'); ap.add_argument('out')
    ap.add_argument('--n', type=int, default=0); ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--sanity', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = json.load(open(a.states))['res']
    if a.n:
        res = res[:a.n]
    tasks = [(k, r, a.sanity) for k, r in enumerate(res)]
    t0 = time.time(); out = []
    with Pool(a.procs) as pool:
        for r in pool.imap_unordered(job, tasks):
            json.dump(r, open(os.path.join(a.out, f"state_{r['k']}.json"), 'w'), indent=1)
            out.append(r)
            print(r['k'], r['status'], 'float %.12g' % r['float_value'], 'exact [%s, %s]' % (r['exact_lo'], r['exact_hi']),
                  'eps %.3g' % r['eps_used'], 'exact_s %.2f' % r['exact_s'], flush=True)
    out.sort(key=lambda r: r['k'])
    json.dump(dict(n=len(out), sanity=a.sanity, wall_s=time.time() - t0, res=out),
              open(os.path.join(a.out, 'tw2_all.json'), 'w'), indent=1)
    print('wall_s', round(time.time() - t0, 1))
