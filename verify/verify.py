"""Release verifier for the Protocol 10R certificate witnesses.

For each record it recomputes, from the witness alone (no optimisation is rerun):
  * the route-A certified lower bound from the saved state (w, V, lambda);
  * the two-sided Arb enclosure [lo, hi] of the exact certificate of every saved dense matrix S;
  * the enclosed error-correction leakage and the net-rate bounds.
A record PASSES when every recomputed endpoint equals the saved one. It FAILS (verification refused) otherwise.

Usage:
  python verify.py ../data/out_p10r.json.gz            # all 160 records
  python verify.py ../data/out_p10r.json.gz --n 12     # first 12 records
  python verify.py tampered.json --procs 2
"""
import sys, os, json, gzip, argparse, time
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']: os.environ[k] = '1'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
from verify_witness import check


def load(path):
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rt') as f: return json.load(f)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('witnesses'); ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--procs', type=int, default=4); ap.add_argument('--out', default='verify_result.json')
    a = ap.parse_args()
    recs = load(a.witnesses)['res']
    if a.n: recs = recs[:a.n]
    t0 = time.time()
    with Pool(a.procs) as pool: res = pool.map(check, recs)
    bad = [r for r in res if not r['ok']]
    import flint, numpy, scipy
    json.dump({'n': len(res), 'n_fail': len(bad), 'versions': {'python-flint': flint.__version__, 'numpy': numpy.__version__,
               'scipy': scipy.__version__}, 'wall_s': time.time() - t0, 'res': res}, open(a.out, 'w'), indent=1)
    for r in bad: print('FAIL', r['cohort'], r['i'], r['bad'][:3])
    print(f'VERIFIED {len(res) - len(bad)}/{len(res)} records; refused {len(bad)}  (python-flint {flint.__version__}, numpy {numpy.__version__})')
