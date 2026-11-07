"""Standalone verifier for Protocol 10R witnesses.
For every record: rebuild the route-A state from the saved witness (w, V, lambda) and re-run the certified check;
re-enclose every saved dense matrix S exactly as stored; recompute leakage and net rates in Arb.
Every recomputed endpoint must equal the saved one. Writes verify_report.json.
Usage: python verify_witness.py out_p10r.json"""
import sys, os, json
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']: os.environ[k] = '1'
import numpy as np
from multiprocessing import Pool

M = lambda d: np.array(d['re']) + 1j * np.array(d['im'])
_C = {}


def check(rec):
    from flint import arb, ctx
    from fastcheck import FastChecker2, lo, up
    from enclosed_leak import leakage_bits
    import t1_standard_float as T1
    Nc, p, wt = rec['Nc'], rec['params'], rec['witness']
    if Nc not in _C: _C[Nc] = FastChecker2(Nc, 400)
    st = {'w': np.array(wt['w']), 'V': M(wt['V']), 'lambda': np.array(wt['lambda'])}
    bad = []
    a = _C[Nc].check(st, p)['certified_lower_bound']
    if a != rec['A']: bad.append(('A', rec['A'], a))
    for e in rec.get('dense', []):
        r = T1.exact_bound(M(e['S']), st['lambda'], p, 400)
        if r['status'] != e['status'] or (r['status'] == 'ok' and (r['lo'] != e['lo'] or r['hi'] != e['hi'])):
            bad.append(('dense', e['eps'], e.get('lo'), r.get('lo'), e.get('hi'), r.get('hi')))
    ctx.prec = 128
    leak, _ = leakage_bits(*p); ln2 = arb(2).log()
    if (lo(leak), up(leak)) != (rec['leak_lo'], rec['leak_hi']): bad.append(('leak',))
    if lo(arb(rec['A']) / ln2 - leak) != rec['rateA_lo']: bad.append(('rateA',))
    for e in rec.get('dense', []):
        if e['status'] == 'ok' and (lo(arb(e['lo']) / ln2 - leak) != e['rate_lo'] or up(arb(e['hi']) / ln2 - leak) != e['rate_hi']):
            bad.append(('rate', e['eps']))
    return {'cohort': rec['cohort'], 'i': rec['i'], 'ok': not bad, 'bad': bad}


if __name__ == '__main__':
    d = json.load(open(sys.argv[1]))
    with Pool(4) as pool: res = pool.map(check, d['res'])
    nbad = sum(not r['ok'] for r in res)
    json.dump({'n': len(res), 'n_mismatch': nbad, 'res': res}, open('verify_report.json', 'w'))
    print('VERIFY', len(res), 'records, mismatches:', nbad, flush=True)
