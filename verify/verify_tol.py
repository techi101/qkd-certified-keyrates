"""Cross-platform variant of verify_witness.check (verify_witness.py itself is frozen by protocols/SOURCE_HASHES_10R.json).

With tol=0 it applies exactly the same test as verify_witness.check: every endpoint must be bit-identical.
With tol>0, the two lower ends that start from a float eigenvalue (route A and each dense lo) may differ from the saved
ones by at most tol, which absorbs BLAS last-bit differences between platforms. Every recomputed value is itself an
Arb-proven bound. Every upper end, the leakage and the net rates must still match bit for bit.
"""
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']: os.environ[k] = '1'
import numpy as np

M = lambda d: np.array(d['re']) + 1j * np.array(d['im'])
_C = {}


def check(rec, tol=0.0):
    from flint import arb, ctx
    from fastcheck import FastChecker2, lo, up
    from enclosed_leak import leakage_bits
    import t1_standard_float as T1
    Nc, p, wt = rec['Nc'], rec['params'], rec['witness']
    if Nc not in _C: _C[Nc] = FastChecker2(Nc, 400)
    st = {'w': np.array(wt['w']), 'V': M(wt['V']), 'lambda': np.array(wt['lambda'])}
    bad = []
    a = _C[Nc].check(st, p)['certified_lower_bound']
    dev = abs(a - rec['A'])
    if dev > tol: bad.append(('A', rec['A'], a))
    for e in rec.get('dense', []):
        r = T1.exact_bound(M(e['S']), st['lambda'], p, 400)
        if r['status'] != e['status'] or (r['status'] == 'ok' and (abs(r['lo'] - e['lo']) > tol or r['hi'] != e['hi'])):
            bad.append(('dense', e['eps'], e.get('lo'), r.get('lo'), e.get('hi'), r.get('hi')))
        elif r['status'] == 'ok':
            dev = max(dev, abs(r['lo'] - e['lo']))
    ctx.prec = 128
    leak, _ = leakage_bits(*p); ln2 = arb(2).log()
    if (lo(leak), up(leak)) != (rec['leak_lo'], rec['leak_hi']): bad.append(('leak',))
    if lo(arb(rec['A']) / ln2 - leak) != rec['rateA_lo']: bad.append(('rateA',))
    for e in rec.get('dense', []):
        if e['status'] == 'ok' and (lo(arb(e['lo']) / ln2 - leak) != e['rate_lo'] or up(arb(e['hi']) / ln2 - leak) != e['rate_hi']):
            bad.append(('rate', e['eps']))
    return {'cohort': rec['cohort'], 'i': rec['i'], 'ok': not bad, 'max_lower_end_dev': dev, 'bad': bad}
