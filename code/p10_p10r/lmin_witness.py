"""Certified lower bounds on lambda_min of every stored dense matrix S (Protocol 10R witnesses) and of R_z.
lambda_min(S) >= t is proved by an Arb LDL test that S - t I is positive definite (ball_pd), bisecting t
downward from the float estimate. Reports min over states of t / (eps/d) per eps. Usage: python lmin_witness.py out_p10r.json"""
import sys, json
import numpy as np
from flint import arb, ctx
from fastcheck import to_acb, herm, eye, ball_pd
from qkd import Model

ctx.prec = 400
d = 36
D = json.load(open(sys.argv[1]))['res']
M = lambda x: np.array(x['re']) + 1j * np.array(x['im'])


def cert_lmin(A):
    n = A.shape[0]; X = herm(to_acb(A)); w = np.linalg.eigvalsh((A + A.conj().T) / 2)[0]
    t = w * (1 - 1e-9) if w > 0 else w - abs(w) * 1e-9 - 1e-300
    for _ in range(60):
        if ball_pd(X - eye(n) * arb(float(t))): return float(t), float(w)
        t = t - abs(t) * 1e-6 if t > 0 else t * 2 - 1e-300
    return None, float(w)


m = Model(8); Rlo = []
for A in getattr(m, 'R', None) or []:
    Rlo.append(cert_lmin(np.asarray(A, dtype=complex))[0])
out = {'R_lmin_certified': Rlo, 'rows': []}
for r in D:
    if r['Nc'] != 8: continue
    for e in r['dense']:
        t, w = cert_lmin(M(e['S'])); out['rows'].append(dict(cohort=r['cohort'], i=r['i'], eps=e['eps'], lmin_lo=t, lmin_float=w))
json.dump(out, open('lmin_witness.json', 'w'))
print('R_z certified lambda_min lower bounds:', Rlo)
for eps in sorted({x['eps'] for x in out['rows']}):
    R = [x for x in out['rows'] if x['eps'] == eps]; ok = [x['lmin_lo'] for x in R if x['lmin_lo'] is not None]
    rat = np.array(ok) / (eps / d)
    print(f'eps {eps:.0e}: certified {len(ok)}/{len(R)}; min lambda_min(S)/(eps/d) = {rat.min():.6f}, max {rat.max():.6f}')
