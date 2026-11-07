"""Refusal test: the verifier must REJECT tampered witnesses.
Builds tampered.json from the first 3 cohort-A records of the witness file:
  T1: a saved dense matrix S made indefinite (largest-magnitude negative shift) -> status changes, must fail;
  T2: a claimed exact upper endpoint raised by 1e-6 nats (an overclaim) -> must fail;
  T3: the route-A multipliers perturbed by 1e-3 -> recomputed route-A bound differs, must fail;
plus the 3 untouched records, which must pass.
Usage: python refusal_test.py ../data/out_p10r.json.gz && python verify.py tampered.json --procs 3 --out refusal_result.json"""
import sys, json, gzip, copy
import numpy as np

src = sys.argv[1]
with (gzip.open if src.endswith('.gz') else open)(src, 'rt') as f: R = json.load(f)['res']
base = [r for r in R if r['cohort'] == 'A' and r['Nc'] == 8][:3]
out = [copy.deepcopy(r) for r in base]
t1, t2, t3 = (copy.deepcopy(r) for r in base)
S = np.array(t1['dense'][2]['S']['re']) + 1j * np.array(t1['dense'][2]['S']['im'])
S = S - 1e-3 * np.eye(S.shape[0])                      # now indefinite
t1['dense'][2]['S'] = {'re': S.real.tolist(), 'im': S.imag.tolist()}; t1['tamper'] = 'S made indefinite'
t2['dense'][0]['hi'] = t2['dense'][0]['hi'] + 1e-6; t2['tamper'] = 'claimed hi raised by 1e-6'
t3['witness']['lambda'] = (np.array(t3['witness']['lambda']) * (1 + 1e-3)).tolist(); t3['tamper'] = 'lambda perturbed'
for t in (t1, t2, t3): t['cohort'] = 'TAMPERED:' + t['tamper']
json.dump({'res': out + [t1, t2, t3]}, open('tampered.json', 'w'))
print('wrote tampered.json: 3 untouched (must pass) + 3 tampered (must be refused)')
