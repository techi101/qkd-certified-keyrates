"""Clipping-floor causal test on the EXACT Kaggle matrices saved by Protocol 10R (all 120 Nc=8 states, all eps).
Float side recomputed here (numpy eigh); exact side = saved Arb enclosure [lo, hi] of the same S.
Reports: clipped key-map eigenvalues, and overstatement vs floor setting."""
import sys, os, json
import numpy as np
SRC = r'D:/research paper/research_07nov_qkd_p10r/src'; sys.path.insert(0, SRC); os.chdir(SRC)
from qkd import Model
dm = Model(8)
D = json.load(open(r'D:/research paper/research_07nov_qkd_p10r/kaggle_out/out_p10r.json'))['res']
M = lambda d: np.array(d['re']) + 1j * np.array(d['im'])


def logm(X, floor):
    w, V = np.linalg.eigh((X + X.conj().T) / 2); n = int((w < floor).sum())
    return (V * np.log(np.clip(w, floor, None))) @ V.conj().T, n


def cert(S, lam, gam, fS, fB):
    g, nS = logm(S, fS); nB = 0
    for q, W in zip(dm.q, dm.W):
        Y = W.conj().T @ q @ S @ q @ W; L, n = logm(Y, fB); nB += n
        g = g - q @ W @ L @ W.conj().T @ q
    return float(dm.cert((g + g.conj().T) / 2, lam, gam)), nS, nB


FL = [(1e-300, 1e-300), (1e-300, 1e-30), (1e-300, 1e-16)]
rows = []
for r in D:
    if r['Nc'] != 8: continue
    lam = np.array(r['witness']['lambda']); gam = dm.gamma(*r['params'])
    for e in r['dense']:
        S = M(e['S']); out = [cert(S, lam, gam, a, b) for a, b in FL]
        rows.append(dict(eps=e['eps'], kaggle_float=e['float_std'], lo=e['lo'], hi=e['hi'],
                         here=[o[0] for o in out], clipS=out[0][1], clipB=out[0][2]))
json.dump(rows, open(r'D:/research paper/research_07nov_qkd_deep/floor_on_witness.json', 'w'))
for eps in sorted({x['eps'] for x in rows}):
    R = [x for x in rows if x['eps'] == eps]
    dk = np.array([x['here'][0] - x['kaggle_float'] for x in R])
    line = f"eps {eps:.0e}: laptop-vs-Kaggle float |diff| max {np.abs(dk).max():.1e}; clipped B eigs median {np.median([x['clipB'] for x in R]):.0f} (states with any {sum(x['clipB'] > 0 for x in R)}); clipped S {sum(x['clipS'] > 0 for x in R)} states"
    for k, (a, b) in enumerate(FL):
        ov = np.array([x['here'][k] - x['hi'] for x in R])
        line += f" | Bfloor {b:.0e}: over>0 {int((ov > 0).sum())}, >1e-6 {int((ov > 1e-6).sum())}, max {ov.max():+.2e}, under(<lo) {sum(x['here'][k] < x['lo'] for x in R)}"
    print(line)
