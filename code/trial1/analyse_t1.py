import json, glob, numpy as np, sys
R = []
for f in sorted(glob.glob(sys.argv[1] + '/t1_part*.json')): R += json.load(open(f))
print('records', len(R))
LN2 = np.log(2)
for eps in sorted({r['eps'] for r in R}):
    rs = [r for r in R if r['eps'] == eps]
    st = {}
    for r in rs: st[r['status']] = st.get(r['status'], 0) + 1
    print('\n== eps', eps, 'n', len(rs), st)
    ok = [r for r in rs if r['status'] == 'ok']
    if not ok:
        fv = [r['float_std'] for r in rs]; print('  float_std range', min(fv), max(fv)); continue
    for coh in ('A', 'B', 'all'):
        s = [r for r in ok if coh == 'all' or r['cohort'] == coh]
        over = np.array([r['float_minus_hi'] for r in s])        # >0: float PROVABLY above exact value
        gapA = np.array([r['float_std'] - r['A_routeA'] for r in s])  # float_std vs paper route A
        loss = np.array([r['A_routeA'] - r['lo'] for r in s])    # how much lower the eps-candidate's exact bound is
        rate_f = np.array([r['float_std'] / LN2 - r['leak'] for r in s])
        rate_A = np.array([r['A_routeA'] / LN2 - r['leak'] for r in s])
        rate_x = np.array([r['hi'] / LN2 - r['leak'] for r in s])   # exact value of this candidate (upper end)
        flips = int(((rate_f > 0) & (rate_x <= 0)).sum())
        print(f'  [{coh}] n={len(s)} proven overstatement: {int((over>0).sum())}/{len(s)}; max {over.max():.3e} median {np.median(over):.3e} nats;'
              f' float_std > routeA bound in {int((gapA>0).sum())}; max(float_std-routeA) {gapA.max():.3e};'
              f' rate>0 claimed by float_std but exact(hi)<=0: {flips}; float_std positive {int((rate_f>0).sum())}, routeA positive {int((rate_A>0).sum())}')
        if flips:
            for r, a, b in zip(s, rate_f, rate_x):
                if a > 0 and b <= 0: print('     flip i=%d float_rate=%.3e exact_hi_rate=%.3e routeA_rate=%.3e' % (r['i'], a, b, r['A_routeA'] / LN2 - r['leak']))
