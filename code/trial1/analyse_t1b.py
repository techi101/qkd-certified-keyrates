import json, glob, numpy as np
R = []
for f in sorted(glob.glob('out/t1_part*.json')): R += json.load(open(f))
L2 = np.log(2)
for eps in (1e-14, 1e-12, 1e-10, 1e-8):
    s = [r for r in R if r['eps'] == eps and r['status'] == 'ok']
    over = np.array([r['float_minus_hi'] for r in s]); rf = np.array([r['float_std']/L2 - r['leak'] for r in s])
    rA = np.array([r['A_routeA']/L2 - r['leak'] for r in s])
    width = np.array([r['hi'] - r['lo'] for r in s])
    print(f"eps={eps:g}: overstate>1e-6 nats: {(over>1e-6).sum()}/120; overstate(bits)>|float rate|: {((over/L2)>np.abs(rf)).sum()};"
          f" float_rate>0 & routeA_rate<=0: {((rf>0)&(rA<=0)).sum()}; max enclosure width {width.max():.2e}; "
          f"float_std below routeA by median {np.median(np.array([r['A_routeA']-r['float_std'] for r in s])):.2e}; mean time {np.mean([r['s'] for r in s]):.2f}s; precs {sorted(set(r['prec'] for r in s))}")
